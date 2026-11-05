"""Registered continuation of one completed, backed-up full-run checkpoint."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from experiments.shorterbetter.distributed import atomic_json
from experiments.shorterbetter.prepare import sha
from .full_run import validate

ROOT = Path(__file__).resolve().parents[3]


def scientific_protocol(protocol):
    p = copy.deepcopy(protocol)
    for key in ('id', 'scope', 'authorization', 'budget', 'lineage', 'phases', 'continuation_parent'):
        p.pop(key, None)
    p['execution'].pop('full_run_continuation', None)
    p['stopping'].pop('quality_stopping_enabled', None)
    return p


def validate_lineage(protocol, prior, state, root):
    validate(protocol)
    parent = protocol['continuation_parent']
    source = Path(root) / parent['directory']
    if source.resolve().parent != (Path(root)/'data').resolve():
        raise ValueError('Continuation source must be a registered data payload')
    original_path = source/'protocol.json'
    original = json.loads(original_path.read_text())
    if (not protocol['execution'].get('full_run_continuation')
            or protocol['stopping'].get('quality_stopping_enabled') is not False
            or not protocol['authorization'].get('quality_stopping_disabled_by_user')
            or sha(original_path) != parent['protocol_sha256']
            or prior['protocol_sha256'] != parent['protocol_sha256']
            or prior['job_name'] != parent['job_name']
            or prior['data_manifest_sha256'] != protocol['data']['manifest_sha256']
            or state['global_step'] != parent['optimizer_step']
            or not 0 < state['global_step'] < 1200 or state['global_step'] % 40
            or scientific_protocol(protocol) != scientific_protocol(original)):
        raise ValueError('Continuation changes scientific protocol or checkpoint lineage')
    return source


def verify_source(protocol, root):
    parent = protocol['continuation_parent']
    source = Path(root)/parent['directory']
    manifest = source/'files.json'
    if sha(manifest) != parent['files_sha256']:
        raise ValueError('Continuation manifest hash mismatch')
    files = json.loads(manifest.read_text())
    checkpoint = f"training/checkpoint-{parent['optimizer_step']}"
    required = {f'{checkpoint}/{name}' for name in ('model.safetensors','optimizer.pt','scheduler.pt',
                'trainer_state.json',*[f'rng_state_{i}.pth' for i in range(4)])}
    required |= {'training/run-provenance.json','protocol.json','readiness.json'}
    if not required <= set(files):
        raise ValueError('Incomplete registered continuation manifest')
    for name, digest in files.items():
        path = source/name
        if not path.resolve().is_relative_to(source.resolve()) or sha(path) != digest:
            raise ValueError('Continuation source file mismatch: '+name)
    prior = json.loads((source/'training/run-provenance.json').read_text())
    state = json.loads((source/checkpoint/'trainer_state.json').read_text())
    validate_lineage(protocol, prior, state, root)
    return source


def replace_baseline(training, baseline, world=4):
    # Preserve the historical archive; future comparisons use a pinned fresh
    # baseline actually evaluated on the new physical worker, with provenance.
    for rank in range(world):
        relative = f'rank-{rank:03d}/rollouts.jsonl'
        new = [json.loads(s) for s in (baseline/relative).read_text().splitlines()]
        if len(new) != 128 or any(r['optimizer_step'] != 0 or r['split'] != 'development' for r in new):
            raise RuntimeError('Incomplete same-worker continuation baseline')
        path = training/relative
        old = [json.loads(s) for s in path.read_text().splitlines()]
        kept = [r for r in old if not (r['optimizer_step'] == 0 and r['split'] == 'development')]
        path.write_text(''.join(json.dumps(r)+'\n' for r in new+kept))


def run(protocol, path, lease, evidence):
    source = verify_source(protocol, ROOT)
    if lease.get('provider') != 'vast' or lease['gpu_count'] != 4 or lease['gpu'] != protocol['execution']['gpu']:
        raise RuntimeError('Continuation requires its registered four-A100 Vast lease')
    # Resume equivalence/export were measured on the original job. Record reuse
    # explicitly; all actual-device checks and the fresh baseline run below.
    previous = json.loads((source/'readiness.json').read_text())
    for key in ('trainer_gpu_smoke','distributed_gpu_check','resume_equivalence',
                'export_reload','matched_validation','cost_projection'):
        if previous.get(key) is not True:
            raise ValueError('Missing verified parent evidence: '+key)
        evidence[key] = True
    evidence['continuation_parent_evidence'] = protocol['continuation_parent']
    ready = ROOT/'output/continuation-readiness.json'
    atomic_json(ready, evidence)
    baseline = ROOT/'output/continuation-baseline'
    atomic_json(ROOT/'output/phase.json', {'phase':'continuation_baseline', 'time':time.time()})
    command = [sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1',
               '--nproc_per_node=4','--max_restarts=0','-m','experiments.shorterbetter.train',
               '--protocol',str(path),'--arm','shorterbetter','--seed','17','--execute',
               '--readiness',str(ready),'--job-name',lease['job_name'],
               '--out',str(baseline),'--baseline-check']
    subprocess.run(command, check=True)
    for rank in range(4):
        for filename in ('gpu-memory-check.json','gpu-sampling-check.json'):
            if json.loads((baseline/f'rank-{rank:03d}'/filename).read_text())['status'] != 'passed':
                raise RuntimeError('New-worker GPU check failed')
    from experiments.shorterbetter.launch_checks import seed_resume
    training = ROOT/'output/training'
    step = protocol['continuation_parent']['optimizer_step']
    # Loading never modifies checkpoint files; later saves use new step paths.
    seed_resume(source/'training', training, 4, step=step, copy_function=os.link)
    if shutil.disk_usage(ROOT).free < 24*2**30:
        raise RuntimeError('Insufficient free disk for next checkpoint, best copy and final export')
    replace_baseline(training, baseline)
    atomic_json(training/'continuation-provenance.json', {
        'parent':protocol['continuation_parent'], 'new_job_name':lease['job_name'],
        'same_worker_baseline':str(baseline.relative_to(ROOT/'output')),
        'historical_validation_through_step':step,
        'historical_points_from_previous_worker':True,
        'quality_stopping_enabled':False, 'time':time.time()})
    atomic_json(training/'progress.json', {'optimizer_step':step,'rollout_batches':step/4,
                                          'updated_at':time.time(),'elapsed_seconds':0})
    if (ROOT/'output/stop-request.json').exists():
        raise RuntimeError('Stop requested before continuation')
    from .pilot import run as train
    train(protocol, path, lease, evidence, resume=training/f'checkpoint-{step}')
