"""One registered 300-rollout run, gated by same-worker deterministic resume."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from experiments.shorterbetter.config import assert_launch_ready
from experiments.shorterbetter.distributed import atomic_json
from experiments.shorterbetter.prepare import sha

ROOT = Path(__file__).resolve().parents[3]


def validate(protocol):
    e, t, d = (protocol[k] for k in ('execution', 'training', 'data'))
    verification = e.get('multidomain_full_verification') is True
    paired = e.get('paired_short_experiment') is True
    expected = 2 if verification else (30 if paired else 300)
    if paired and (protocol['arms']['shorterbetter'] not in
                   ({'alpha':2.0,'length_beta':0.001},{'alpha':5.0,'length_beta':0.001})
                   or protocol['reward']['alpha'] != protocol['arms']['shorterbetter']['alpha']
                   or e.get('preserve_every_rollouts') != 5
                   or protocol['stopping'].get('quality_stopping_enabled') is not False):
        raise ValueError('Unregistered paired ShorterBetter experiment')
    if (not e.get('scaled') or not e.get('multidomain')
            or (verification and e.get('multidomain_full_training'))
            or e.get('gpu') != 'NVIDIA A100-SXM4-80GB' or e.get('world_size') != 4
            or e.get('phase') != ('benchmark' if verification else 'training')
            or e.get('hardware_comparison') or e.get('accelerated_multidomain_benchmark')
            or t.get('rollout_batches') != expected or t.get('optimizer_updates_total') != expected*4
            or t.get('prompts_per_rollout_batch') != 64 or t.get('generations_per_prompt') != 8
            or t.get('prompts_per_optimizer_minibatch') != 16 or t.get('max_completion_tokens') != 6000
            or t.get('parameter_dtype') != 'float32'
            or not e.get('deterministic_training') or not e.get('deterministic_ddp')
            or e.get('shutdown_reserve_seconds') != 1200 or not e.get('terminal_evaluation_required')
            or not d.get('cover_all_training_rows')
            or d.get('monitor_subset_questions') != (16 if verification else 64)
            or protocol['monitoring']['every_rollout_batches'] != (1 if verification else 10)
            or protocol['authorization'].get('full_training_authorized') is not True
            or protocol['seeds'] != [17] or set(protocol['arms']) != {'shorterbetter'}):
        raise ValueError('Unregistered full multidomain run dimensions')
    return e['gpu'], 4, expected


def technical_protocol(protocol):
    validate(protocol)
    p = copy.deepcopy(protocol)
    p['id'] += '-resume-verification'
    p['execution'].pop('multidomain_full_training')
    p['execution'].update(multidomain_full_verification=True, phase='benchmark',
                          resume_diagnostic=True, save_total_limit=2)
    if p['execution'].pop('paired_short_experiment', False):
        # The isolated two-rollout engineering test is not the main experiment.
        p['stopping']['quality_stopping_enabled'] = True
    p['training'].update(rollout_batches=2, optimizer_updates_total=8, training_completions_per_arm=1024)
    p['monitoring']['every_rollout_batches'] = 1
    # Full panel is explicitly grouped as 16 questions/domain; retain four/domain.
    ids = p['data']['monitor_ids']
    p['data'].update(monitor_subset_questions=16,
                     monitor_ids=[i for start in range(0, 64, 16) for i in ids[start:start+4]])
    validate(p)
    return p


def run(protocol, path, lease, evidence):
    validate(protocol)
    if lease.get('provider') != 'vast' or lease['gpu_count'] != 4 or lease['gpu'] != protocol['execution']['gpu']:
        raise RuntimeError('Full run requires its registered four-A100 Vast lease')
    if shutil.disk_usage(ROOT).free < 40*2**30:
        raise RuntimeError('Insufficient worker disk for complete verification and checkpoints')
    output = ROOT/'output'
    technical = technical_protocol(protocol)
    technical_path = output/'full-verification-protocol.json'
    atomic_json(technical_path, technical)
    ready = dict(evidence, protocol_sha256=sha(technical_path))
    assert_launch_ready(technical, ready)
    ready_path = output/'full-verification-readiness.json'
    atomic_json(ready_path, ready)
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
               '--nproc_per_node=4', '--max_restarts=0', '-m', 'experiments.shorterbetter.train',
               '--protocol', str(technical_path), '--arm', 'shorterbetter', '--seed', '17',
               '--execute', '--readiness', str(ready_path), '--job-name', lease['job_name']]
    reference, resumed = output/'verification-reference', output/'verification-resumed'
    atomic_json(output/'phase.json', {'phase':'deterministic_resume_reference', 'time':time.time()})
    subprocess.run(command + ['--out', str(reference)], check=True)
    from experiments.shorterbetter.launch_checks import seed_resume
    seed_resume(reference, resumed, 4, step=4)
    atomic_json(output/'phase.json', {'phase':'gpu_resume_verification', 'time':time.time()})
    subprocess.run(command + ['--out', str(resumed), '--resume', str(resumed/'checkpoint-4')], check=True)
    from experiments.shorterbetter.resume_repair import compare
    from experiments.shorterbetter.verify_cheap import optimizer_compare, compact_technical_artifacts
    result = compare(reference, resumed, 4, step=4)
    result['optimizer'] = optimizer_compare(reference, resumed)
    for folder in (reference, resumed):
        if json.loads((folder/'training-finished.json').read_text())['optimizer_steps'] != 8:
            raise RuntimeError('Full-run resume verification ended early')
    if not result['bit_identical'] or not result['optimizer']['bit_identical']:
        raise RuntimeError('Full-run deterministic resume was not bit-identical')
    curve = json.loads((reference/'learning-curve.json').read_text())
    if any([r['optimizer_step'] for r in v['checkpoints']] != [0, 4, 8]
           for v in curve['domains'].values()) or set(curve['domains']) != {'math','coding','tools','instructions'}:
        raise RuntimeError('Missing same-worker matched validation')
    atomic_json(output/'deterministic-resume-result.json', result)
    from .pilot import projection, run as train
    # Technical panel has 16 questions, so project its measured validation accordingly.
    projected_protocol = copy.deepcopy(protocol)
    projected_protocol['data']['monitor_subset_questions'] = 16
    timing = projection(reference, projected_protocol, lease)
    if timing.get('status') != 'early_measured_projection':
        raise RuntimeError('Missing full-length timing measurement')
    hours = timing['targets'][str(protocol['training']['rollout_batches'])]['hours_range'][1]
    required_end = time.time() + hours*3600
    if required_end > lease['hard_deadline']:
        raise RuntimeError('Measured full run exceeds its registered runtime/cost envelope')
    atomic_json(output/'full-run-projection.json', timing)
    evidence.update(trainer_gpu_smoke=True, distributed_gpu_check=True, resume_equivalence=True,
                    export_reload=True, matched_validation=True, cost_projection=True,
                    deterministic_resume_result=result)
    assert_launch_ready(protocol, evidence)
    compact_technical_artifacts(reference, resumed,
        extra=(reference/'checkpoint-4', reference/'model')
        if protocol['execution'].get('paired_short_experiment') else ())
    if (output/'stop-request.json').exists():
        raise RuntimeError('Stop requested during full-run verification')
    train(protocol, path, lease, evidence)
