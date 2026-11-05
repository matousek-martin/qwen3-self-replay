"""Prepare, stage, and launch the scoped experiment through the canonical allocator.

prepare is read/CPU-only. stage creates an immutable payload. run never changes
authorization, raises spending ceilings, or re-rents an existing job.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'ops'))
from experiments import cloud
from experiments.shorterbetter.hardware import SCOPE, ENTRY, GPU_RATE_CEILINGS
from experiments.shorterbetter.stopping import DEFAULTS
from experiments.shorterbetter.train import check_data
from isolated_cloud import seal, verify

BASE_PAYLOAD = ROOT / 'lab/isolated/shorterbetter-retry4-20261027T075430Z'


def prepare(directory, gpu, count, phase='benchmark', cap=35.):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    p = json.loads((ROOT / 'studies/shorterbetter/protocol.json').read_text())
    p.update(id='shorterbetter-scaled-v1', status='prepared_not_locked', paid_launch_ready=False,
             question='Does SOL training improve our model, with registered operational early stopping?')
    p['arms'] = {'shorterbetter': p['arms']['shorterbetter']}
    p['seeds'] = [17]
    rollouts = 7 if phase == 'benchmark' else 300
    p['training'].update(parameter_dtype='float32', world_size_initial=count, rollout_batches=rollouts,
        optimizer_updates_total=4*rollouts, training_completions_per_arm=512*rollouts,
        backend='Pinned TRL 0.24.0 DDP + per-rank vLLM TP=1; FP32 Adam / BF16 computation')
    p['data']['monitor_subset_questions'] = 128
    check_data(ROOT / p['data']['directory'])
    p['monitoring']['every_rollout_batches'] = 7 if phase == 'benchmark' else 10
    p['monitoring']['divergence'] = 'See registered stopping rules; nonfinite parameters/gradients, wrong dtype or rank divergence fail immediately.'
    p['stopping'] = copy.deepcopy(DEFAULTS)
    p['execution'] = {'scaled': True, 'phase': phase, 'world_size': count, 'gpu': gpu,
        'vllm_gpu_memory_utilization': .45, 'eval_batch_size': 8, 'save_total_limit': 3,
        'training_walltime_target_hours': 8}
    p['authorization'] = {'preparation': True, 'accelerated_hardware_authorized': True,
        'telegram_progress_authorized': True, 'automatic_stopping_authorized': True,
        'new_paid_jobs_started': False, 'existing_project_ceiling_usd': cloud.CEILING}
    p['budget'] = {'proposed_total_pilot_cap_usd': cap, 'cap_authorized': False}
    p['phases'] = ['Fresh pinned model; verify memory, sampling, FP32 updates and rank agreement.',
        'Baseline + 7 full rollout batches for throughput, or registered 300-rollout training with stopping.',
        'Preserve final and best eligible checkpoint; verify archive and release; notify Telegram.']
    p['selection'] = {'rule': 'Best development-eligible checkpoint, preserved separately; thresholds in stopping.',
        'warning': 'Repeated monitoring and adaptive stopping are not final confirmation or a comparison to a control arm.'}
    cloud.write(directory / 'protocol.json', p)
    evidence = {k: False for k in ('data_audit', 'reward_parity', 'backend_cpu_parity',
        'distributed_cpu_check', 'stopping_tests', 'backup_release_integration', 'budget_reservation',
        'pilot_cap_enforced', 'trainer_gpu_smoke', 'distributed_gpu_check', 'resume_equivalence',
        'export_reload', 'matched_validation', 'cost_projection')}
    evidence['protocol_sha256'] = hashlib.sha256((directory/'protocol.json').read_bytes()).hexdigest()
    cloud.write(directory/'readiness.json', evidence)
    cloud.write(directory/'plan.json', {'status': 'prepared_not_launched', 'gpu': gpu, 'gpu_count': count,
        'phase': phase, 'proposed_cap_usd': cap, 'telegram_authorized': True,
        'hardware_scope': SCOPE, 'entry': ENTRY, 'prepared_at': datetime.now(timezone.utc).isoformat()})


def stage(directory, release):
    directory, release = Path(directory).resolve(), Path(release).resolve()
    evidence = json.loads((directory/'readiness.json').read_text())
    proof = ROOT/evidence['cpu_verification_path']
    if hashlib.sha256(proof.read_bytes()).hexdigest() != evidence['cpu_verification_sha256']:
        raise ValueError('CPU verification artifact changed')
    for name, expected in json.loads(proof.read_text())['source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected:
            raise ValueError('Source changed since CPU verification: ' + name)
    if release.exists():
        raise ValueError('Use a new immutable release directory')
    release.mkdir(parents=True)
    for folder in ('experiments', 'ThinkingCap-Test'):
        shutil.copytree(ROOT/folder, release/folder,
            ignore=shutil.ignore_patterns('.git', '__pycache__', '.venv*', 'runs', '.env*', '*.log', 'launch'))
    target = release/'experiments/shorterbetter/launch'
    target.mkdir()
    for name in ('protocol.json', 'readiness.json'):
        shutil.copy2(directory/name, target/name)
    shutil.copy2(proof, target/'cpu-verification.json')
    (release/'lab/experiments').mkdir(parents=True)
    # Existing artifacts are already immutable, public wheels + audited train/monitor data.
    shutil.copytree(BASE_PAYLOAD/'lab/experiments/data', release/'lab/experiments/data', copy_function=os.link)
    shutil.copy2(BASE_PAYLOAD/'lab/gpu-requirements.lock', release/'lab/gpu-requirements.lock')
    digest = seal(release)
    plan = json.loads((directory/'plan.json').read_text())
    cloud.write(directory/'launch.json', {**plan, 'release': str(release), 'payload_sha256': digest,
        'job_name': 'shorterbetter-scaled-' + directory.name.lower().replace('_','-'),
        'hours': 2 if plan.get('completion_budget_scope') else .5, 'max_cost_usd': plan['proposed_cap_usd']})


def run(directory):
    directory = Path(directory).resolve()
    launch = json.loads((directory/'launch.json').read_text())
    release, name = Path(launch['release']), launch['job_name']
    if verify(release) != launch['payload_sha256']:
        raise ValueError('Payload hash mismatch')
    p = json.loads((release/'experiments/shorterbetter/launch/protocol.json').read_text())
    if p['status'] != 'locked' or p['budget']['cap_authorized'] is not True:
        raise RuntimeError('This draft has no authorized spending envelope; prepare a new locked release after approval')
    state = json.loads((cloud.STATE/'ledger.json').read_text())
    old = state['jobs'].get(name)
    if old:
        if old.get('released_at'):
            return
        if not old.get('pod_id') or not old.get('identity_confirmed'):
            raise RuntimeError('Reconcile existing allocation; refusing duplicate rental')
        cloud.watch(name)
        return
    from experiments.shorterbetter.budget import ticket
    authorized = ticket(cloud.STATE, name, {'gpu': launch['gpu'], 'gpu_count': launch['gpu_count'],
        'hardware_scope': SCOPE, 'entry': ENTRY, 'max_cost_usd': launch['max_cost_usd'],
        'completion_budget_scope': launch.get('completion_budget_scope'),
        'provider': launch.get('provider', 'runpod'), 'offer_id': launch.get('offer_id'),
        'disk_gb': launch.get('disk_gb', 80), 'max_runtime_hours': launch.get('max_runtime_hours')})
    if launch.get('provider') == 'vast' and not authorized:
        raise RuntimeError('Vast launch requires its own named bounded authorization')
    if launch['max_cost_usd'] is None and not authorized:
        raise RuntimeError('Missing named authorization for completion-driven training')
    if not authorized and cloud.liability(state['jobs']) + launch['max_cost_usd'] > cloud.USABLE:
        raise RuntimeError('Shared project spending ceiling has insufficient remaining budget')
    registry = ROOT/'lab/experiments/shorterbetter-scaled/notifications/jobs.json'
    registry.parent.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (registry.parent/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = json.loads(registry.read_text()) if registry.exists() else {}
        rows[name] = {'telegram_authorized': True, 'directory': str(directory)}
        cloud.write(registry, rows)
    command = [sys.executable, str(ROOT/'ops/isolated_cloud.py'), 'run', '--release', str(release),
        '--name', name, '--entry', ENTRY, '--gpu', launch['gpu'], '--gpu-count', str(launch['gpu_count']),
        '--hardware-scope', SCOPE, '--hours', str(launch['hours']), '--data-upload-timeout', '900']
    if launch.get('completion_budget_scope'):
        command += ['--completion-budget-scope', launch['completion_budget_scope'], '--direct-pinned-wheels']
    else:
        hours = launch['max_runtime_hours'] if launch.get('provider') == 'vast' else (2 if launch['phase'] == 'benchmark' else 8)
        command += ['--max-cost-usd', str(launch['max_cost_usd']), '--max-runtime-hours', str(hours)]
    if launch.get('provider') == 'vast':
        command += ['--provider', 'vast', '--vast-offer-id', str(launch['offer_id']),
                    '--vast-disk-gb', str(launch.get('disk_gb', 80)), '--direct-pinned-wheels']
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare','stage','run'])
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--gpu', choices=GPU_RATE_CEILINGS, default='NVIDIA A100-SXM4-80GB')
    parser.add_argument('--gpu-count', type=int, choices=(1,2,4,8), default=4)
    parser.add_argument('--phase', choices=('benchmark','training'), default='benchmark')
    parser.add_argument('--cap', type=float, default=35.)
    parser.add_argument('--release', type=Path)
    args = parser.parse_args()
    if args.action == 'prepare': prepare(args.directory, args.gpu, args.gpu_count, args.phase, args.cap)
    elif args.action == 'stage':
        if args.release is None: parser.error('stage needs --release')
        stage(args.directory, args.release)
    else: run(args.directory)


if __name__ == '__main__': main()
