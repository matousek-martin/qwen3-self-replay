"""Three sequential paired workers through the existing shared cloud controller.

Each worker trains both registered arms, then evaluates a fresh base and both
adapters together. This module does not implement GPU allocation or cleanup.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

from . import replay_selection as recipe
from . import study_eval

ROOT = recipe.ROOT
LOCAL = ROOT / 'lab/experiments/replay-selection-v1'
STAGED = ROOT / 'lab/experiments/data/replay-selection-v1'


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    study_eval.write_json(path, value)


def evaluation_config(protocol, seed):
    config = dict(protocol['evaluation']['config'])
    config['candidates'] = [name.format(seed=seed) for name in config['candidates']]
    return config


def training_status(path, protocol):
    status = json.loads(Path(path).read_text())
    if (status.get('status') != 'completed'
            or status.get('completed_groups') != protocol['training']['max_steps']
            or status.get('optimizer_updates', 0) < 1):
        raise RuntimeError('Training did not complete the registered group budget')
    return status


def verify_sources(launch):
    for path, expected in launch['source_sha256'].items():
        if digest(ROOT / path) != expected:
            raise RuntimeError('Launch source changed: ' + path)


def read_launch(staged=STAGED):
    launch = json.loads((staged / 'launch.json').read_text())
    if digest(staged / 'protocol.json') != launch['protocol_sha256']:
        raise RuntimeError('Launch protocol mismatch')
    verify_sources(launch)
    return launch


def prepare():
    if (LOCAL / 'campaign.json').exists() or (STAGED / 'launch.json').exists():
        raise RuntimeError('Launch already registered; inspect/resume the existing campaign')
    protocol = recipe.load_protocol()
    study_eval.load_selection(ROOT / 'ThinkingCap-Test/questions',
                             ROOT / 'lab/experiments/data/study/splits.json', 'tuning')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    STAGED.mkdir(parents=True, exist_ok=True)
    for name in ('protocol.json', 'registration.json'):
        shutil.copyfile(LOCAL / name, STAGED / name)
    entries = {}
    for seed in protocol['training_seeds']:
        entry = f'experiments/run-replay-selection-seed{seed}.sh'
        Path(ROOT / entry).write_text('#!/usr/bin/env bash\nset -euo pipefail\n'
            'source /workspace/project/experiments/setup_gpu.sh\n'
            'export NLTK_DATA=/root/thinkingcap_nltk_data\n'
            'export OPENBLAS_NUM_THREADS=1\nexport OMP_NUM_THREADS=1\n'
            'mkdir -p -m 700 "$NLTK_DATA"\nchmod 700 "$NLTK_DATA"\n'
            f'python -m experiments.replay_selection_campaign worker --seed {seed}\n')
        entries[str(seed)] = entry
    paths = [ROOT / f for f in [
        'experiments/replay_selection.py', 'experiments/replay_selection_campaign.py',
        'experiments/cloud.py', 'experiments/setup_gpu.sh', 'experiments/gpu_policy.py',
        'experiments/worker_monitor.py', 'experiments/study_eval.py', 'experiments/pilot.py',
        'experiments/vllm_compat.py', 'experiments/vllm_lora_smoke.py',
        'experiments/offline_training/generate.py', 'experiments/offline_training/score.py',
        'experiments/offline_training/common.py']]
    paths += sorted((ROOT / 'experiments/rl_training').glob('*.py'))
    paths += [ROOT / e for e in entries.values()]
    paths += [p for p in (ROOT / 'ThinkingCap-Test').rglob('*.py')
              if not any(part.startswith(('.venv', '.git')) for part in p.parts)]
    paths += [ROOT / 'ThinkingCap-Test/questions/manifest.json',
              ROOT / 'ThinkingCap-Test/questions/questions.jsonl']
    launch = {'schema_version': 1, 'registered_at': now(), 'stamp': stamp,
              'protocol_sha256': digest(STAGED / 'protocol.json'),
              'source_sha256': {str(p.relative_to(ROOT)): digest(p) for p in sorted(set(paths))},
              'entries': entries, 'seeds': protocol['training_seeds'],
              'initial_reservation_hours_per_worker': 8,
              'initial_reservation_usd_per_worker': 10.4,
              'planning_range_usd': [20, 35],
              'budget': 'Shared existing $200 cap/$180 usable; sequential reservations renewed by cloud controller, no new per-window budget.',
              'authorization': {'session': '01a173bb-3140-7e32-b592-c68da5d288d3',
                                'user_message': 'muzem spustit?', 'scope': 'Launch registered replay-selection v1 within existing shared cap; no teacher/adaptive paid runs'},
              'worker': 'fresh control + mixed training, validation, fresh base + two adapters paired evaluation per seed on one worker'}
    write(STAGED / 'launch.json', launch)
    write(LOCAL / 'launch.json', launch)
    state = {'state': 'prepared', 'created_at': now(), 'launch_sha256': digest(STAGED / 'launch.json'),
             'trials': {str(seed): {'state': 'queued', 'job': f'replay-selection-seed{seed}-{stamp}'}
                        for seed in protocol['training_seeds']}}
    write(LOCAL / 'campaign.json', state)
    print(json.dumps({'state':'prepared','jobs':state['trials'],'launch_sha256':state['launch_sha256']}, indent=2))


def worker(seed):
    data, output = ROOT / 'data', ROOT / 'output'
    staged = data / 'replay-selection-v1'
    launch = read_launch(staged)
    protocol = recipe.load_protocol(staged / 'protocol.json', worker=True)
    if seed not in launch['seeds']:
        raise ValueError('Unregistered seed')
    from .gpu_policy import verify_gpu_names
    gpu_names = subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'], text=True)
    verify_gpu_names(gpu_names)
    gpu_identity = subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv,noheader'], text=True).strip()
    write(output / 'replay-selection-provenance.json', {'launch': launch, 'protocol': protocol,
          'seed': seed, 'gpu_identity': gpu_identity, 'started_at': now()})
    # Tensor-only GPU diagnostics, no model downloads or optimizer training.
    subprocess.run([sys.executable, '-m', 'unittest', 'experiments.rl_training.test_torch',
                    'experiments.rl_training.test_replay_selection',
                    'experiments.rl_training.test_adaptive_replay', '-v'], cwd=ROOT, check=True)
    states = {}
    stage = output / 'evaluation-inputs'
    for arm in protocol['arms']:
        name = f'{arm}-seed{seed}'
        command = recipe.training_command(protocol, arm, seed, data, output / 'training')
        subprocess.run(command, cwd=ROOT, check=True)
        target = output / 'training' / name
        states[arm] = training_status(target / 'status.json', protocol)
        config = json.loads((target / 'config.json').read_text())
        if config['replay_admission'] != protocol['arms'][arm] or config['replay_reward'] != 'fixed':
            raise RuntimeError('Training configuration differs from registered recipe')
        from .pilot import validate
        validate(name, target / 'adapter')
        shutil.copytree(target / 'adapter', stage / 'adapters' / name)
    shutil.copyfile(data / 'study/splits.json', stage / 'splits.json')
    config_path = stage / 'evaluation.json'
    write(config_path, evaluation_config(protocol, seed))
    for command in [
        ['experiments.vllm_compat', '--out', output / 'vllm-compat.json'],
        ['experiments.vllm_lora_smoke', '--config', config_path, '--out', output / 'vllm-lora-smoke.json'],
        ['experiments.study_eval', '--config', config_path, '--out', output / 'study']]:
        subprocess.run([sys.executable, '-m', *map(str, command)], cwd=ROOT, check=True)
    result = json.loads((output / 'study/study-result.json').read_text())
    if not result['complete'] or set(result['completed_candidates']) != set(evaluation_config(protocol, seed)['candidates']):
        raise RuntimeError('Paired evaluation incomplete')
    subprocess.run([sys.executable, ROOT / 'ThinkingCap-Test/compare.py',
                    output / f'study/runs/control-seed{seed}', output / f'study/runs/mixed-seed{seed}',
                    '--out', output / 'mixed-vs-control.json'], cwd=ROOT, check=True)
    write(output / 'role-result.json', {'seed': seed, 'training': states, 'development_complete': True,
          'gpu_identity': gpu_identity, 'protocol_sha256': launch['protocol_sha256'], 'finished_at': now()})


@contextmanager
def campaign_lock():
    with (LOCAL / 'campaign.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def existing_action(job):
    if not job:
        return 'run'
    if job.get('released_at'):
        if job.get('state') == 'complete':
            return 'verify'
        raise RuntimeError('Prior unsuccessful job released; no automatic rerental')
    if not job.get('pod_id') or not job.get('identity_confirmed'):
        raise RuntimeError('Existing allocation must be reconciled; no duplicate rental')
    return 'watch'


def campaign():
    # ops/runpod and its credentials exist only on the controller, not workers.
    from . import cloud
    with campaign_lock():
        launch = read_launch()
        recipe.load_protocol()
        state = json.loads((LOCAL / 'campaign.json').read_text())
        if digest(STAGED / 'launch.json') != state['launch_sha256']:
            raise RuntimeError('Launch changed after preparation')
        try:
            state['state'] = 'running'
            write(LOCAL / 'campaign.json', state)
            for seed in launch['seeds']:
                verify_sources(launch)
                trial = state['trials'][str(seed)]
                with cloud.ledger() as ledger:
                    job = dict(ledger['jobs'].get(trial['job'], {}))
                action = existing_action(job)
                if action != 'verify':
                    trial['state'] = 'running' if action == 'watch' else 'allocating'
                    write(LOCAL / 'campaign.json', state)
                    command = [sys.executable, '-m', 'experiments.cloud', action, '--name', trial['job']]
                    if action == 'run':
                        command += ['--entry', launch['entries'][str(seed)], '--gpu', 'NVIDIA GeForce RTX 4090',
                                    '--hours', str(launch['initial_reservation_hours_per_worker'])]
                    with (LOCAL / f'seed{seed}.controller.log').open('a') as stream:
                        subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
                with cloud.ledger() as ledger:
                    job = dict(ledger['jobs'][trial['job']])
                result = cloud.verify_backup(trial['job'])
                role = json.loads((cloud.STATE / trial['job'] / 'output/role-result.json').read_text())
                if (result['exit_code'] != 0 or job.get('state') != 'complete' or not job.get('released_at')
                        or not job.get('backup_verified') or role.get('seed') != seed
                        or not role.get('development_complete') or role.get('protocol_sha256') != launch['protocol_sha256']):
                    raise RuntimeError('Paired worker requires inspection; stopping next allocations')
                trial.update(state='complete', released=True, backup_verified=True,
                             charged_estimate_usd=job['charged_estimate_usd'])
                write(LOCAL / 'campaign.json', state)
            state.update(state='development_complete', completed_at=now())
            write(LOCAL / 'campaign.json', state)
        except BaseException as exc:
            state.update(state='needs_inspection', error=str(exc), updated_at=now())
            if 'trial' in locals():
                with cloud.ledger() as ledger:
                    failed = ledger['jobs'].get(trial['job'], {})
                if failed.get('state') == 'allocation_unavailable' and failed.get('safely_no_pod'):
                    trial['state'] = 'queued_capacity'
                    state['state'] = 'queued_capacity'
                elif trial.get('state') != 'complete':
                    trial['state'] = 'needs_inspection'
            write(LOCAL / 'campaign.json', state)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('prepare')
    sub.add_parser('campaign')
    sub.add_parser('worker').add_argument('--seed', required=True, type=int, choices=(17,29,43))
    args = parser.parse_args()
    if args.command == 'prepare': prepare()
    elif args.command == 'worker': worker(args.seed)
    else: campaign()


if __name__ == '__main__':
    main()
