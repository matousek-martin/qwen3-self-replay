"""Registered nine-run replay repeatability/refinement batch.

Workers train and run paired DEVELOPMENT evaluations. No confirmation outputs
are read, and no existing study artifact or checkpoint is replaced.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / 'lab/experiments/replay-round2-4090'
PLAN = ROOT / 'lab/experiments/data/replay-round2-4090/plan.json'
SEEDS = (17, 29, 43)
RECIPES = {'control': (0.1, 0.1), 'weaker-length': (0.05, 0.1),
           'stronger-replay': (0.1, 0.2)}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def trials():
    return [{'name': f'{recipe}-seed{seed}', 'recipe': recipe, 'seed': seed,
             'length_coefficient': coefficients[0], 'replay_weight': coefficients[1]}
            for seed in SEEDS for recipe, coefficients in RECIPES.items()]


def load_plan(path):
    plan = json.loads(path.read_text())
    if plan.get('trials') != trials() or plan.get('max_parallel') != 3:
        raise ValueError('Registered trial grid or concurrency changed')
    if plan.get('gpu') != 'NVIDIA GeForce RTX 4090' or plan.get('total_campaign_ceiling_usd') != 200:
        raise ValueError('Registered hardware or campaign budget changed')
    return plan


def training_command(trial, data, output):
    return [sys.executable, '-m', 'experiments.rl_training.train',
            '--data', str(data / 'pilot.jsonl'), '--out', str(output),
            '--variant', 'replay', '--optimizer', 'dr_grpo', '--replay-reward', 'fixed',
            '--max-steps', '128', '--max-seconds', '0', '--max-new-tokens', '2048',
            '--final-cap', '1024', '--group-size', '4', '--save-every', '32',
            '--references', str(data / 'shared/references.json'),
            '--seed', str(trial['seed']), '--length-coefficient', str(trial['length_coefficient']),
            '--replay-weight', str(trial['replay_weight'])]


def initialize():
    if PLAN.exists() or (LOCAL / 'campaign.json').exists():
        raise RuntimeError('Round-two registration already exists; do not replace it')
    sys.path.insert(0, str(ROOT / 'experiments'))
    import cloud
    ledger = json.loads((cloud.STATE / 'ledger.json').read_text())
    spent = cloud.liability(ledger['jobs'])
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    data = ROOT / 'lab/experiments/data'
    plan = {'schema_version': 1, 'registered_at': datetime.now(timezone.utc).isoformat(),
            'stamp': stamp, 'trials': trials(), 'max_parallel': 3, 'gpu': 'NVIDIA GeForce RTX 4090',
            'training_groups': 128, 'group_size': 4,
            'development_questions': 512, 'development_samples': 2,
            'evaluation': 'Fresh matched base plus trained adapter for each trial; original tuning subset only.',
            'confirmation': 'Not part of these nine jobs. Requires a fresh disjoint cohort and a separately locked winner; do not reuse prior confirmation as untouched evidence.',
            'selection': 'Report all seeds and domains. Prefer repeatable thinking-token savings with small accuracy loss; do not select a lucky seed or claim zero regression from nonsignificance.',
            'total_campaign_ceiling_usd': 200, 'conservative_spend_before_batch_usd': spent,
            'nine_run_planning_range_usd': [30, 40], 'future_confirmation_allowance_usd': 10,
            'initial_reservation_hours_per_job': 4,
            'reservation_policy': 'Hours reserve budget, not a runtime cutoff. Renew while alive under the shared campaign ceiling.',
            'cost_basis': 'Prior L4 replay training+validation rental 1.491 hours, development batch of base+three adapters 1.638 hours. Nine paired trials extrapolate to about 20.8 GPU-hours at the conservative 1.30 USD/hour ledger rate, about 27 USD before contingency. The 30-40 USD planning range is an estimate, not a hard per-batch limit or invoice.',
            'data_sha256': {rel: digest(data / rel) for rel in
                            ['pilot.jsonl', 'val-pilot.jsonl', 'shared/references.json', 'study/splits.json']}}
    write(PLAN, plan)
    write(LOCAL / 'registration.json', {'plan_sha256': digest(PLAN), 'plan': plan})
    for trial in plan['trials']:
        entry = ROOT / ('experiments/run-replay-round2-' + trial['name'] + '.sh')
        entry.write_text('#!/usr/bin/env bash\nset -euo pipefail\n'
                         'source /workspace/project/experiments/setup_gpu.sh\n'
                         'export NLTK_DATA=/root/thinkingcap_nltk_data\n'
                         'export OPENBLAS_NUM_THREADS=1\nexport OMP_NUM_THREADS=1\n'
                         'mkdir -p -m 700 "$NLTK_DATA"\nchmod 700 "$NLTK_DATA"\n'
                         'python -m experiments.replay_round2 worker --trial ' + trial['name'] + '\n')
    print(json.dumps({'plan': str(PLAN), 'trials': len(plan['trials']), 'remaining_before_batch_usd': 200-spent}))


def worker(name):
    data, output = ROOT / 'data', ROOT / 'output'
    plan = load_plan(data / 'replay-round2-4090/plan.json')
    trial = next(t for t in plan['trials'] if t['name'] == name)
    for rel, expected in plan['data_sha256'].items():
        if digest(data / rel) != expected:
            raise ValueError('Registered data changed: ' + rel)
    write(output / 'round2-registration.json', {'plan': plan, 'trial': trial,
                                              'plan_sha256': digest(data / 'replay-round2-4090/plan.json')})
    target = output / 'training' / name
    subprocess.run(training_command(trial, data, target), cwd=ROOT, check=True)
    status = json.loads((target / 'status.json').read_text())
    if status['status'] != 'completed' or status['completed_groups'] != 128 or status['optimizer_updates'] < 1:
        raise RuntimeError('Training did not complete its registered budget')
    from experiments.pilot import validate
    validate(name, target / 'adapter')
    stage = output / 'evaluation-inputs'
    shutil.copytree(target / 'adapter', stage / 'adapters' / name)
    shutil.copyfile(data / 'study/splits.json', stage / 'splits.json')
    config = stage / 'evaluation.json'
    write(config, {'schema_version': 1, 'split_manifest': 'splits.json', 'subset': 'tuning',
                   'candidates': ['base', name], 'samples': 2, 'workers': 4,
                   'candidate_timeout_seconds': 0})
    commands = [
        ['experiments.vllm_compat', '--out', output / 'vllm-compat.json'],
        ['experiments.vllm_lora_smoke', '--config', config, '--out', output / 'vllm-lora-smoke.json'],
        ['experiments.study_eval', '--config', config, '--out', output / 'study']]
    for command in commands:
        subprocess.run([sys.executable, '-m', *map(str, command)], cwd=ROOT, check=True)
    write(output / 'role-result.json', {'trial': trial, 'training': status,
          'development_complete': True, 'scope': 'Round-two exploratory development; not fresh confirmation'})


def campaign():
    sys.path.insert(0, str(ROOT / 'experiments'))
    import cloud
    plan = load_plan(PLAN)
    registration = json.loads((LOCAL / 'registration.json').read_text())
    if registration['plan_sha256'] != digest(PLAN):
        raise ValueError('Plan changed after registration')
    recovery_path = LOCAL / 'recovery.json'
    recovery = json.loads(recovery_path.read_text()) if recovery_path.exists() else {}
    if recovery and recovery.get('plan_sha256') != digest(PLAN):
        raise ValueError('Recovery map belongs to a different plan')
    path = LOCAL / 'campaign.json'
    state = json.loads(path.read_text()) if path.exists() else {
        'state': 'running', 'plan_sha256': digest(PLAN),
        'trials': {t['name']: {'state': 'queued', 'job': 'replay2-' + t['name'] + '-' + plan['stamp']}
                   for t in plan['trials']}}
    state['state'] = 'running'
    write(path, state)
    stop = threading.Event()

    def execute(trial):
        name = trial['name']
        jobname = 'replay2-' + name + '-' + plan['stamp']
        replacement = recovery.get('replacements', {}).get(name)
        if replacement:
            if replacement.get('original_job') != jobname or replacement.get('replacement_job') != jobname + '-retry1':
                raise ValueError('Unsafe recovery job identity')
            with cloud.ledger() as ledger:
                original = dict(ledger['jobs'].get(jobname, {}))
            if not original.get('released_at') or original.get('state') == 'complete':
                raise ValueError('Recovery requires a released unsuccessful original job')
            jobname = replacement['replacement_job']
        if stop.is_set():
            return name, {'state': 'queued', 'job': jobname, 'reason': 'Earlier trial needs inspection'}
        with cloud.ledger() as ledger:
            prior = dict(ledger['jobs'].get(jobname, {}))
        if prior.get('released_at'):
            code = 0 if prior['state'] == 'complete' else 1
        else:
            if prior:
                if not prior.get('pod_id') or not prior.get('identity_confirmed'):
                    stop.set()
                    return name, {'state': 'needs_inspection', 'job': jobname, 'reason': 'Existing allocation must be reconciled; no blind retry'}
                command = ['watch', '--name', jobname]
            else:
                command = ['run', '--name', jobname, '--entry', 'experiments/run-replay-round2-' + name + '.sh',
                           '--hours', str(plan['initial_reservation_hours_per_job']), '--gpu', plan['gpu']]
            with (LOCAL / (name + '.controller.log')).open('a') as log:
                code = subprocess.run([sys.executable, str(ROOT / 'experiments/cloud.py'), *command],
                                      cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
        with cloud.ledger() as ledger:
            job = dict(ledger['jobs'].get(jobname, {}))
        success = code == 0 and job.get('state') == 'complete' and bool(job.get('released_at'))
        if success:
            try:
                archive = cloud.verify_backup(jobname)
                role = json.loads((cloud.STATE / jobname / 'output/role-result.json').read_text())
                success = archive['exit_code'] == 0 and role['trial'] == trial and role['development_complete']
            except Exception as exc:
                success = False
                job['inspection_error'] = str(exc)
        if not success:
            stop.set()
        return name, {'state': 'complete' if success else 'needs_inspection', 'job': jobname,
                      'released': bool(job.get('released_at')), 'error': job.get('inspection_error') or job.get('error')}

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(execute, trial) for trial in plan['trials']]
        for future in as_completed(futures):
            name, result = future.result()
            state['trials'][name] = result
            write(path, state)
    state['state'] = 'development_complete' if all(r['state'] == 'complete' for r in state['trials'].values()) else 'needs_inspection'
    write(path, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('init')
    sub.add_parser('campaign')
    remote = sub.add_parser('worker')
    remote.add_argument('--trial', required=True, choices=[t['name'] for t in trials()])
    args = parser.parse_args()
    if args.command == 'init':
        initialize()
    elif args.command == 'worker':
        worker(args.trial)
    else:
        campaign()


if __name__ == '__main__':
    main()
