"""Run the bounded pilot dependency graph, with at most three rented GPUs.

Launch under systemd. No automatic full training or contest evaluations follow.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))
from cloud import STATE, write


def execute(role, stamp):
    name = role + '-' + stamp
    log = STATE / (name + '.controller.log')
    STATE.mkdir(parents=True, exist_ok=True)
    with log.open('w') as out:
        result = subprocess.run([sys.executable, str(ROOT / 'experiments/cloud.py'), 'run',
            '--name', name, '--entry', 'experiments/run-' + role + '.sh', '--hours', '6'],
            stdout=out, stderr=subprocess.STDOUT, cwd=ROOT)
    ledger = json.loads((STATE / 'ledger.json').read_text())
    job = ledger['jobs'].get(name, {})
    finished = job.get('state') == 'complete' or (job.get('state') == 'delete_requested' and job.get('release_reason') == 'complete')
    success = result.returncode == 0 and finished and job.get('backup_verified')
    return {'role': role, 'name': name, 'success': bool(success), 'returncode': result.returncode,
            'output': str(STATE / name / 'output'), 'cleanup_confirmed': bool(job.get('released_at'))}


def main():
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    status = {'started_at': stamp, 'budget_ceiling_usd': 200, 'max_concurrent_gpus': 3,
              'state': 'running', 'jobs': [], 'scope': 'bounded pilots, no automatic scale-up'}
    status_path = ROOT / 'lab/experiments/campaign.json'
    write(status_path, status)
    with ThreadPoolExecutor(max_workers=3) as pool:
        bootstrap = pool.submit(execute, 'bootstrap', stamp)
        controls = pool.submit(execute, 'rl-controls', stamp)
        initial = bootstrap.result()
        status['jobs'].append(initial)
        write(status_path, status)
        pending = []
        if initial['success']:
            shared = Path(initial['output']) / 'shared'
            destination = ROOT / 'lab/experiments/data/shared'
            if destination.exists():
                raise RuntimeError('Shared artifacts already exist; do not overwrite previous campaign')
            shutil.copytree(shared, destination)
            pending += [pool.submit(execute, role, stamp) for role in ['sft', 'dpo']]
        else:
            status['dependent_jobs'] = 'not launched: bootstrap failed; inspect controller and runner logs'
        control_result = controls.result()
        status['jobs'].append(control_result)
        if initial['success'] and control_result['success']:
            pending += [pool.submit(execute, role, stamp) for role in ['rl-shaping', 'rl-preserve']]
        elif not control_result['success']:
            status['rl_dependent_jobs'] = 'not launched: RL controls did not complete successfully'
        write(status_path, status)
        for future in as_completed(pending):
            status['jobs'].append(future.result())
            write(status_path, status)
    status['state'] = 'completed' if len(status['jobs']) == 6 and all(j['success'] for j in status['jobs']) else 'needs_review'
    write(status_path, status)
    subprocess.run([sys.executable, str(ROOT / 'experiments/summarize.py'), '--campaign', str(status_path)], cwd=ROOT, check=False)
    print(json.dumps(status, indent=2))


if __name__ == '__main__':
    main()
