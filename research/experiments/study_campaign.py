"""Finish the missing matched RL arms; keep one GPU slot for evaluation."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from cloud import ROOT, STATE, write


def execute(arm, stamp):
    name = 'study-' + arm + '-' + stamp
    with (STATE / (name + '.controller.log')).open('w') as log:
        result = subprocess.run([sys.executable, str(ROOT / 'experiments/cloud.py'), 'run',
            '--name', name, '--entry', 'experiments/run-study-' + arm + '.sh', '--hours', '5'],
            stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
    job = json.loads((STATE / 'ledger.json').read_text())['jobs'].get(name, {})
    complete = job.get('state') == 'complete' or (job.get('state') == 'delete_requested' and job.get('release_reason') == 'complete')
    return {'arm': arm, 'job': name, 'success': bool(result.returncode == 0 and complete and job.get('backup_verified')),
            'output': str(STATE / name / 'output'), 'cleanup_confirmed': bool(job.get('released_at'))}


def main():
    path = ROOT / 'lab/experiments/study/training-campaign.json'
    if path.exists():
        raise SystemExit('Study state already exists; inspect and explicitly recover rather than duplicating jobs')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    state = {'started_at': stamp, 'state': 'running', 'jobs': [], 'max_training_gpus': 2,
             'note': 'Third GPU slot reserved for evaluation; total campaign ceiling remains200USD'}
    write(path, state)
    with ThreadPoolExecutor(max_workers=2) as pool:
        for arms in [('grpo', 'dr-grpo'), ('rl-fixed', 'rl-adaptive', 'rl-difficulty', 'rl-progressive', 'rl-replay')]:
            futures = [pool.submit(execute, arm, stamp) for arm in arms]
            for future in as_completed(futures):
                state['jobs'].append(future.result())
                write(path, state)
            if not all(j['success'] for j in state['jobs']):
                state['state'] = 'needs_review'
                write(path, state)
                return
    state['state'] = 'completed'
    write(path, state)


if __name__ == '__main__':
    main()
