"""Single registered pilot through the existing allocator/guard, no rerent loop."""
import json
import argparse
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import cloud
from isolated_cloud import verify

DIRECTORY = ROOT / 'lab/experiments/shorterbetter-launch-20261026'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=DIRECTORY)
    directory = parser.parse_args().directory.resolve()
    spec = json.loads((directory / 'launch.json').read_text())
    release = ROOT / spec['release']
    if verify(release) != spec['payload_sha256']:
        raise RuntimeError('Payload identity changed')
    name = spec['job_name']
    ledger = json.loads((cloud.STATE / 'ledger.json').read_text())
    job = ledger['jobs'].get(name)
    if not job:
        action = 'run'
    elif job.get('released_at'):
        cloud.write(directory / 'state.json', {'phase': job['state'], 'job_name': name,
                    'backup_verified': job.get('backup_verified', False), 'gpu_released': True,
                    'updated_at': time.time()})
        return
    elif job.get('pod_id') and job.get('identity_confirmed'):
        action = 'watch'
    else:
        raise RuntimeError('Existing allocation requires reconciliation; no duplicate rental')
    cloud.write(directory / 'state.json', {'phase': 'starting' if action == 'run' else 'watching',
                'job_name': name, 'updated_at': time.time(),
                'max_cost_usd': None if job and cloud.completion_budget_authorized(name, job) else spec['max_cost_usd'],
                'completion_budget_override': bool(job and cloud.completion_budget_authorized(name, job))})
    result = subprocess.run([sys.executable, str(ROOT / 'ops/isolated_cloud.py'), action,
        '--release', str(release), '--name', name, '--entry', spec['entry'],
        '--hours', str(spec['hours']), '--max-cost-usd', str(spec['max_cost_usd']),
        '--data-upload-timeout', str(spec.get('data_upload_timeout', 300))], cwd=ROOT)
    ledger = json.loads((cloud.STATE / 'ledger.json').read_text())
    job = ledger['jobs'].get(name, {})
    cloud.write(directory / 'state.json', {'phase': job.get('state', 'needs_inspection'),
                'job_name': name, 'controller_exit_code': result.returncode,
                'backup_verified': job.get('backup_verified', False),
                'gpu_released': bool(job.get('released_at')), 'updated_at': time.time()})
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
