"""Run one authorized bounded diagnostic under the shared supervisor."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    protocol = ROOT / 'experiments/shorterbetter/launch/protocol.json'
    lease = json.loads((ROOT / 'worker-lease.json').read_text())
    evidence = json.loads((ROOT / 'experiments/shorterbetter/launch/readiness.json').read_text())
    registered = json.loads(protocol.read_text())
    cap = registered['budget']['proposed_total_pilot_cap_usd']
    evidence['budget_reservation'] = lease['max_cost_usd'] == cap and 1 < cap <= 20
    evidence['pilot_cap_enforced'] = lease.get('hard_deadline', 0) > lease['created_at']
    if not evidence['budget_reservation'] or not evidence['pilot_cap_enforced']:
        raise RuntimeError('Missing bounded allocation ticket')
    evidence_path = ROOT / 'output/readiness-at-start.json'
    evidence_path.write_text(json.dumps(evidence, indent=2) + '\n')
    command = [sys.executable, '-m', 'experiments.shorterbetter.train',
                    '--protocol', str(protocol), '--arm', 'shorterbetter', '--seed', '17',
                    '--execute', '--readiness', str(evidence_path), '--job-name', lease['job_name']]
    if registered.get('execution', {}).get('run_startup_optimizer_check'):
        subprocess.run(command + ['--out', str(ROOT / 'output/startup-check'), '--startup-check'], check=True)
        check = json.loads((ROOT / 'output/startup-check/parameter-update-check.json').read_text())
        if check['status'] != 'passed':
            raise RuntimeError('Startup optimizer verification failed')
    subprocess.run(command + ['--out', str(ROOT / 'output/training')], check=True)


if __name__ == '__main__':
    main()
