"""User-authorized, one-time Telegram completion notice for the 4090 batch."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))
sys.path.insert(0, '/srv/playground')
import cloud
from project_alerts import Alerts
from experiments.replay_round2 import LOCAL, PLAN, load_plan

KEY = 'replay-round2-4090-completed'


def completion_ready(plan, campaign, ledger):
    expected = {t['name'] for t in plan['trials']}
    results = campaign.get('trials', {})
    if campaign.get('state') != 'development_complete' or set(results) != expected:
        return False
    for trial in results.values():
        job = ledger.get('jobs', {}).get(trial.get('job'), {})
        if (trial.get('state') != 'complete' or job.get('state') != 'complete'
                or not job.get('released_at') or not job.get('backup_verified')
                or job.get('gpu') != 'NVIDIA GeForce RTX 4090'):
            return False
    return not any(n.startswith('replay2-') and not j.get('released_at')
                   for n, j in ledger.get('jobs', {}).items())


def main():
    os.umask(0o077)
    directory = LOCAL / 'notifications'
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    alerts = Alerts(directory / 'delivery.sqlite3', 'ThinkingCap')
    # Never retry a previous attempt automatically, including uncertain delivery.
    previous = alerts.db.execute('SELECT delivery FROM incidents WHERE key=?', (KEY,)).fetchone()
    if previous:
        print(json.dumps({'notification': previous['delivery'], 'retry': False}))
        return
    plan = load_plan(PLAN)
    campaign = json.loads((LOCAL / 'campaign.json').read_text())
    ledger = json.loads((cloud.STATE / 'ledger.json').read_text())
    if not completion_ready(plan, campaign, ledger):
        print(json.dumps({'notification': 'waiting_for_verified_completion'}))
        return
    verified = []
    for trial in plan['trials']:
        jobname = campaign['trials'][trial['name']]['job']
        archive = cloud.verify_backup(jobname)
        output = cloud.STATE / jobname / 'output'
        role = json.loads((output / 'role-result.json').read_text())
        evaluation = json.loads((output / 'study/study-result.json').read_text())
        if (archive['exit_code'] != 0 or role['trial'] != trial
                or not role.get('development_complete') or not evaluation.get('complete')
                or set(evaluation['completed_candidates']) != {'base', trial['name']}):
            raise RuntimeError('Completion evidence differs for ' + trial['name'])
        if 'GPU requirement verified: NVIDIA GeForce RTX 4090' not in (output / 'runner.log').read_text(errors='replace'):
            raise RuntimeError('Actual GPU startup verification absent for ' + trial['name'])
        verified.append(jobname)
    pods = cloud.list_pods()
    if any(p.get('name', '').startswith('thinkingcap-pilot-replay2-') for p in pods):
        print(json.dumps({'notification': 'waiting_for_provider_release'}))
        return
    cloud.write(directory / 'completion-evidence.json', {
        'verified_at': datetime.now(timezone.utc).isoformat(), 'jobs': verified,
        'archives_verified': True, 'provider_release_verified': True})
    accepted = alerts.emit(KEY, 'Completion',
        'Batch complete: all 9 RTX 4090 replay trials finished training and paired baseline evaluation. '
        'Results are backed up and verified; all batch GPUs are released. Scientific comparison is ready for review.')
    print(json.dumps({'telegram_accepted': accepted,
                     'note': 'Acceptance is not proof the recipient has read it; uncertain delivery is not retried.'}))


if __name__ == '__main__':
    main()
