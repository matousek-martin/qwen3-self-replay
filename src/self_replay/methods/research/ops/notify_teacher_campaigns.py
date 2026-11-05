"""Send only explicitly authorized, verified completion notices; never launch work."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, '/srv/playground')
from experiments import cloud
from project_alerts import Alerts

DIRECTORY = ROOT / 'lab/experiments/completion-notifications'


def completion_ready(target, campaign, results, ledger):
    trials = campaign.get('trials', {})
    if (campaign.get('state') != 'development_complete'
            or set(trials) != {str(s) for s in target['seeds']}
            or results.get('complete') is not True):
        return False
    expected = {(s, a) for s in target['seeds'] for a in target['arms']}
    comparisons = results.get('comparisons', [])
    actual = {(r.get('seed'), r.get('arm')) for r in comparisons}
    if actual != expected or len(comparisons) != len(expected):
        return False
    names = []
    for trial in trials.values():
        name = trial.get('job', '')
        job = ledger.get('jobs', {}).get(name, {})
        if (not name.startswith(target['job_prefix']) or trial.get('state') != 'complete'
                or job.get('state') != 'complete' or job.get('backup_verified') is not True
                or not job.get('released_at')):
            return False
        names.append(name)
    if len(set(names)) != len(names):
        return False
    return not any(n.startswith(target['job_prefix']) and not j.get('released_at')
                   for n, j in ledger.get('jobs', {}).items())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    os.umask(0o077)
    DIRECTORY.mkdir(parents=True, exist_ok=True, mode=0o700)
    config = json.loads((DIRECTORY / 'config.json').read_text())
    if config.get('telegram_authorized') is not True:
        raise ValueError('Completion notifications not authorized')
    with (DIRECTORY / 'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        alerts = Alerts(DIRECTORY / 'delivery.sqlite3', 'ThinkingCap')
        ledger = json.loads((cloud.STATE / 'ledger.json').read_text())
        for target in config['targets']:
            previous = alerts.db.execute('SELECT delivery FROM incidents WHERE key=?',
                                         (target['id'],)).fetchone()
            if previous:
                print(json.dumps({'target': target['id'], 'delivery': previous['delivery'], 'retry': False}))
                continue
            campaign_path = ROOT / target['campaign']
            results_path = ROOT / target['results']
            if not campaign_path.exists() or not results_path.exists():
                print(json.dumps({'target': target['id'], 'state': 'waiting_for_results'}))
                continue
            campaign = json.loads(campaign_path.read_text())
            results = json.loads(results_path.read_text())
            if not completion_ready(target, campaign, results, ledger):
                print(json.dumps({'target': target['id'], 'state': 'waiting_for_verified_completion'}))
                continue
            if args.dry_run:
                print(json.dumps({'target': target['id'], 'state': 'completion_metadata_ready', 'sent': False}))
                continue
            # Verify archived files again and provider absence before notifying.
            names = [r['job'] for r in campaign['trials'].values()]
            for name in names:
                archive = cloud.verify_backup(name)
                if archive['exit_code'] != 0:
                    raise ValueError('Nonzero archived exit code')
                output = cloud.STATE / name / 'output'
                role = json.loads((output / 'role-result.json').read_text())
                if role.get('development_complete') is not True:
                    raise ValueError('Incomplete archived role')
                if target.get('require_full_evaluation') and role.get('full_evaluation_complete') is not True:
                    raise ValueError('Incomplete full evaluation')
            pods = cloud.list_pods()
            ids = {ledger['jobs'][n].get('pod_id') for n in names}
            pod_names = {ledger['jobs'][n].get('pod_name') for n in names}
            if any(p.get('id') in ids or p.get('name') in pod_names for p in pods):
                print(json.dumps({'target': target['id'], 'state': 'waiting_for_provider_release'}))
                continue
            cloud.write(DIRECTORY / (target['id'] + '-evidence.json'), {
                'verified_at': datetime.now(timezone.utc).isoformat(), 'jobs': names,
                'archives_verified': True, 'provider_release_verified': True,
                'results_path': str(results_path),
            })
            accepted = alerts.emit(target['id'], 'Completion', target['message'])
            print(json.dumps({'target': target['id'], 'telegram_accepted': accepted}))


if __name__ == '__main__':
    main()
