"""One user-authorized pilot completion notice; never launches or alters GPU work."""
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
from project_alerts import Alerts, format_message

DIRECTORY = ROOT / 'lab/experiments/shorterbetter-retry4-20261027/notification'


def summarize(output, job, launch, archive_verified):
    """Use only main-run, fully archived monitoring; startup checks are excluded."""
    report = {}
    finished = {}
    if archive_verified:
        for filename, target in [('learning-curve.json', report), ('training-finished.json', finished)]:
            path = output / 'training' / filename
            if path.exists():
                target.update(json.loads(path.read_text()))
    checkpoints = report.get('checkpoints', [])
    baseline = next((r for r in checkpoints if r['optimizer_step'] == 0 and r['questions'] == 128), None)
    candidates = [r for r in checkpoints if r['optimizer_step'] > 0 and r['questions'] == 128]
    latest = max(candidates, key=lambda r: r['optimizer_step']) if candidates else None
    completed = finished.get('training_completed') is True and job['state'] == 'complete'
    if job.get('user_stop_requested_at') and not completed:
        parts = ['Pilot ukončen na přání uživatele kvůli podezření na ztrátu aktualizací v BF16.']
    else:
        parts = ['Pilot dokončen.' if completed else 'Pilot ukončen předčasně (' + job['state'] + ').']
    if baseline and latest:
        parts.append(f"Interní validace, krok {latest['optimizer_step']}: přesnost {baseline['accuracy']:.1%} → {latest['accuracy']:.1%}; tokeny {baseline['mean_total_tokens']:.0f} → {latest['mean_total_tokens']:.0f}; useknutí {baseline['truncation_rate']:.1%} → {latest['truncation_rate']:.1%}.")
        parts.append('1 seed; nejde o finální důkaz účinnosti.')
    else:
        parts.append('Chybí ověřené srovnání baseline a trénovaného modelu; zlepšení nelze posoudit.')
    if finished:
        parts.append(f"Trénink {finished['optimizer_steps']}/80 aktualizací.")
    cost = None
    if job.get('charged_estimate_usd') is not None:
        cost = launch['prior_cost_estimate_usd'] + job['charged_estimate_usd']
        limit = ('limit pilotu zrušen uživatelem' if job.get('completion_budget_override', {}).get('authorized')
                 else 'limit 20 USD')
        parts.append(f'Náklady všech pokusů odhadem {cost:.2f} USD; {limit}.')
    parts.append('GPU uvolněna; ' + ('záloha ověřena.' if archive_verified else 'záloha není ověřena.'))
    return ' '.join(parts), {'completed': completed, 'monitoring': report,
        'training_finished': finished, 'cumulative_estimated_usd': cost,
        'archive_verified': archive_verified}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    os.umask(0o077)
    config = json.loads((DIRECTORY / 'config.json').read_text())
    if config.get('telegram_authorized') is not True:
        raise ValueError('Notification not authorized')
    launch = json.loads((DIRECTORY.parent / 'launch.json').read_text())
    name = launch['job_name']
    if name != config['job_name']:
        raise ValueError('Notification target changed')
    with (DIRECTORY / 'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        alerts = Alerts(DIRECTORY / 'delivery.sqlite3', 'ThinkingCap / ShorterBetter')
        old = alerts.db.execute('SELECT delivery FROM incidents WHERE key=?', (name,)).fetchone()
        if old:
            print(json.dumps({'delivery': old['delivery'], 'automatic_retry': False}))
            return
        ledger = json.loads((cloud.STATE / 'ledger.json').read_text())
        job = ledger['jobs'][name]
        if not job.get('released_at'):
            print('Waiting for pilot completion and GPU release; no message sent.')
            return
        pods = cloud.list_pods()
        if any(p['id'] == job.get('pod_id') or p['name'] == job.get('pod_name') for p in pods):
            print('Waiting for provider-confirmed GPU absence; no message sent.')
            return
        archive_verified = False
        if job.get('backup_verified'):
            try:
                cloud.verify_backup(name)
                archive_verified = True
            except (OSError, ValueError, RuntimeError):
                print('Archive verification failed; notification will report unavailable results.')
        message, evidence = summarize(cloud.STATE / name / 'output', job, launch, archive_verified)
        if args.dry_run:
            print(format_message('ThinkingCap / ShorterBetter', 'Results', message))
            return
        evidence.update(job_name=name, provider_release_verified=True,
                        checked_at=datetime.now(timezone.utc).isoformat(), message=message)
        cloud.write(DIRECTORY / 'results.json', evidence)
        accepted = alerts.emit(name, 'Results', message)
        print(json.dumps({'telegram_accepted': accepted}))


if __name__ == '__main__':
    main()
