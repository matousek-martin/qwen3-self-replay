"""Hourly disk checks for the four authorized isolated ThinkingCap experiments."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import cloud
from worker_disk_check import pressure

STATE = ROOT / 'lab/experiments/disk-monitor'
PREFIXES = ('replay-selection-seed', 'adaptive-isolated-v2-seed', 'opd-isolated-v2-seed',
            'teacher-student-inference-', 'teacher-matched-isolated-v1-seed',
            'completion-evaluations-', 'adaptive-completion-v1-seed',
            'teacher-completion-student-', 'teacher-completion-v2-seed',
            'teacher-completion-eval-preflight-', 'teacher-completion-extra-',
            'teacher-full-v1-seed', 'teacher-full-v2-seed',
            'round3-student-', 'teacher-round3-v1-', 'teacher-round3-v2-')


def selected(jobs):
    return {name: job for name, job in jobs.items()
            if name.startswith(PREFIXES) and job.get('pod_id') and job.get('identity_confirmed')
            and not job.get('released_at')}


def inspect(name, job):
    pod = cloud.api('pods/' + job['pod_id'])
    if pod.get('id') != job['pod_id'] or pod.get('name') != job['pod_name']:
        raise ValueError('Provider identity mismatch')
    if pod.get('status') != 'RUNNING':
        return {'job': name, 'pod_id': job['pod_id'], 'status': 'not_running'}
    code = (ROOT / 'ops/worker_disk_check.py').read_text()
    result = json.loads(cloud.remote(pod, 'python3 -c ' + shlex.quote(code), timeout=150))
    return {'job': name, 'pod_id': job['pod_id'], 'status': 'checked', **result}


def campaigns_complete():
    round3 = ROOT / 'lab/experiments/teacher-round3-20261026/pipeline.json'
    if round3.exists() and json.loads(round3.read_text()).get('phase') != 'development_complete':
        return False
    full = ROOT / 'lab/experiments/teacher-full-20261026-v2/campaign.json'
    if not full.exists():
        full = ROOT / 'lab/experiments/teacher-full-20261026/campaign.json'
    if full.exists() and json.loads(full.read_text()).get('state') != 'development_complete':
        return False
    completion = ROOT / 'lab/experiments/completion-20261026/summary.json'
    if completion.exists():
        return json.loads(completion.read_text()).get('complete') is True
    paths = ['replay-selection-v1', 'adaptive-replay-isolated-v2', 'on-policy-distillation-isolated-v2']
    for path in paths:
        f = ROOT / 'lab/experiments' / path / 'campaign.json'
        if not f.exists() or json.loads(f.read_text()).get('state') != 'development_complete':
            return False
    f = ROOT / 'lab/experiments/teacher-generation/isolated-pipeline-20261025/pipeline.json'
    if not f.exists():
        return False
    phase = json.loads(f.read_text()).get('phase', '')
    return phase == 'development_complete' or phase.startswith('complete_')


def main():
    os.umask(0o077)
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / 'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = json.loads((STATE / 'config.json').read_text())
        alerts = None
        if config.get('telegram_authorized') is True:
            sys.path.insert(0, '/srv/playground')
            from project_alerts import Alerts
            alerts = Alerts(STATE / 'alerts.sqlite', 'ThinkingCap disk monitor')
        oldpath = STATE / 'latest.json'
        old = json.loads(oldpath.read_text()) if oldpath.exists() else {}
        incidents = old.get('incidents', {})
        changes = []

        def incident(key, level, problem):
            previous = incidents.get(key)
            if level == 'healthy':
                if not previous:
                    return
                incidents.pop(key, None)
                message = 'Resolved: ' + problem
                changes.append({'key': key, 'level': 'recovered', 'message': message})
                if alerts:
                    alerts.resolve(key, 'Technical', message)
            else:
                signature = level + ':' + problem
                if previous == signature:
                    return
                incidents[key] = signature
                changes.append({'key': key, 'level': level, 'message': problem})
                if alerts:
                    alerts.emit(key, 'Technical', problem)

        jobs = selected(json.loads((cloud.STATE / 'ledger.json').read_text())['jobs'])
        results = []
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {pool.submit(inspect, n, j): n for n, j in jobs.items()}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    if result['status'] == 'not_running':
                        continue
                    incident(name + ':access', 'healthy', name + ': worker disk checks reachable again.')
                    for mount, disk in result['after'].items():
                        level = pressure(disk)
                        # Fixed threshold descriptions deduplicate small fluctuations.
                        problem = (f'{name}: {mount} disk space is healthy again.' if level == 'healthy'
                                   else f'{name}: {mount} disk pressure is {level} after safe cleanup; experiment writes may be at risk.')
                        incident(name + ':' + mount, level, problem)
                    if result['cleanup']['status'] == 'cleaned' and result['freed_bytes'] > 1024 ** 3:
                        changes.append({'key': name + ':cleanup', 'level': 'recovered',
                                        'message': f"{name}: unused installation cache cleaned; freed {result['freed_bytes']/1024**3:.1f} GiB."})
                except Exception as exc:
                    results.append({'job': name, 'status': 'check_failed', 'error_type': type(exc).__name__})
                    incident(name + ':access', 'unknown', name + ': disk check unavailable; protection cannot be verified.')
        # A released worker no longer has a disk at risk. Do not touch its files.
        for key in list(incidents):
            if key.split(':', 1)[0] not in jobs:
                incident(key, 'healthy', key.split(':', 1)[0] + ': worker released; monitoring incident closed.')
        complete = not jobs and campaigns_complete()
        report = {'checked_at': time.time(), 'telegram_authorized': bool(alerts), 'results': results,
                  'incidents': incidents, 'changes': changes, 'campaigns_complete': complete}
        cloud.write(oldpath, report)
        with (STATE / 'history.jsonl').open('a') as stream:
            stream.write(json.dumps(report) + '\n')
        for change in changes:
            print(json.dumps(change), flush=True)
        if complete:
            subprocess.run(['systemctl', 'disable', '--now', 'thinkingcap-disk-monitor.timer'], check=True)
            print('All four campaigns completed; hourly disk timer disabled.', flush=True)


if __name__ == '__main__':
    main()
