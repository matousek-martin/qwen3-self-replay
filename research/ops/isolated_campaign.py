"""Sequence registered paired jobs through the one canonical cloud controller."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
from isolated_cloud import ROOT, verify, digest
from experiments import cloud


def write(path, value):
    cloud.write(path, value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--predecessor', help='Wait for this exact failed allocation to reconcile before new work')
    args = parser.parse_args()
    release, directory = args.release.resolve(), args.state.resolve()
    lock_hash = verify(release)
    config = json.loads((release / 'lab/experiments/data/isolated/config.json').read_text())
    policy_path = ROOT / 'lab/experiments/completion-20261026/execution-policy.json'
    policy = json.loads(policy_path.read_text()) if policy_path.exists() else {}
    for prerequisite in policy.get('prerequisites', {}).get(config['id'], []):
        from completion_jobs import run_job
        output = run_job(ROOT / prerequisite['release'], prerequisite['name'],
                         prerequisite['entry'], prerequisite['hours'])
        if not json.loads((output / 'role-result.json').read_text()).get('development_complete'):
            raise RuntimeError('Campaign prerequisite failed; training not dispatched')
    concurrency = policy.get('parallel_campaigns', {}).get(config['id'], 1)
    if concurrency > 1:
        if args.predecessor:
            raise ValueError('Parallel dispatch cannot bypass predecessor reconciliation')
        if type(concurrency) is not int or not 1 < concurrency <= cloud.MAX_LIVE_PODS:
            raise ValueError('Parallelism exceeds shared worker limit')
        from parallel_campaign import run
        return run(release, directory, config, lock_hash, concurrency)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'campaign.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = directory / 'campaign.json'
        if path.exists():
            state = json.loads(path.read_text())
            if state['payload_sha256'] != lock_hash:
                raise RuntimeError('Campaign release identity changed')
        else:
            state = {'release': str(release), 'payload_sha256': lock_hash,
                     'created_at': time.time(), 'state': 'prepared',
                     'trials': {str(s): {'job': f"{config['id']}-seed{s}-{lock_hash[:10]}",
                                        'state': 'queued'} for s in config['seeds']}}
            write(path, state)
        try:
            if args.predecessor:
                while True:
                    with cloud.ledger() as ledger:
                        previous = dict(ledger['jobs'][args.predecessor])
                    if previous.get('released_at'):
                        if not previous.get('safely_no_pod'):
                            raise RuntimeError('Predecessor was not proven absent; inspect before retry')
                        break
                    state.update(state='waiting_for_reconciliation', predecessor=args.predecessor)
                    write(path, state)
                    time.sleep(30)
            for seed, trial in state['trials'].items():
                if trial['state'] == 'complete':
                    continue
                verify(release)
                with cloud.ledger() as ledger:
                    job = dict(ledger['jobs'].get(trial['job'], {}))
                if not job:
                    while True:
                        with cloud.ledger() as ledger:
                            slots = cloud.potentially_live_pods(ledger['jobs'])
                            available = cloud.USABLE - cloud.liability(ledger['jobs'])
                        if slots < cloud.MAX_LIVE_PODS and available >= config['hours'] * 1.3:
                            break
                        state['state'] = 'queued_capacity' if slots >= cloud.MAX_LIVE_PODS else 'queued_budget'
                        trial['state'] = state['state']
                        state['updated_at'] = time.time()
                        write(path, state)
                        time.sleep(45)
                    action = 'run'
                elif job.get('released_at'):
                    if job.get('state') != 'complete':
                        raise RuntimeError('Known unsuccessful job requires inspection; no automatic rerental')
                    action = None
                elif job.get('pod_id') and job.get('identity_confirmed'):
                    action = 'watch'
                else:
                    raise RuntimeError('Existing allocation uncertain; reconcile without duplicate rental')
                state['state'] = 'running'
                trial['state'] = 'running'
                write(path, state)
                if action:
                    command = [sys.executable, str(ROOT / 'ops/isolated_cloud.py'), action,
                               '--release', str(release), '--name', trial['job'],
                               '--entry', f'experiments/run-isolated-seed{seed}.sh',
                               '--hours', str(config['hours'])]
                    with (directory / f'seed{seed}.controller.log').open('a') as log:
                        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
                with cloud.ledger() as ledger:
                    job = dict(ledger['jobs'][trial['job']])
                result = cloud.verify_backup(trial['job'])
                role = json.loads((cloud.STATE / trial['job'] / 'output/role-result.json').read_text())
                if (result['exit_code'] != 0 or job.get('state') != 'complete' or not job.get('released_at')
                        or not job.get('backup_verified') or not role.get('development_complete')
                        or role['seed'] != int(seed)):
                    raise RuntimeError('Incomplete paired result; remaining jobs stopped')
                trial.update(state='complete', backup_verified=True, gpu_released=True)
                write(path, state)
            state.update(state='development_complete', completed_at=time.time())
            write(path, state)
        except BaseException as exc:
            state.update(state='needs_inspection', error=str(exc), updated_at=time.time())
            write(path, state)
            raise


if __name__ == '__main__':
    main()
