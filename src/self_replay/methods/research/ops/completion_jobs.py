"""Resume identified jobs or launch sealed completion work through the shared allocator."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
from isolated_cloud import ROOT, verify
from experiments import cloud

STATE = ROOT / 'lab/experiments/completion-20261026'


def run_job(release, name, entry, hours=4):
    release = Path(release).resolve(); payload = verify(release)
    directory = STATE/name;directory.mkdir(parents=True, exist_ok=True)
    def status(phase, **extra):
        cloud.write(directory/'state.json', {'phase':phase,'job':name,'payload_sha256':payload,
                                            'updated_at':time.time(),**extra})
    with (directory/'lock').open('a') as lock:
        # A prefetch and its downstream consumer may request the same sealed
        # job. Wait for its owner, then reconcile the ledger inside the lock;
        # never launch a second rental or run two backup/watch loops.
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            with cloud.ledger() as ledger:job=dict(ledger['jobs'].get(name,{}))
            if not job:
                while True:
                    with cloud.ledger() as ledger:
                        slots=cloud.potentially_live_pods(ledger['jobs'])
                        available=cloud.USABLE-cloud.liability(ledger['jobs'])
                    if slots<cloud.MAX_LIVE_PODS and available>=hours*1.3:break
                    status('queued_capacity' if slots>=cloud.MAX_LIVE_PODS else 'queued_budget')
                    time.sleep(30)
                action='run'
            elif job.get('released_at'):
                if job.get('state')!='complete':raise RuntimeError('Failed job needs investigation; no duplicate rental')
                action=None
            elif job.get('pod_id') and job.get('identity_confirmed'):
                action='watch'
            else:raise RuntimeError('Ambiguous allocation needs reconciliation')
            status('running')
            if action:
                with (directory/'controller.log').open('a') as log:
                    subprocess.run([sys.executable, ROOT/'ops/isolated_cloud.py',action,'--release',release,
                                    '--name',name,'--entry',entry,'--hours',str(hours)],cwd=ROOT,
                                   stdout=log,stderr=subprocess.STDOUT,check=True)
            with cloud.ledger() as ledger:job=dict(ledger['jobs'][name])
            result=cloud.verify_backup(name)
            if result['exit_code']!=0 or job.get('state')!='complete' or not job.get('released_at') or not job.get('backup_verified'):
                raise RuntimeError('Worker failed or backup/release incomplete')
            status('complete',backup_verified=True,gpu_released=True)
            return cloud.STATE/name/'output'
        except BaseException as exc:
            status('needs_inspection',error=str(exc));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--release',required=True);p.add_argument('--name',required=True)
    p.add_argument('--entry',required=True);p.add_argument('--hours',type=float,default=4)
    a=p.parse_args();run_job(a.release,a.name,a.entry,a.hours)
