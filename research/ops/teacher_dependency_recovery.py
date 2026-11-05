"""One-time recovery of known, already reconciled controller failures.

Never interrupt an API call; restart only after the known local ImportError has
ended the process. Durable generation/review receipts prevent paid replays.
"""
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'lab/experiments/teacher-generation/completion-v2-20261026'


def main():
    while True:
        state=json.loads((STATE/'pipeline.json').read_text())
        if state['phase'] in ['matched_training_campaign','development_complete']:
            return
        if state['phase']=='needs_inspection':
            error=state.get('error','')
            old_nonblocking_lock='Resource temporarily unavailable' in error
            reconciled_sampling=error=='Failed job needs investigation; no duplicate rental'
            if 'jinja2' not in error and not old_nonblocking_lock and not reconciled_sampling:
                raise RuntimeError('Unrelated failure; automatic restart refused')
            if reconciled_sampling:
                policy=json.loads((ROOT/'lab/experiments/completion-20261026/execution-policy.json').read_text())
                audit=policy.get('r2_startup_reconciliation',{})
                original='teacher-completion-student-r2-70289c50bc'
                if audit.get('old_job')!=original or not audit.get('worker_never_launched') or not audit.get('provider_absence_verified'):
                    raise RuntimeError('Missing explicit startup-failure reconciliation')
                if policy.get('sampling_job_overrides',{}).get(original)!='teacher-completion-student-r2b-70289c50bc':
                    raise RuntimeError('Unexpected replacement job identity')
                ledger=json.loads((ROOT/'lab/experiments/cloud/ledger.json').read_text())
                if not ledger['jobs'][original].get('released_at'):
                    raise RuntimeError('Original failed worker still live')
            if old_nonblocking_lock:
                ledger=json.loads((ROOT/'lab/experiments/cloud/ledger.json').read_text())
                prefetched=[j for n,j in ledger['jobs'].items() if n.startswith('teacher-completion-student-r1-')]
                if len(prefetched)!=1:raise RuntimeError('Cannot identify the exact prefetched job')
                job=prefetched[0]
                if not job.get('released_at'):
                    time.sleep(15);continue
                if job.get('state')!='complete' or not job.get('backup_verified'):
                    raise RuntimeError('Prefetched job failed; do not retry allocation')
            unit='thinkingcap-completion-teacher.service'
            active=subprocess.check_output(['systemctl','show',unit,'--property=ActiveState','--value'],text=True).strip()
            if active not in ['failed','inactive']:
                time.sleep(2);continue
            for p in STATE.glob('student-review-*/run/requests/*.json'):
                receipt=json.loads(p.read_text())
                accounted_failure=(receipt.get('state')=='needs_reconciliation'
                    and receipt.get('wire',{}).get('http_status') in (500,502,503,504)
                    and (p.parents[2]/'reviews.json').exists()
                    and (p.parents[2]/'server-error-continuation.json').exists())
                if receipt.get('state')!='completed' and not accounted_failure:
                    raise RuntimeError('Uncertain API receipt; automatic restart refused')
            subprocess.run(['systemctl','restart',unit],check=True)
            (STATE/('r2-controller-recovery.json' if reconciled_sampling else 'jinja-recovery.json')).write_text(json.dumps({'at':time.time(),
                'reason':error,
                'action':'Restarted terminated controller; reused completed generation/review receipts; no GPU restart.'},indent=2)+'\n')
            return
        time.sleep(15)


if __name__=='__main__':main()
