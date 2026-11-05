"""Sidecar for an already-running worker: expire its lease during target validation.

The reviewed worker checks this lease ONLY at the next training-loop boundary,
after validation and the complete checkpoint. Never signal torchrun or change
the model/data/optimizer, and never extend the provider/controller budget lease.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import time


def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temp.open('w') as f:
        json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    temp.replace(path)


def decision(progress, target):
    current=progress.get('current',-1)
    if progress.get('phase') in ('stopped','completed'):return 'already_stopped'
    if current<target:return 'waiting'
    if current==target and progress.get('phase')=='validation':return 'stop_after_validation'
    return 'late_boundary_stop'


def shortened_lease(lease, request, now):
    for key in ('job_name','pod_id','provider','hardware_scope','gpu','gpu_count','created_at'):
        if lease.get(key)!=request['identity'].get(key):raise RuntimeError('Stop request identity differs: '+key)
    old=lease['hard_deadline']
    if not isinstance(old,(int,float)) or not math.isfinite(old):raise ValueError('Invalid lease deadline')
    # Existing worker stops when time > hard_deadline - 600 at loop entry.
    new=min(old,now+590)
    if new<=lease['created_at']:raise ValueError('Invalid shortened deadline')
    return dict(lease,hard_deadline=new,requested_stop_after_validation=request['target'],user_requested_stop=True)


def run(root, request_path):
    root=Path(root);request=json.loads(Path(request_path).read_text());control=root/'control/replay-stop'
    control.mkdir(parents=True,exist_ok=True)
    lock=(control/'watch.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:return
    target=request['target'];identity=request['identity'];out=root/'output/training'
    if target!=1024 or identity['job_name']!='replay-scaled-runpod-resume-20261102t162212z':raise ValueError('Unregistered stop request')
    last_write=0
    while True:
        now=time.time()
        if (root/'output/exit.json').exists():
            write(control/'status.json',dict(state='worker_exited',updated_at=now,pid=os.getpid(),target=target));return
        try:progress=json.loads((out/'progress.json').read_text())
        except (OSError,ValueError):time.sleep(1);continue
        action=decision(progress,target)
        if action=='already_stopped':
            write(control/'status.json',dict(state=action,progress=progress,updated_at=now,pid=os.getpid(),target=target));return
        if action=='waiting':
            if now-last_write>=10:
                write(control/'status.json',dict(state='armed',current=progress['current'],phase=progress['phase'],updated_at=now,pid=os.getpid(),target=target));last_write=now
            time.sleep(1);continue
        lease_path=root/'worker-lease.json';lease=json.loads(lease_path.read_text())
        updated=shortened_lease(lease,request,now)
        receipt=dict(state='stop_requested',mode=action,target=target,observed_progress=progress,
            requested_at=now,old_worker_deadline=lease['hard_deadline'],new_worker_deadline=updated['hard_deadline'],
            canonical_budget_unchanged=True,job_name=identity['job_name'])
        write(control/'intent.json',receipt)
        write(lease_path,updated)
        write(out/'scheduled-stop.json',receipt)
        write(control/'status.json',dict(receipt,updated_at=time.time(),pid=os.getpid()))
        return


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--request',type=Path,required=True);a=p.parse_args();run(a.root,a.request)
