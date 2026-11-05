"""Arm and verify a user-requested stop; never rent or restart a training worker."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments import cloud
from experiments.replay_scaled.core import sha,write
from experiments.replay_scaled.notify import deliver_once

JOB='replay-scaled-runpod-resume-20261102t162212z'
REG=ROOT/'lab/experiments/replay-scaled/20261102-runpod-resume850'
REQUEST=REG/'stop-after-validation-1024.json'
REMOTE='/workspace/project/control/replay-stop'


def read(p):return json.loads(Path(p).read_text())


def inspect(pod):
    script="import json;from pathlib import Path;r=Path('/workspace/project');files={'progress':'output/training/progress.json','lease':'worker-lease.json','status':'control/replay-stop/status.json'};print(json.dumps({k:json.loads((r/f).read_text()) for k,f in files.items() if (r/f).exists()}))"
    return json.loads(cloud.remote(pod,'python3 -c '+shlex.quote(script),timeout=25))


def ensure_sidecar(pod, request):
    observed=inspect(pod)
    status=observed.get('status',{})
    if status.get('state') in ('stop_requested','already_stopped','worker_exited'):return observed
    if status.get('state')=='armed' and time.time()-status.get('updated_at',0)<40:return observed
    cloud.remote(pod,'mkdir -p '+REMOTE)
    cloud.sync(pod,ROOT/'experiments/replay_scaled/scheduled_stop.py',REMOTE+'/watch.py',upload=True)
    cloud.sync(pod,REQUEST,REMOTE+'/request.json',upload=True)
    command='nohup python3 '+REMOTE+'/watch.py --root /workspace/project --request '+REMOTE+'/request.json > '+REMOTE+'/watch.log 2>&1 < /dev/null &'
    cloud.remote(pod,command,timeout=20)
    return observed


def arm():
    job=read(cloud.STATE/'ledger.json')['jobs'][JOB]
    if job.get('released_at'):raise RuntimeError('Training already ended; inspect archive before setting a stop')
    if REQUEST.exists():raise RuntimeError('Stop is already registered; use watch')
    pod=cloud.api('pods/'+job['pod_id']);observed=inspect(pod);progress=observed['progress']
    if progress['current']>=1024:raise RuntimeError('Target already reached; inspect actual state before arming')
    release=Path(read(REG/'launch.json')['release'])
    expected={f:sha(release/f) for f in ['experiments/replay_scaled/worker.py','experiments/replay_scaled/lease.py']}
    script="import json,hashlib;from pathlib import Path;r=Path('/workspace/project');files="+repr(list(expected))+";print(json.dumps({f:hashlib.sha256((r/f).read_bytes()).hexdigest() for f in files}))"
    actual=json.loads(cloud.remote(pod,'python3 -c '+shlex.quote(script),timeout=25))
    if actual!=expected:raise RuntimeError('Live training source differs from reviewed lease-boundary implementation')
    lease=observed['lease'];identity={k:lease[k] for k in ['job_name','pod_id','provider','hardware_scope','gpu','gpu_count','created_at']}
    if identity['job_name']!=JOB or identity['pod_id']!=job['pod_id']:raise RuntimeError('Training identity mismatch')
    request=dict(target=1024,identity=identity,requested_at=time.time(),observed_progress=progress,source_sha256=expected,
        helper_sha256=sha(ROOT/'experiments/replay_scaled/scheduled_stop.py'),stop_after_validation=True,no_auto_resume=True,
        instruction='User explicitly requests stopping at1024 after completed validation, full checkpoint and backup, to review accuracy before any further training.')
    write(REQUEST,request)
    cloud.update(JOB,requested_stop_after_validation=1024,user_stop_requested_at=request['requested_at'],no_auto_resume=True)
    campaign=read(REG/'campaign.json');campaign.update(requested_stop_after_validation=1024,no_auto_resume=True);write(REG/'campaign.json',campaign)
    subprocess.run(['systemctl','stop','thinkingcap-replay-runpod-resume850-queue.timer'],check=True)
    ensure_sidecar(pod,request)
    return request


def verify_archive(job):
    root=cloud.STATE/JOB/'output/training';completion=read(root/'completion.json')
    checkpoint=root/'checkpoints/group-001024';manifest=read(checkpoint/'complete.json');validation=read(root/'validations/001024.json')
    if not job.get('backup_verified') or not job.get('released_at'):raise RuntimeError('Backup/release not complete')
    if completion['groups']!=1024 or manifest['group']!=1024 or validation['iteration']!=1024:raise RuntimeError('Stop boundary mismatch')
    if manifest['world_size']!=2 or manifest['pilot'] or not {'state.pt','rng-0.pt','rng-1.pt','adapter/adapter_model.safetensors'}.issubset(manifest['files']):raise RuntimeError('Incomplete resume state')
    if any(sha(checkpoint/f)!=h for f,h in manifest['files'].items()):raise RuntimeError('Checkpoint SHA verification failed')
    if len((root/'validations/001024.jsonl').read_text().splitlines())!=1024:raise RuntimeError('Incomplete256x4 validation')
    return dict(state='complete',checkpoint=1024,validation=1024,backup_verified=True,released=True,verified_at=time.time(),checkpoint_manifest_sha256=sha(checkpoint/'complete.json'))


def watch():
    request=read(REQUEST)
    while True:
        job=read(cloud.STATE/'ledger.json')['jobs'][JOB]
        if job.get('released_at'):
            try:result=verify_archive(job)
            except Exception as exc:
                result=dict(state='stopped_elsewhere_or_incomplete',error=str(exc),observed_at=time.time(),job_state=job.get('state'))
            write(REG/'scheduled-stop-status.json',result)
            return
        pod=cloud.api('pods/'+job['pod_id']);observed=ensure_sidecar(pod,request)
        write(REG/'scheduled-stop-status.json',dict(state='watching',observed_at=time.time(),remote=observed))
        if observed.get('status',{}).get('mode')=='late_boundary_stop':
            deliver_once(REG/'notifications.sqlite3',JOB+':late-scheduled-stop',
                '<b>Project: ThinkingCap / Self-replay scaled</b>\nSummary: Požadovaná hranice1024 byla překročena; provádí se šetrné zastavení a kontrola checkpointu.')
        time.sleep(20)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['arm','watch']);a=p.parse_args()
    if a.action=='arm':print(json.dumps(arm(),indent=2))
    else:watch()
