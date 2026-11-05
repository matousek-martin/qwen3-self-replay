"""One finite RunPod continuation from850; wait without renting unavailable GPUs."""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments import cloud
from experiments.replay_scaled.core import GPU,SCOPE,ENTRY,sha,write,ticket
from experiments.replay_scaled.notify import deliver_once
from ops import runpod
from ops.isolated_cloud import seal,verify

PARENT='replay-scaled-training-20261101t122440z'
REG=ROOT/'lab/experiments/replay-scaled/20261102-runpod-resume850'
UNIT='thinkingcap-replay-runpod-resume850'
INSTRUCTION='2026-11-02: User requests using existing RunPod credits for remaining self-replay training and explicitly chooses unchanged data; complete-state continuation from850, no external teacher or reward change.'

def read(path):return json.loads(Path(path).read_text())

def backup_space_ready():
    s=os.statvfs(cloud.STATE)
    return (s.f_bfree if os.geteuid()==0 else s.f_bavail)*s.f_frsize>=6*1024**3

def repair_budget(jobs, names):
    if len(names)!=len(set(names)):raise RuntimeError('Duplicate previous attempt')
    spent=0.
    for name in names:
        j=jobs[name]
        if not j.get('released_at') or not j.get('backup_verified') or j.get('state')!='failed':
            raise RuntimeError('Previous attempt must be reconciled and archived')
        progress=read(cloud.STATE/name/'output/training/progress.json')
        if progress.get('current')!=850 or progress.get('optimizer_updates')!=849:
            raise RuntimeError('New training state exists; select latest complete checkpoint')
        if any(read(p)['group']>850 for p in (cloud.STATE/name/'output/training/checkpoints').glob('*/complete.json')):
            raise RuntimeError('New checkpoint must not be discarded')
        charge=j.get('charged_estimate_usd')
        if not isinstance(charge,(int,float)) or not math.isfinite(charge) or charge<0:raise RuntimeError('Unknown prior charge')
        spent+=charge
    cap=round(65-spent,4)
    if cap<=1:raise RuntimeError('Original65USD envelope exhausted')
    return cap

def prepare(repair=False):
    if not backup_space_ready():raise RuntimeError('Insufficient local checkpoint backup reserve')
    exists=(REG/'launch.json').exists()
    if exists and not repair:raise RuntimeError('Continuation already registered')
    if repair and not exists:raise RuntimeError('No previous continuation')
    jobs=read(cloud.STATE/'ledger.json')['jobs'];parent=jobs[PARENT]
    previous=[];old_campaign={}
    if repair:
        old_campaign=read(REG/'campaign.json')
        previous=old_campaign.get('main_previous_jobs',[])+[read(REG/'launch.json')['job_name']]
    cap=repair_budget(jobs,previous) if repair else 65
    if not parent.get('released_at') or not parent.get('backup_verified'):raise RuntimeError('Parent not reconciled')
    if cloud.list_pods('runpod'):raise RuntimeError('Reconcile existing RunPod rentals first')
    account=runpod.account();credit=float(account['clientBalance'])
    if not math.isfinite(credit) or credit<cap+3 or account['currentSpendPerHr']!=0:raise RuntimeError('Funded65USD stage unavailable')
    old=ROOT/'lab/experiments/replay-scaled/20261101-checkpoints-v5'
    original=Path(read(old/'launch.json')['release'])
    protected=read(original/'payload-manifest.json')['files']
    # Scheduling/provider code can change; objective, worker, data and RNG code cannot.
    funded=old/'funding-extension-20261101/source'
    for f in ['experiments/replay_scaled/worker.py','experiments/replay_scaled/lease.py']:
        if sha(ROOT/f)!=sha(funded/f):raise RuntimeError('Verified funded-continuation source changed: '+f)
    for f in ['experiments/replay_scaled/'+x for x in ['generation.py','communication.py','metrics.py']]+['experiments/rl_training/train.py','experiments/rl_training/objectives.py','experiments/shorterbetter/multidomain/rewards.py']:
        if sha(ROOT/f)!=protected[f]:raise RuntimeError('Scientific implementation changed: '+f)
    training=cloud.STATE/PARENT/'output/training';checkpoint=training/'checkpoints/group-000850'
    m=read(checkpoint/'complete.json')
    if m['group']!=850 or m['world_size']!=2 or m['pilot'] or m['contract']!=sha(old/'plan.json'):raise RuntimeError('Checkpoint contract mismatch')
    if any(sha(checkpoint/f)!=h for f,h in m['files'].items()):raise RuntimeError('Corrupt850 checkpoint')
    stamp=time.strftime('%Y%m%dt%H%M%Sz',time.gmtime()).lower();name='replay-scaled-runpod-resume-'+stamp
    release=ROOT/'lab/isolated'/name
    ignore=shutil.ignore_patterns('.git','.env*','.venv*','__pycache__','runs','*.log')
    for folder in ['experiments','ThinkingCap-Test']:shutil.copytree(ROOT/folder,release/folder,ignore=ignore)
    stage=release/'lab/experiments/data/replay-scaled';stage.mkdir(parents=True)
    REG.mkdir(parents=True,exist_ok=repair)
    if repair:
        archive=REG/'attempts'/previous[-1];archive.mkdir(parents=True,exist_ok=False)
        for f in ['launch.json','campaign.json','live.json','account-capacity-verification.json']:
            if (REG/f).exists():shutil.copy2(REG/f,archive/f)
    for f in ['plan.json','train.jsonl','development.jsonl']:
        if sha(old/f)!=protected['lab/experiments/data/replay-scaled/'+f]:raise RuntimeError('Frozen data changed')
        shutil.copy2(old/f,stage/f);shutil.copy2(old/f,REG/f)
    shutil.copy2(cloud.STATE/PARENT/'output/gpu-verification.json',stage/'gpu-verification.json')
    shutil.copytree(training,stage/'resume-training',ignore=shutil.ignore_patterns('checkpoints','completion.json'))
    shutil.copytree(checkpoint,stage/'resume-training/checkpoints/group-000850')
    hashes={str(p.relative_to(stage/'resume-training')):sha(p) for p in (stage/'resume-training').rglob('*') if p.is_file()}
    write(stage/'resume-files.json',hashes)
    write(stage/'launch-mode.json',dict(mode='training',resume_group=850,verify_before_training=False))
    shutil.copy2(original/'lab/gpu-requirements.lock',release/'lab/gpu-requirements.lock')
    launch=dict(job_name=name,release=str(release),payload_sha256=seal(release),provider='runpod',gpu=GPU,gpu_count=2,
        hardware_scope=SCOPE,entry=ENTRY,max_cost_usd=cap,max_runtime_hours=24,max_gpu_rate=1.79,phase='training',previous_attempts=previous,
        resume_parent=PARENT,resume_group=850,resume_manifest_sha256=sha(checkpoint/'complete.json'))
    write(REG/'launch.json',launch)
    campaign=dict(state='queued_runpod',job_name=name,total=3840,gpu_count=2,full_training_authorized=True,
        telegram_authorized=True,released=False,main_previous_jobs=previous,
        main_authorization=dict(cap_usd=65,source_user_instruction=INSTRUCTION),
        queued_at=time.time(),queue_expires_at=old_campaign.get('queue_expires_at',time.time()+24*3600),credit_at_prepare=credit,resume_group=850)
    write(REG/'campaign.json',campaign)
    live=read(old/'live.json');live['progress']=read(training/'progress.json');live['progress']['phase']='queued_runpod';live['observed_at']=time.time()
    write(REG/'live.json',live)
    write(cloud.STATE/'authorizations'/(name+'.json'),dict(launch,authorized=True,scope='replay_runpod_resume_v1',source_user_instruction=INSTRUCTION))
    if not ticket(cloud.STATE,name,launch):raise RuntimeError('RunPod authorization validation failed')
    write(ROOT/'lab/experiments/replay-scaled/current.json',dict(path=str((REG/'campaign.json').relative_to(ROOT))))
    return launch

def check():
    c=read(REG/'campaign.json');launch=read(REG/'launch.json')
    if c['state']!='queued_runpod':
        subprocess.run(['systemctl','stop',UNIT+'-queue.timer'],check=False)
        return
    if time.time()>c['queue_expires_at']:
        c.update(state='queue_expired');write(REG/'campaign.json',c)
        from datetime import datetime
        from zoneinfo import ZoneInfo
        stamp=datetime.now(ZoneInfo('Europe/Prague')).strftime('%Y-%m-%d %H:%M:%S %Z (Praha)')
        deliver_once(REG/'notifications.sqlite3','queue-expired','<b>Project: ThinkingCap / Self-replay scaled</b>\nSummary: RunPod fronta po 24 hodinách vypršela bez pronájmu GPU; checkpoint 850 zachován.\nTimestamp: '+stamp)
        subprocess.run(['systemctl','stop',UNIT+'-queue.timer'],check=False)
        return
    name=launch['job_name'];jobs=read(cloud.STATE/'ledger.json')['jobs']
    if name in jobs:raise RuntimeError('Existing canonical job; never allocate twice')
    account=runpod.account()
    q=runpod.api('catalog/gpus/'+urllib.parse.quote(GPU)+'?include=AVAILABILITY&product=POD&count=2&minCudaVersion=12.8&cloud=SECURE')
    credit=float(account['clientBalance']);rate=float(q['price']['secure'])
    c.update(last_capacity_check=time.time(),credit_at_check=credit,quoted_rate_per_gpu=rate,availability=q['availability'])
    write(REG/'campaign.json',c)
    if (q['availability'] not in ('LOW','MEDIUM','HIGH') or not 0<rate<=launch['max_gpu_rate']
        or not math.isfinite(credit) or credit<launch['max_cost_usd']+3 or account['currentSpendPerHr']!=0 or cloud.list_pods('runpod') or not backup_space_ready()):return
    if verify(launch['release'])!=launch['payload_sha256']:raise RuntimeError('Payload changed')
    # Persist intent first. An uncertain local start is reconciled, never retried as a rental.
    c.update(state='starting',launch_requested_at=time.time());write(REG/'campaign.json',c)
    subprocess.run(['systemd-run','--unit='+UNIT+'-'+name.rsplit('-',1)[-1],'--property=Restart=on-failure','--property=RestartSec=30',
        '--working-directory='+str(ROOT),sys.executable,str(Path(__file__).resolve()),'run'],check=True)
    subprocess.run(['systemctl','stop',UNIT+'-queue.timer'],check=False)

def run():
    from ops import replay_scaled
    replay_scaled.REG=REG
    name=read(REG/'launch.json')['job_name'];job=read(cloud.STATE/'ledger.json')['jobs'].get(name)
    if job and job.get('released_at'):return
    # A recovered controller observes the existing worker, never launches it again.
    replay_scaled.run(observe_only=job is not None)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','prepare-repair','check','run']);a=p.parse_args()
    if a.action in ('prepare','prepare-repair'):print(json.dumps(prepare(repair=a.action=='prepare-repair'),indent=2))
    elif a.action=='check':check()
    else:run()
