"""One unchanged full-state continuation1024→1280, with canonical allocation."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments import cloud
from experiments.replay_scaled.core import GPU,SCOPE,ENTRY,sha,write,ticket
from ops import vastai,replay_scaled
from ops.isolated_cloud import seal,verify

PARENT='replay-scaled-runpod-resume-20261102t162212z'
REG=ROOT/'lab/experiments/replay-scaled/20261105-vast-resume1024'
CAP=28
INSTRUCTION='2026-11-05: User approves unchanged original training continuation1024 to1280 on2A100/Vast after7–8h/18–21USD estimate; finite28USD protective cap, stop after validation1280, backup/release/UI/Telegram; no automatic continuation.'

def read(p):return json.loads(Path(p).read_text())

def prepare():
    if REG.exists():raise RuntimeError('Continuation already registered; reconcile before any retry')
    jobs=read(cloud.STATE/'ledger.json')['jobs']
    parent=jobs[PARENT]
    if not parent.get('released_at') or not parent.get('backup_verified'):raise RuntimeError('Parent not archived/released')
    if cloud.list_pods('vast'):raise RuntimeError('Reconcile existing Vast allocations')
    if any(j.get('provider')=='vast' and not j.get('released_at') for j in jobs.values()):raise RuntimeError('Unresolved Vast liability')
    account=vastai.api('v0/users/current/');credit=float(account['credit'])
    if not account.get('can_pay') or not math.isfinite(credit) or credit<CAP+5:raise RuntimeError('Credit cannot cover bounded continuation')
    space=os.statvfs(cloud.STATE)
    if space.f_bavail*space.f_frsize<8*1024**3:raise RuntimeError('Insufficient backup reserve')
    quote=vastai.quote(GPU,2,60,min_duration=10.5*3600,allowed_countries=('US','CA','GB','DE','FR','NL','PL','FI','SE','CZ','ES','IT','AT','RO','LT','BG'))
    if not quote or quote['hourly_ceiling']>2.65:raise RuntimeError('No eligible2A100 offer within quoted cost')
    original=Path(read(ROOT/'lab/experiments/replay-scaled/20261102-runpod-resume850/launch.json')['release'])
    if verify(original)!=read(ROOT/'lab/experiments/replay-scaled/20261102-runpod-resume850/launch.json')['payload_sha256']:
        raise RuntimeError('Parent source payload changed')
    olddata=original/'lab/experiments/data/replay-scaled'
    training=cloud.STATE/PARENT/'output/training';checkpoint=training/'checkpoints/group-001024'
    manifest=read(checkpoint/'complete.json')
    if (manifest['group']!=1024 or manifest['world_size']!=2 or manifest['pilot']
        or manifest['contract']!=sha(olddata/'plan.json')
        or any(sha(checkpoint/f)!=h for f,h in manifest['files'].items())):raise RuntimeError('Invalid complete1024 checkpoint')
    stamp=time.strftime('%Y%m%dt%H%M%Sz',time.gmtime()).lower();name='replay-scaled-vast-resume1024-'+stamp
    release=ROOT/'lab/isolated'/name
    ignore=shutil.ignore_patterns('__pycache__','.git','.env*','.venv*','*.log')
    for folder in ('experiments','ThinkingCap-Test'):
        shutil.copytree(original/folder,release/folder,ignore=ignore)
    # Preserve historical worker, reward, sampling, optimizer and graders byte-for-byte.
    overlays=['experiments/replay_scaled/run.sh','experiments/replay_scaled/resume_stage.py']
    for f in overlays:shutil.copy2(ROOT/f,release/f)
    protected=read(original/'payload-manifest.json')['files']
    scientific={f:h for f,h in protected.items() if f.startswith(('experiments/','ThinkingCap-Test/')) and f not in overlays}
    if any(sha(release/f)!=h for f,h in scientific.items()):raise RuntimeError('Scientific source changed')
    stage=release/'lab/experiments/data/replay-scaled';stage.mkdir(parents=True)
    REG.mkdir(parents=True)
    for f in ('plan.json','train.jsonl','development.jsonl','gpu-verification.json'):
        shutil.copy2(olddata/f,stage/f)
        if f!='gpu-verification.json':shutil.copy2(olddata/f,REG/f)
    shutil.copytree(training,stage/'resume-training',ignore=shutil.ignore_patterns('checkpoints','completion.json','scheduled-stop.json'))
    shutil.copytree(checkpoint,stage/'resume-training/checkpoints/group-001024')
    write(stage/'resume-files.json',{str(p.relative_to(stage/'resume-training')):sha(p) for p in (stage/'resume-training').rglob('*') if p.is_file()})
    write(stage/'launch-mode.json',dict(mode='training',resume_group=1024,stop_after=1280,verify_before_training=False))
    shutil.copy2(original/'lab/gpu-requirements.lock',release/'lab/gpu-requirements.lock')
    write(REG/'scientific-source-verification.json',dict(parent_release=str(original),files_sha256=scientific,launch_overlays=overlays))
    launch=dict(job_name=name,release=str(release),payload_sha256=seal(release),provider='vast',gpu=GPU,gpu_count=2,
        hardware_scope=SCOPE,entry=ENTRY,max_cost_usd=CAP,max_runtime_hours=10,
        max_gpu_rate=math.ceil(quote['hourly_ceiling']/2*1000)/1000,phase='training',offer_id=quote['offer_id'],disk_gb=60,direct_ssh=True,
        resume_parent=PARENT,resume_group=1024,resume_manifest_sha256=sha(checkpoint/'complete.json'),stop_after=1280)
    write(REG/'launch.json',launch);write(REG/'quote.json',quote)
    write(REG/'campaign.json',dict(state='prepared',job_name=name,total=3840,gpu_count=2,full_training_authorized=True,
        telegram_authorized=True,released=False,main_previous_jobs=[],main_authorization=dict(cap_usd=CAP,source_user_instruction=INSTRUCTION),
        credit_at_prepare=credit,resume_group=1024,requested_stop_after_validation=1280,no_auto_resume=True,created_at=time.time()))
    write(cloud.STATE/'authorizations'/(name+'.json'),dict(launch,authorized=True,scope='replay_vast_bounded_v1',source_user_instruction=INSTRUCTION))
    if not ticket(cloud.STATE,name,launch):raise RuntimeError('Authorization rejected')
    write(ROOT/'lab/experiments/replay-scaled/current.json',dict(path=str((REG/'campaign.json').relative_to(ROOT))))
    return launch

def run():
    replay_scaled.REG=REG
    name=read(REG/'launch.json')['job_name'];job=read(cloud.STATE/'ledger.json')['jobs'].get(name)
    if job and job.get('released_at'):return
    replay_scaled.run(observe_only=job is not None)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);a=p.parse_args()
    if a.action=='prepare':print(json.dumps(prepare(),indent=2))
    else:run()
