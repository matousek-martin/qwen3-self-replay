"""One inference-only mode diagnostic, including provenance of reused answers."""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments import cloud
from experiments.opencoder_sft.runtime import read,rows,sha,write
from experiments.opencoder_sft.mode_budget import SCOPE,PREFIX,ENTRY,GPU
from experiments.opencoder_sft.mode_probe import pair_check
from experiments.replay_scaled.notify import deliver_once
from ops import vastai
from ops.isolated_cloud import seal,verify
REG=ROOT/'lab/experiments/opencoder-mode-probe'
PARENT_JOB='opencoder-sft-768-20261103t222535z'


def prepare():
    if (REG/'current.json').exists():raise RuntimeError('Diagnostic already registered; reconcile')
    jobs=read(cloud.STATE/'ledger.json')['jobs']
    if any(n.startswith(PREFIX) for n in jobs) or vastai.list_pods():raise RuntimeError('Existing rental/diagnostic requires reconciliation')
    old=jobs[PARENT_JOB]
    if not old.get('released_at') or not old.get('backup_verified'):raise RuntimeError('SFT stop/archive not reconciled')
    cloud.verify_backup(PARENT_JOB)
    prior=ROOT/'lab/experiments/opencoder-sft/20261103t222535z';oldlaunch=read(prior/'launch.json');oldstage=Path(oldlaunch['release'])/'lab/experiments/data/opencoder-sft'
    if verify(oldlaunch['release'])!=oldlaunch['payload_sha256']:raise RuntimeError('Original payload changed')
    archive=cloud.STATE/PARENT_JOB/'output';cp=archive/'training/checkpoints/step-000016'
    for f,h in read(cp/'complete.json')['files'].items():
        if sha(cp/f)!=h:raise RuntimeError('Checkpoint16 corrupt')
    credit=float(vastai.api('v0/users/current/')['credit'])
    if credit<6:raise RuntimeError('Need4USD plus2USD reserve')
    excluded={j.get('provider_quote',{}).get('offer',{}).get('machine_id') for j in jobs.values() if j.get('startup_failed_before_launch')}
    quote=vastai.quote(GPU,1,60,min_duration=5*3600,exclude_machine_ids=excluded-{None},allowed_countries=('US','CA','GB','DE','FR','NL','PL','FI','SE','CZ','ES','IT','AT','RO','LT','BG'))
    if not quote or quote['hourly_ceiling']>.6:raise RuntimeError('No suitable4090')
    quote=vastai.quote(GPU,1,60,quote['offer_id'],5*3600)
    if not quote or quote['hourly_ceiling']>.6:raise RuntimeError('Exact offer unavailable')
    stamp=time.strftime('%Y%m%dt%H%M%Sz',time.gmtime()).lower();name=PREFIX+stamp;reg=REG/stamp;reg.mkdir(parents=True)
    release=ROOT/'lab/isolated'/name
    shutil.copytree(ROOT/'experiments',release/'experiments',ignore=shutil.ignore_patterns('__pycache__','.git','.env*','.venv*','*.log'))
    (release/'ThinkingCap-Test').mkdir()
    stage=release/'lab/experiments/data/opencoder-mode-probe';stage.mkdir(parents=True)
    shutil.copytree(oldstage/'adapter',stage/'adapters/parent768');shutil.copytree(cp/'adapter',stage/'adapters/sft16')
    shutil.copy2(oldstage/'monitor.jsonl',stage/'questions.jsonl');questions=rows(stage/'questions.jsonl');ids={q['id'] for q in questions}
    provenance=dict(original_job=PARENT_JOB,original_payload_sha256=oldlaunch['payload_sha256'],development_sha256=sha(oldstage/'monitor.jsonl'),
        parent_adapter_sha256=sha(oldstage/'adapter/adapter_model.safetensors'),candidate_adapter_sha256=sha(cp/'adapter/adapter_model.safetensors'),
        candidate_checkpoint_manifest_sha256=sha(cp/'complete.json'),source_gpu_model=GPU,source_physical_gpu_reused=False,cached_records={})
    for label,step in [('parent768-on',0),('sft16-on',16)]:
        path=archive/'validation'/f'{step:06d}'/'records.jsonl';record=[x for x in rows(path) if x['id'] in ids];pair_check(record,questions)
        receipt=read(path.parent/'complete.json')
        if not receipt['complete'] or sha(path)!=receipt['records_sha256']:raise RuntimeError('Cached validation incomplete')
        (stage/(label+'.jsonl')).write_text(''.join(json.dumps(x)+'\n' for x in record))
        provenance['cached_records'][label]=dict(source=str(path),sha256=sha(path),original_responses=len(rows(path)),selected_responses=len(record),mode=True)
    versions={}
    wanted=['torch','transformers','vllm','peft','tokenizers','safetensors']
    for line in (archive/'packages.txt').read_text().splitlines():
        if '==' in line:
            key,value=line.split('==',1)
            if key in wanted:versions[key]=value
    if set(versions)!=set(wanted):raise RuntimeError('Missing source runtime versions')
    config=dict(scope=SCOPE,inference_only=True,training_authorized=False,enable_thinking=False,questions=192,samples=2,
        max_new_tokens=2048,temperature=.6,top_p=.95,top_k=20,seed_protocol='sha256 opencoder-dev:id:sample:9127, first8hex',
        mode_aware_completion='OFF: nonempty final answer with no reasoning tags; ON cached requires completed nonempty thinking. Domain graders unchanged.',
        conditions=['parent768-on','parent768-off','sft16-on','sft16-off'],runtime_versions=versions,provenance=provenance,
        files_sha256={str(p.relative_to(stage)):sha(p) for p in stage.rglob('*') if p.is_file()})
    write(stage/'plan.json',config);write(reg/'plan.json',config)
    preload=ROOT/'lab/experiments/replay-scaled-evaluation/20261102t102548z';target=release/'lab/experiments/data/replay-evaluation';target.mkdir()
    shutil.copy2(preload/'model-preload-source-verification.json',target/'model-preload.json');shutil.copy2(preload/'uv-wheel-source.json',target/'uv-wheel-source.json')
    shutil.copy2(ROOT/'lab/gpu-requirements.lock',release/'lab/gpu-requirements.lock');digest=seal(release)
    launch=dict(job_name=name,release=str(release),payload_sha256=digest,authorized=True,scope=SCOPE,hardware_scope=SCOPE,entry=ENTRY,
        provider='vast',gpu=GPU,gpu_count=1,offer_id=quote['offer_id'],disk_gb=60,max_cost_usd=4,max_runtime_hours=4,
        max_gpu_rate=math.ceil(quote['hourly_ceiling']*1000)/1000,evaluation_only=True,training_authorized=False,questions=192,samples=2,
        conditions=['parent768-off','sft16-off'],direct_ssh=True,source_user_instruction='2026-11-04: User approves proposed inference-only parent768/SFT16 thinking-on/off development diagnostic, reuse completed on-mode records; no new training.')
    write(reg/'launch.json',launch);write(reg/'quote.json',quote);write(reg/'campaign.json',dict(state='prepared',job_name=name,created_at=time.time(),credit_at_prepare=credit,cap_usd=4))
    write(REG/'current.json',dict(directory=str(reg)));return reg


def prepare_repair():
    old=Path(read(REG/'current.json')['directory']);launch=read(old/'launch.json');jobs=read(cloud.STATE/'ledger.json')['jobs']
    previous=launch.get('previous_attempts',[])+[launch['job_name']]
    if len(previous)!=len(set(previous)):raise RuntimeError('Duplicate attempt list')
    spent=0.
    for n in previous:
        j=jobs[n]
        if not j.get('released_at') or not j.get('startup_failed_before_launch') or j.get('worker_launch_attempted_at'):raise RuntimeError('Prior worker not a reconciled pre-launch failure')
        spent+=j['charged_estimate_usd']
    cap=round(4-spent,4)
    if cap<=1 or vastai.list_pods():raise RuntimeError('Budget or existing rental prevents repair')
    if verify(launch['release'])!=launch['payload_sha256']:raise RuntimeError('Scientific payload changed')
    credit=float(vastai.api('v0/users/current/')['credit'])
    if credit<cap+2:raise RuntimeError('Insufficient remaining credit')
    excluded={j.get('provider_quote',{}).get('offer',{}).get('machine_id') for j in jobs.values() if j.get('startup_failed_before_launch')}
    quote=vastai.quote(GPU,1,60,min_duration=5*3600,exclude_machine_ids=excluded-{None},allowed_countries=('US','CA','GB','DE','FR','NL','PL','FI','SE','CZ','ES','IT','AT','RO','LT','BG'))
    if not quote or quote['hourly_ceiling']>.6:raise RuntimeError('No suitable replacement4090')
    quote=vastai.quote(GPU,1,60,quote['offer_id'],5*3600)
    if not quote or quote['hourly_ceiling']>.6:raise RuntimeError('Replacement offer unavailable')
    stamp=time.strftime('%Y%m%dt%H%M%Sz',time.gmtime()).lower();name=PREFIX+stamp;reg=REG/stamp;reg.mkdir()
    launch.update(job_name=name,offer_id=quote['offer_id'],max_gpu_rate=math.ceil(quote['hourly_ceiling']*1000)/1000,
                  max_cost_usd=cap,previous_attempts=previous,combined_cap_usd=4)
    write(reg/'launch.json',launch);write(reg/'plan.json',read(old/'plan.json'));write(reg/'quote.json',quote)
    write(reg/'campaign.json',dict(state='prepared',job_name=name,created_at=time.time(),cap_usd=cap,credit_at_prepare=credit,previous_attempts=previous,combined_cap_usd=4))
    write(REG/'current.json',dict(directory=str(reg)));return reg


def run(reg):
    launch=read(reg/'launch.json');name=launch['job_name'];campaign=read(reg/'campaign.json')
    if verify(launch['release'])!=launch['payload_sha256']:raise RuntimeError('Payload changed')
    jobs=read(cloud.STATE/'ledger.json')['jobs']
    if name in jobs:command=[sys.executable,str(ROOT/'ops/isolated_cloud.py'),'watch','--release',launch['release'],'--name',name]
    else:
        if vastai.list_pods() or float(vastai.api('v0/users/current/')['credit'])<6:raise RuntimeError('Allocation/credit changed')
        write(cloud.STATE/'authorizations'/(name+'.json'),launch)
        command=[sys.executable,str(ROOT/'ops/isolated_cloud.py'),'run','--release',launch['release'],'--name',name,'--entry',ENTRY,'--gpu',GPU,
            '--gpu-count','1','--hardware-scope',SCOPE,'--hours','1','--max-cost-usd',str(launch['max_cost_usd']),'--max-runtime-hours','4','--provider','vast',
            '--vast-offer-id',str(launch['offer_id']),'--vast-disk-gb','60','--data-upload-timeout','1800']
    campaign.update(state='starting');write(reg/'campaign.json',campaign)
    with (reg/'controller.log').open('a') as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        while process.poll() is None:
            job=read(cloud.STATE/'ledger.json')['jobs'].get(name,{})
            if job.get('state')=='running' and job.get('pod_id') and not job.get('released_at'):
                try:
                    pod=cloud.api('pods/'+job['pod_id'])
                    script="import json;from pathlib import Path;p=Path('/workspace/project/output/opencoder-mode-probe');print(json.dumps({n:json.loads((p/n).read_text()) for n in ['progress.json','results.json','preflight.json','failure.json'] if (p/n).exists()}))"
                    live=json.loads(cloud.remote(pod,'python3 -c '+shlex.quote(script),timeout=20));campaign.update(state='running',observed_at=time.time(),live=live);campaign.pop('monitor_error',None)
                except Exception as e:campaign['monitor_error']=str(e)[:300]
                write(reg/'campaign.json',campaign)
            time.sleep(20)
    job=read(cloud.STATE/'ledger.json')['jobs'].get(name,{});output=cloud.STATE/name/'output/opencoder-mode-probe'
    done=read(output/'completion.json') if (output/'completion.json').exists() else {}
    complete=bool(job.get('released_at') and job.get('backup_verified') and done.get('complete') and sha(output/'results.json')==done['results_sha256'])
    campaign.update(state='complete' if complete else 'failed',backup_verified=bool(job.get('backup_verified')),released=bool(job.get('released_at')),output=str(output),finished_at=time.time())
    lines=['<b>Project: ThinkingCap / OpenCoder mode diagnostic</b>','Summary: '+('Porovnání thinking ON/OFF dokončeno; žádný nový trénink.' if complete else 'Diagnostika nedokončena; výsledky zachovány.'),'GPU: 1× RTX 4090 (Vast)','Development: 192 otázek ×2; thinking ON z ověřeného archivu.']
    if complete:
        result=read(output/'results.json');campaign['results']=result
        for label in ['parent768-on','parent768-off','sft16-on','sft16-off']:
            m=result['conditions'][label];lines+=['',label,f"Celkem: {100*m['overall']['accuracy']:.1f}% · {m['overall']['mean_tokens']:.0f} tokenů",
                f"Kódování: {100*m['coding']['accuracy']:.1f}% · useknuto {100*m['coding']['truncation']:.1f}%",
                f"Instrukce: {100*m['instructions']['accuracy']:.1f}%",f"OpenCoder: {100*m['opencoder']['accuracy']:.1f}%"]
    if (output/'failure.json').exists():lines+=['Důvod: '+read(output/'failure.json')['error']]
    lines+=['','GPU uvolněna: '+('ano' if job.get('released_at') else 'ne'),'Timestamp: '+datetime.now(ZoneInfo('Europe/Prague')).strftime('%Y-%m-%d %H:%M:%S %Z (Praha)')]
    write(reg/'campaign.json',campaign);deliver_once(reg/'notifications.sqlite3',name+':terminal','\n'.join(lines))
    subprocess.run([sys.executable,str(ROOT/'experiment-ui/publish.py')],check=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','prepare-repair','run']);p.add_argument('--directory',type=Path);a=p.parse_args()
    if a.action=='prepare':print(prepare())
    elif a.action=='prepare-repair':print(prepare_repair())
    else:run(a.directory or Path(read(REG/'current.json')['directory']))
