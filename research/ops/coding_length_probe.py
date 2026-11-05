"""Register and run one capped coding inference comparison using canonical cloud."""
import argparse
import collections
from datetime import datetime
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments import cloud
from experiments.replay_scaled.core import write,sha
from experiments.replay_scaled.length_budget import SCOPE,PREFIX,ENTRY,GPU,remaining_budget
from experiments.replay_scaled.notify import deliver_once
from ops import vastai
from ops.isolated_cloud import seal,verify

REG=ROOT/'lab/experiments/coding-length-probe'
PARENT='replay-scaled-training-20261101t122440z'


def read(path):return json.loads(Path(path).read_text())


def prepare_repair():
    old=Path(read(REG/'current.json')['directory']);launch=read(old/'launch.json')
    jobs=read(cloud.STATE/'ledger.json')['jobs']
    previous=launch.get('previous_attempts',[])+[launch['job_name']]
    if len(previous)!=len(set(previous)):raise RuntimeError('Duplicate previous probe attempts')
    cap=remaining_budget([jobs[n] for n in previous])
    if vastai.list_pods():raise RuntimeError('Reconcile existing Vast rentals before repair')
    credit=float(vastai.api('v0/users/current/')['credit'])
    if credit<cap+2:raise RuntimeError('Insufficient credit for remaining bounded repair')
    if verify(launch['release'])!=launch['payload_sha256']:raise RuntimeError('Scientific probe payload changed')
    excluded={j.get('provider_quote',{}).get('offer',{}).get('machine_id') for j in jobs.values() if j.get('startup_failed_before_launch')}
    quote=vastai.quote(GPU,1,60,min_duration=7*3600,exclude_machine_ids=excluded-{None},
        allowed_countries=('GB','DE','FR','NL','PL','FI','SE','CZ','ES','IT','AT','RO','LT','BG'))
    if not quote or quote['hourly_ceiling']>.5:raise RuntimeError('No suitable alternate European 4090')
    quote=vastai.quote(GPU,1,60,quote['offer_id'],6.5*3600)
    if not quote or quote['hourly_ceiling']>.5:raise RuntimeError('Exact offer unavailable')
    stamp=time.strftime('%Y%m%dt%H%M%Sz',time.gmtime()).lower();name=PREFIX+stamp;reg=REG/stamp;reg.mkdir()
    launch.update(job_name=name,offer_id=quote['offer_id'],max_gpu_rate=math.ceil(quote['hourly_ceiling']*1000)/1000,
        max_cost_usd=cap,previous_attempts=previous,combined_cap_usd=4,
        source_user_instruction='2026-11-02: User explicitly requests diagnosing and repairing failed coding-length test, within original4USD cap; RunPod training remains unchanged.')
    write(reg/'launch.json',launch);write(reg/'quote.json',quote);write(reg/'plan.json',read(old/'plan.json'))
    write(reg/'campaign.json',dict(state='prepared',job_name=name,created_at=time.time(),cap_usd=cap,combined_cap_usd=4,
        previous_attempts=previous,credit_at_prepare=credit,checkpoint_group=850,questions=47,samples=4,caps=launch['caps']))
    write(REG/'current.json',dict(directory=str(reg)))
    return reg


def prepare():
    if (REG/'current.json').exists(): raise RuntimeError('Probe already registered; inspect before any new rental')
    if vastai.list_pods(): raise RuntimeError('Reconcile existing Vast rentals before probe')
    jobs=read(cloud.STATE/'ledger.json')['jobs']
    if any(n.startswith(PREFIX) for n in jobs): raise RuntimeError('Existing probe must be reconciled')
    credit=float(vastai.api('v0/users/current/')['credit'])
    if credit<6: raise RuntimeError('Insufficient Vast credit for 4USD cap plus reserve')
    stat=__import__('os').statvfs(ROOT)
    if stat.f_bfree*stat.f_frsize<4*1024**3: raise RuntimeError('Insufficient local backup space')
    checkpoint=cloud.STATE/PARENT/'output/training/checkpoints/group-000850'
    manifest=read(checkpoint/'complete.json')
    if manifest['group']!=850 or manifest['pilot'] or not jobs[PARENT].get('backup_verified'): raise RuntimeError('Unverified parent checkpoint')
    if any(sha(checkpoint/f)!=h for f,h in manifest['files'].items()): raise RuntimeError('Checkpoint SHA mismatch')
    source=ROOT/'lab/experiments/replay-scaled/20261101-checkpoints-v5';original=read(source/'plan.json')
    if manifest['contract']!=sha(source/'plan.json'): raise RuntimeError('Checkpoint training contract mismatch')
    rows=[json.loads(l) for l in (source/'development.jsonl').open()]
    selected=[r for r in rows if r['task']=='coding' and r.get('difficulty') in ('introductory','interview')]
    if collections.Counter(r['difficulty'] for r in selected)!={'introductory':22,'interview':25}: raise RuntimeError('Cohort changed')
    train=[json.loads(l) for l in (source/'train.jsonl').open()]
    if {r['id'] for r in selected}&{r['id'] for r in train}: raise RuntimeError('Development/training ID overlap')
    if {r['split_family'] for r in selected}&{r['split_family'] for r in train if 'split_family' in r}: raise RuntimeError('Development/training family overlap')
    excluded={j.get('provider_quote',{}).get('offer',{}).get('machine_id') for j in jobs.values() if j.get('startup_failed_before_launch')}
    quote=vastai.quote(GPU,1,60,min_duration=6.5*3600,exclude_machine_ids=excluded-{None},
        allowed_countries=('US','CA','GB','DE','FR','NL','PL','FI','SE','CZ','ES','IT','AT','RO','LT','BG'))
    if not quote or quote['hourly_ceiling']>.5: raise RuntimeError('No suitable bounded RTX4090 offer')
    stamp=time.strftime('%Y%m%dt%H%M%Sz',time.gmtime()).lower();name=PREFIX+stamp;reg=REG/stamp;reg.mkdir(parents=True)
    release=ROOT/'lab/isolated'/name
    # No official evaluator/cohort or shared baseline is needed for this distinct probe.
    shutil.copytree(ROOT/'experiments',release/'experiments',ignore=shutil.ignore_patterns('__pycache__','.env*','.venv*','study_eval.py'))
    (release/'ThinkingCap-Test').mkdir()
    stage=release/'lab/experiments/data/coding-length-probe';stage.mkdir(parents=True)
    shutil.copytree(checkpoint/'adapter',stage/'adapter')
    (stage/'questions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected))
    preload=ROOT/'lab/experiments/replay-scaled-evaluation/20261102t102548z'
    shutil.copy2(preload/'model-preload-source-verification.json',stage/'model-preload.json')
    uv=release/'lab/experiments/data/replay-evaluation';uv.mkdir()
    shutil.copy2(preload/'uv-wheel-source.json',uv/'uv-wheel-source.json')
    (release/'lab').mkdir(exist_ok=True);shutil.copy2(ROOT/'lab/gpu-requirements.lock',release/'lab/gpu-requirements.lock')
    config=dict(schema_version=1,checkpoint_group=850,caps=[2048,4096,8192],samples=4,questions=47,
        seed=17001,batch_size=8,temperature=original['temperature'],top_k=original['top_k'],top_p=original['top_p'],
        training_authorized=False,protocol='paired-prefix-hf-row-seeded-v1',
        source_development_sha256=sha(source/'development.jsonl'),source_train_sha256=sha(source/'train.jsonl'),
        checkpoint_manifest_sha256=sha(checkpoint/'complete.json'),
        selection='All independent-development introductory/interview coding questions; selection by metadata, not outcomes. No competition tasks.',
        decision='Compare paired question-level correctness gain, truncation, token/time cost; select a useful tested cap, not claim a globally optimal length. No training automatically follows.',
        files_sha256={str(p.relative_to(stage)):sha(p) for p in stage.rglob('*') if p.is_file()})
    write(stage/'plan.json',config);write(reg/'plan.json',config)
    digest=seal(release)
    launch=dict(job_name=name,release=str(release),payload_sha256=digest,provider='vast',hardware_scope=SCOPE,entry=ENTRY,
        gpu=GPU,gpu_count=1,offer_id=quote['offer_id'],disk_gb=60,max_cost_usd=4,max_runtime_hours=6,
        max_gpu_rate=math.ceil(quote['hourly_ceiling']*1000)/1000,phase='evaluation',evaluation_only=True,training_authorized=False,
        checkpoint_group=850,caps=config['caps'],samples=4,questions=47,direct_ssh=True,
        source_user_instruction='2026-11-02: User requests scheduling coding-length test on Vast; latest clarification leaves existing RunPod training running. Inference-only checkpoint850 test before a new training branch.')
    write(reg/'launch.json',launch);write(reg/'quote.json',quote)
    write(reg/'campaign.json',dict(state='prepared',job_name=name,cap_usd=4,credit_at_prepare=credit,created_at=time.time(),checkpoint_group=850,questions=47,samples=4,caps=config['caps']))
    write(REG/'current.json',dict(directory=str(reg)))
    return reg


def run(reg):
    launch=read(reg/'launch.json');name=launch['job_name'];campaign=read(reg/'campaign.json')
    if verify(launch['release'])!=launch['payload_sha256']: raise RuntimeError('Probe payload changed')
    jobs=read(cloud.STATE/'ledger.json')['jobs']
    if name in jobs:
        command=[sys.executable,str(ROOT/'ops/isolated_cloud.py'),'watch','--release',launch['release'],'--name',name]
    else:
        if vastai.list_pods(): raise RuntimeError('Other Vast rental appeared; refusing duplicate')
        write(cloud.STATE/'authorizations'/(name+'.json'),dict(launch,authorized=True,scope=SCOPE))
        command=[sys.executable,str(ROOT/'ops/isolated_cloud.py'),'run','--release',launch['release'],'--name',name,'--entry',ENTRY,
            '--gpu',GPU,'--gpu-count','1','--hardware-scope',SCOPE,'--hours','1','--max-cost-usd',str(launch['max_cost_usd']),'--max-runtime-hours','6',
            '--provider','vast','--vast-offer-id',str(launch['offer_id']),'--vast-disk-gb','60','--data-upload-timeout','1800']
    campaign.update(state='starting');write(reg/'campaign.json',campaign)
    with (reg/'controller.log').open('a') as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        while process.poll() is None:
            job=read(cloud.STATE/'ledger.json')['jobs'].get(name,{})
            if job.get('state')=='running' and job.get('pod_id') and not job.get('released_at'):
                try:
                    pod=cloud.api('pods/'+job['pod_id'])
                    script="import json;from pathlib import Path;p=Path('/workspace/project/output/coding-length-probe');print(json.dumps({n:json.loads((p/n).read_text()) for n in ['progress.json','results.json','preflight.json'] if (p/n).exists()}))"
                    import shlex
                    observed=json.loads(cloud.remote(pod,'python3 -c '+shlex.quote(script),timeout=20))
                    campaign.update(state='running' if observed.get('progress.json') else 'starting',observed_at=time.time(),live=observed)
                    campaign.pop('monitor_error',None)
                except Exception as exc: campaign['monitor_error']=str(exc)[:300]
                write(reg/'campaign.json',campaign)
            time.sleep(30)
    job=read(cloud.STATE/'ledger.json')['jobs'].get(name,{})
    output=cloud.STATE/name/'output/coding-length-probe'
    complete=bool(job.get('backup_verified') and job.get('released_at') and (output/'completion.json').exists() and read(output/'completion.json').get('complete'))
    campaign.update(state='complete' if complete else 'failed',backup_verified=bool(job.get('backup_verified')),released=bool(job.get('released_at')),output=str(output),finished_at=time.time())
    lines=['<b>Project: ThinkingCap / Coding length probe</b>','Summary: '+('Test délky dokončen; žádný nový trénink.' if complete else 'Test délky selhal; kontroluji archivované výsledky.'),
        'Checkpoint: 850','Otázky: 47 × 4 odpovědi · introductory/interview','GPU: 1× RTX 4090 (Vast)']
    if complete:
        results=read(output/'results.json');campaign['results']=results
        for row in results['metrics']:
            if row['difficulty']!='all':continue
            lines+=['',f"Limit: {row['cap']} tokenů",f"Accuracy: {100*row['accuracy']:.1f}% ({row['correct']}/{row['answers']})",f"Useknuté odpovědi: {row['truncated']}/{row['answers']}",f"Tokeny / odpověď: {row['mean_tokens']:.0f}"]
    lines+=['','GPU uvolněna: '+('ano' if job.get('released_at') else 'ne'),'Timestamp: '+datetime.now(ZoneInfo('Europe/Prague')).strftime('%Y-%m-%d %H:%M:%S %Z (Praha)')]
    write(reg/'campaign.json',campaign)
    deliver_once(reg/'notifications.sqlite3',name+(':complete' if complete else ':terminal'),'\n'.join(lines))
    subprocess.run([sys.executable,str(ROOT/'experiment-ui/publish.py')],check=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','prepare-repair','run']);parser.add_argument('--directory',type=Path);args=parser.parse_args()
    if args.action=='prepare': print(prepare())
    elif args.action=='prepare-repair':print(prepare_repair())
    else: run(args.directory or Path(read(REG/'current.json')['directory']))
