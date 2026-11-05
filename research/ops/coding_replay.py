"""Approved768 coding branch; canonical allocation, monitoring, backup and release."""
import argparse
import collections
from datetime import datetime
import json
import math
from pathlib import Path
import random
import shlex
import shutil
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments import cloud
from experiments.replay_scaled.core import sha,write
from experiments.replay_scaled.coding_budget import SCOPE,PREFIX,ENTRY,GPU
from experiments.replay_scaled.notify import deliver_once,validation_message
from ops import vastai
from ops.isolated_cloud import seal,verify

REG=ROOT/'lab/experiments/coding-replay'


def read(p):return json.loads(Path(p).read_text())


def prepare():
    if (REG/'current.json').exists():raise RuntimeError('Coding branch already registered; reconcile instead of duplicating')
    jobs=read(cloud.STATE/'ledger.json')['jobs']
    if any(n.startswith(PREFIX) for n in jobs) or vastai.list_pods():raise RuntimeError('Existing allocation requires reconciliation')
    credit=float(vastai.api('v0/users/current/')['credit'])
    if credit<20:raise RuntimeError('Need18USD plus2USD account reserve')
    import os
    stat=os.statvfs(ROOT)
    if stat.f_bfree*stat.f_frsize<5*1024**3:raise RuntimeError('Insufficient verified-backup space')
    proposal=ROOT/'lab/experiments/coding-replay-proposal/20261103';spec=read(proposal/'proposal.json')
    checkpoint=ROOT/spec['checkpoint_path'];manifest=read(checkpoint/'complete.json')
    if sha(checkpoint/'complete.json')!=spec['checkpoint_manifest_sha256']:raise RuntimeError('Parent changed')
    if any(sha(checkpoint/f)!=digest for f,digest in manifest['files'].items()):raise RuntimeError('Incomplete768 parent')
    source=ROOT/'lab/experiments/replay-scaled/20261101-checkpoints-v5';original=read(source/'plan.json')
    for f,digest in original['files_sha256'].items():
        if sha(source/f)!=digest:raise RuntimeError('Original data changed')
    excluded={j.get('provider_quote',{}).get('offer',{}).get('machine_id') for j in jobs.values() if j.get('startup_failed_before_launch')}
    quote=vastai.quote(GPU,1,80,min_duration=25*3600,exclude_machine_ids=excluded-{None},
        allowed_countries=('US','CA','GB','DE','FR','NL','PL','FI','SE','CZ','ES','IT','AT','RO','LT','BG'))
    if not quote or quote['hourly_ceiling']>.5:raise RuntimeError('No suitable RTX4090 offer')
    stamp=time.strftime('%Y%m%dt%H%M%Sz',time.gmtime()).lower();name=PREFIX+stamp;reg=REG/stamp;reg.mkdir(parents=True)
    release=ROOT/'lab/isolated'/name
    ignore=shutil.ignore_patterns('__pycache__','.git','.env*','.venv*','runs','*.log')
    shutil.copytree(ROOT/'experiments',release/'experiments',ignore=ignore)
    frozen=ROOT/'lab/isolated/full-evaluation-20261026/teacher-student-v2'
    shutil.copytree(frozen/'ThinkingCap-Test',release/'ThinkingCap-Test',ignore=ignore)
    stage=release/'lab/experiments/data/coding-replay';stage.mkdir(parents=True)
    shutil.copytree(checkpoint/'adapter',stage/'adapter')
    for namefile in ('screening-candidates.jsonl','candidate-metadata.json'):
        shutil.copy2(proposal/namefile,stage/namefile)
    shutil.copy2(source/'development.jsonl',stage/'development.jsonl')
    shutil.copy2(checkpoint.parent.parent/'references.json',stage/'references.json')
    rows=[json.loads(l) for l in (source/'train.jsonl').open()]
    retain=[]
    for domain,number in [('math',10),('instructions',10),('tools',12)]:
        pool=[r for r in rows if r['task']==domain];random.Random(17).shuffle(pool)
        if domain=='tools':
            negative=[r for r in pool if not json.loads(r['answer'])['calls']][:4]
            pool=negative+[r for r in pool if json.loads(r['answer'])['calls']]
        retain.extend(pool[:number])
    random.Random(19).shuffle(retain)
    assert len({r['id'] for r in retain})==32
    (stage/'retention.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in retain))
    candidates=[json.loads(l) for l in (stage/'screening-candidates.jsonl').open()]
    byid={r['id']:r for r in rows}
    assert all(byid[r['id']]==r and r['split']=='train' for r in candidates+retain)
    dev=[json.loads(l) for l in (stage/'development.jsonl').open()]
    assert not {r['id'] for r in candidates+retain}&{r['id'] for r in dev}
    plan=dict(original,branch=SCOPE,parent_group=768,groups=128,gpu_count=1,allowed_gpu_counts=[1],
        milestones=[32,64,96,128],validate_every=32,save_every=32,quality_stopping=False,
        generation=dict(original['generation'],evaluation_batch_size=16,max_padded_tokens=65536),
        initialization='Parent768 adapter, fresh optimizer/RNG, verified screening replay',replay_admission='any_verified_correct')
    config=dict(schema_version=1,parent_group=768,stage_caps_usd={'screening':4,'training':10,'evaluation':4},
        screening_questions=128,samples=4,max_new_tokens=2048,training_plan=plan,
        parent_manifest_sha256=sha(checkpoint/'complete.json'),proposal_sha256=sha(proposal/'proposal.json'),
        files_sha256={str(p.relative_to(stage)):sha(p) for p in stage.rglob('*') if p.is_file()})
    write(stage/'plan.json',config);write(reg/'plan.json',config)
    evaluation=release/'lab/experiments/data/replay-evaluation';evaluation.mkdir()
    preload=ROOT/'lab/experiments/replay-scaled-evaluation/20261102t102548z'
    shutil.copy2(preload/'model-preload-source-verification.json',evaluation/'model-preload.json')
    shutil.copy2(preload/'uv-wheel-source.json',evaluation/'uv-wheel-source.json')
    shutil.copy2(ROOT/'lab/experiments/data/study/splits.json',evaluation/'splits.json')
    shutil.copy2(ROOT/'lab/gpu-requirements.lock',release/'lab/gpu-requirements.lock')
    digest=seal(release)
    from experiments.baseline_cache import check_launch
    check_launch(release,ENTRY,ROOT/'lab/experiments/baseline-cache',GPU)
    launch=dict(job_name=name,release=str(release),payload_sha256=digest,authorized=True,scope=SCOPE,
        provider='vast',hardware_scope=SCOPE,entry=ENTRY,gpu=GPU,gpu_count=1,offer_id=quote['offer_id'],disk_gb=80,
        max_cost_usd=18,max_runtime_hours=24,max_gpu_rate=math.ceil(quote['hourly_ceiling']*1000)/1000,
        phase='screening_training_evaluation',training_authorized=True,evaluation_only=False,
        checkpoint_group=768,screening_questions=128,training_groups=128,stage_caps_usd=config['stage_caps_usd'],direct_ssh=True,
        source_user_instruction='2026-11-03: User approved checkpoint768, prepared128introductory/interview screening tasks, coding-focused128-step replay pilot, full evaluation, Vast, total18USD (4/10/4), UI/Telegram.')
    write(reg/'launch.json',launch);write(reg/'quote.json',quote)
    write(reg/'campaign.json',dict(state='prepared',job_name=name,created_at=time.time(),credit_at_prepare=credit,cap_usd=18,
        parent_group=768,training_groups=128,telegram_authorized=True,proposal=str(proposal)))
    write(REG/'current.json',dict(directory=str(reg)))
    return reg


def notify(reg,key,message):
    try:deliver_once(reg/'notifications.sqlite3',key,message)
    except Exception as e:
        write(reg/'notification-error.json',dict(key=key,error=type(e).__name__,time=time.time()))


def run(reg):
    launch=read(reg/'launch.json');name=launch['job_name'];campaign=read(reg/'campaign.json')
    if verify(launch['release'])!=launch['payload_sha256']:raise RuntimeError('Sealed payload changed')
    jobs=read(cloud.STATE/'ledger.json')['jobs']
    if name in jobs:
        command=[sys.executable,str(ROOT/'ops/isolated_cloud.py'),'watch','--release',launch['release'],'--name',name]
    else:
        if vastai.list_pods():raise RuntimeError('Unexpected existing Vast rental; refusing duplicate')
        if float(vastai.api('v0/users/current/')['credit'])<20:raise RuntimeError('Credit no longer covers authorized cap and reserve')
        write(cloud.STATE/'authorizations'/(name+'.json'),launch)
        command=[sys.executable,str(ROOT/'ops/isolated_cloud.py'),'run','--release',launch['release'],'--name',name,'--entry',ENTRY,
            '--gpu',GPU,'--gpu-count','1','--hardware-scope',SCOPE,'--hours','1','--max-cost-usd','18','--max-runtime-hours','24',
            '--provider','vast','--vast-offer-id',str(launch['offer_id']),'--vast-disk-gb','80','--data-upload-timeout','1800']
    campaign.update(state='starting');write(reg/'campaign.json',campaign)
    with (reg/'controller.log').open('a') as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        while process.poll() is None:
            job=read(cloud.STATE/'ledger.json')['jobs'].get(name,{})
            if job.get('state')=='running' and job.get('pod_id') and not job.get('released_at'):
                try:
                    pod=cloud.api('pods/'+job['pod_id'])
                    script="import json;from pathlib import Path;p=Path('/workspace/project/output');names=['coding-campaign/progress.json','coding-campaign/completion.json','coding-campaign/failure.json','coding-screen/progress.json','coding-screen/preflight.json','coding-screen/results.json','coding-screen/completion.json','training/progress.json','training/latest-checkpoint.json','training/training-series.json','training/validation-curve.json','gpu-verification.json','full-evaluation/completion.json'];print(json.dumps({n:json.loads((p/n).read_text()) for n in names if (p/n).exists()}))"
                    live=json.loads(cloud.remote(pod,'python3 -c '+shlex.quote(script),timeout=20))
                    campaign.update(state='running',observed_at=time.time(),live=live);campaign.pop('monitor_error',None)
                    points=live.get('training/validation-curve.json',{}).get('checkpoints',[])
                    for point in points:
                        if not point['iteration']:continue
                        progress=live.get('training/progress.json',{})
                        msg=validation_message(dict(progress,current=point['iteration']),point,points[0])
                        msg=msg.replace('Self-replay scaled','Coding replay · parent768').replace('1× A100-SXM4-80GB','1× RTX 4090 (Vast)')
                        notify(reg,name+':validation:'+str(point['iteration']),msg)
                except Exception as e:campaign['monitor_error']=str(e)[:300]
                write(reg/'campaign.json',campaign)
            time.sleep(30)
    job=read(cloud.STATE/'ledger.json')['jobs'].get(name,{})
    output=cloud.STATE/name/'output';done=read(output/'coding-campaign/completion.json') if (output/'coding-campaign/completion.json').exists() else {}
    complete=bool(job.get('backup_verified') and job.get('released_at') and done.get('complete'))
    campaign.update(state='complete' if complete else 'stopped',backup_verified=bool(job.get('backup_verified')),released=bool(job.get('released_at')),output=str(output),finished_at=time.time())
    title='Pokus dokončen.' if complete else 'Běh ukončen před dokončením; uložené výstupy zachovány.'
    if done.get('reason')=='insufficient_eligible_questions':title='Výběr dokončen, méně než32 vhodných úloh; trénink nespouštěn.'
    lines=['<b>Project: ThinkingCap / Coding replay · parent768</b>','Summary: '+title,'GPU: 1× RTX 4090 (Vast)']
    if (output/'full-evaluation/comparison.json').exists() and complete and done.get('evaluation_complete'):
        result=read(output/'full-evaluation/comparison.json');campaign['official_results']=result
        a=result['average'];lines += [f'Accuracy proti původní baseline: {100*a["accuracy_delta"]:+.2f} p. b.',f'Úspora tokenů: {100*a["token_reduction"]:.1f} %',
            f'Kódování: {100*result["results"]["mbpp"]["candidate"]["accuracy"]:.2f} %','Plná evaluace: 2759 × 8; baseline z cache.']
        parent=read(cloud.STATE/'replay-evaluation-768-20261102t121206z/output/full-evaluation/comparison.json')
        if parent['cohort_sha256']!=result['cohort_sha256']:raise RuntimeError('Parent comparison cohort changed')
        lines += [f'Změna celkové accuracy proti checkpointu768: {100*(a["accuracy_delta"]-parent["average"]["accuracy_delta"]):+.2f} p. b.']
        from experiments.full_replay_campaign import benchmark_metrics
        b=benchmark_metrics(read(output/'full-evaluation/runs/base/results.json'))
        c=benchmark_metrics(read(output/'full-evaluation/runs/coding-replay-768-128/results.json'))
        for key,label in [('svamp','Matematika'),('mbpp','Kódování'),('bfcl','Nástroje'),('ifeval','Instrukce')]:
            lines += ['',label,f'Accuracy: {100*b[key]["accuracy"]:.1f}% → {100*c[key]["accuracy"]:.1f}%',
                      f'Tokeny / odpověď: {b[key]["mean_output_tokens"]:.0f} → {c[key]["mean_output_tokens"]:.0f}']
    failure=output/'coding-campaign/failure.json'
    if failure.exists():lines+=['Důvod: '+read(failure)['error']]
    lines+=['GPU uvolněna: '+('ano' if job.get('released_at') else 'ne'),'Timestamp: '+datetime.now(ZoneInfo('Europe/Prague')).strftime('%Y-%m-%d %H:%M:%S %Z (Praha)')]
    write(reg/'campaign.json',campaign);notify(reg,name+':terminal','\n'.join(lines))
    subprocess.run([sys.executable,str(ROOT/'experiment-ui/publish.py')],check=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run']);parser.add_argument('--directory',type=Path);args=parser.parse_args()
    if args.action=='prepare':print(prepare())
    else:run(args.directory or Path(read(REG/'current.json')['directory']))
