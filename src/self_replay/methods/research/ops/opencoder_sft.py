"""Registered single4090 OpenCoder pilot, canonical budget and durable monitoring."""
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
from experiments.opencoder_sft.runtime import SCOPE,PREFIX,ENTRY,GPU,read,rows,sha,write
from experiments.replay_scaled.notify import deliver_once,validation_message
from ops import vastai
from ops.isolated_cloud import seal,verify
REG=ROOT/'lab/experiments/opencoder-sft'


def prepare():
    pointer=read(REG/'current.json');source=Path(pointer['directory']);spec=read(source/'plan.json')
    if spec.get('state')!='data_prepared_runtime_pending':raise RuntimeError('Existing registered launch; reconcile')
    jobs=read(cloud.STATE/'ledger.json')['jobs']
    if any(n.startswith(PREFIX) and (j.get('pod_id') or j.get('allocation_attempts') or j.get('state')!='allocation_failed') for n,j in jobs.items()) or vastai.list_pods():raise RuntimeError('Existing allocation needs reconciliation')
    credit=float(vastai.api('v0/users/current/')['credit'])
    if credit<17:raise RuntimeError('Need15USD plus2USD reserve')
    for f,h in spec['files_sha256'].items():
        if sha(source/'prepared'/f)!=h:raise RuntimeError('Prepared data changed')
    parent=ROOT/spec['parent']['path'];manifest=read(parent/'complete.json')
    if sha(parent/'complete.json')!=spec['parent']['complete_sha256']:raise RuntimeError('Parent manifest changed')
    for f,h in manifest['files'].items():
        if sha(parent/f)!=h:raise RuntimeError('Parent file corrupt')
    excluded={j.get('provider_quote',{}).get('offer',{}).get('machine_id') for j in jobs.values() if j.get('startup_failed_before_launch')}
    quote=vastai.quote(GPU,1,80,min_duration=19*3600,exclude_machine_ids=excluded-{None},
        allowed_countries=('US','CA','GB','DE','FR','NL','PL','FI','SE','CZ','ES','IT','AT','RO','LT','BG'))
    if not quote or quote['hourly_ceiling']>.6:raise RuntimeError('No suitable single4090 offer <=0.60USD/h')
    quote=vastai.quote(GPU,1,80,quote['offer_id'],19*3600)
    if not quote or quote['hourly_ceiling']>.6:raise RuntimeError('Offer disappeared before registration')
    stamp=time.strftime('%Y%m%dt%H%M%Sz',time.gmtime()).lower();name=PREFIX+stamp;reg=REG/stamp;reg.mkdir()
    release=ROOT/'lab/isolated'/name
    ignore=shutil.ignore_patterns('__pycache__','.git','.env*','.venv*','runs','*.log')
    shutil.copytree(ROOT/'experiments',release/'experiments',ignore=ignore)
    frozen=ROOT/'lab/isolated/full-evaluation-20261026/teacher-student-v2'
    shutil.copytree(frozen/'ThinkingCap-Test',release/'ThinkingCap-Test',ignore=ignore)
    stage=release/'lab/experiments/data/opencoder-sft';stage.mkdir(parents=True)
    shutil.copytree(parent/'adapter',stage/'adapter')
    shutil.copy2(source/'prepared/train-schedule.jsonl',stage/'train-schedule.jsonl')
    # Preserve every source roster/hash in control registration; only runtime inputs go to worker.
    for f in ['plan.json','data-audit-summary.json','preflight-local.json']:shutil.copy2(source/f,reg/('preparation-'+f))
    original=rows(ROOT/'lab/experiments/replay-scaled/20261101-checkpoints-v5/development.jsonl')
    dev=[dict(r,cohort='original-development') for r in original]
    for r in rows(source/'prepared/development-verified.jsonl'):
        dev.append(dict(id=r['id'],task='coding',cohort='opencoder-development',family=r['family'],
            prompt=[{'role':'user','content':r['problem']+'\n\nImplement the Python function '+r['entry_point']+'.'}],
            answer=json.dumps(dict(task='coding',tests=r['tests']))))
    monitor=[]
    for task in ['math','coding','tools','instructions']:
        pool=sorted([r for r in dev if r['cohort']=='original-development' and r['task']==task],key=lambda r:sha_id(r['id']))
        monitor+=pool[:32]
    monitor+=sorted([r for r in dev if r['cohort']=='opencoder-development'],key=lambda r:sha_id(r['id']))[:64]
    assert len(dev)==512 and len(monitor)==192
    trainids={r['id'] for r in rows(stage/'train-schedule.jsonl')}
    assert not trainids&{r['id'] for r in dev}
    for namefile,records in [('development.jsonl',dev),('monitor.jsonl',monitor)]:
        (stage/namefile).write_text(''.join(json.dumps(r)+'\n' for r in records))
    cfg=dict(scope=SCOPE,parent_checkpoint=768,parent_complete_sha256=sha(parent/'complete.json'),
        coding_questions=2000,optimizer_steps=84,effective_batch_size=32,micro_batch_size=2,gradient_accumulation=16,
        lr=1e-5,weight_decay=0,warmup_steps=5,scheduler='cosine',seed=17,max_length=4096,
        runtime='existing pinned torch/transformers/peft; assistant-only CE, empty-think masked',
        preparation_source=str(source),preparation_sha256=sha(source/'plan.json'),
        changes_from_preparation=['single4090 instead of4A100; effective batch32 unchanged','existing pinned runtime instead of new MS-SWIFT dependencies',
            'development192x2 per16updates; full512x2 at0/84; new answer generation ONLY for validation'],
        checkpoint_steps=[16,32,48,64,80,84],quality_stopping=False,
        files_sha256={str(p.relative_to(stage)):sha(p) for p in stage.rglob('*') if p.is_file()})
    write(stage/'plan.json',cfg);write(reg/'plan.json',cfg)
    evaluation=release/'lab/experiments/data/replay-evaluation';evaluation.mkdir()
    preload=ROOT/'lab/experiments/replay-scaled-evaluation/20261102t102548z'
    shutil.copy2(preload/'model-preload-source-verification.json',evaluation/'model-preload.json')
    shutil.copy2(preload/'uv-wheel-source.json',evaluation/'uv-wheel-source.json')
    shutil.copy2(ROOT/'lab/experiments/data/study/splits.json',evaluation/'splits.json')
    shutil.copy2(ROOT/'lab/gpu-requirements.lock',release/'lab/gpu-requirements.lock')
    digest=seal(release)
    from experiments.baseline_cache import check_launch
    check_launch(release,ENTRY,ROOT/'lab/experiments/baseline-cache',GPU)
    launch=dict(job_name=name,release=str(release),payload_sha256=digest,authorized=True,scope=SCOPE,provider='vast',hardware_scope=SCOPE,
        entry=ENTRY,gpu=GPU,gpu_count=1,offer_id=quote['offer_id'],disk_gb=80,max_cost_usd=15,max_runtime_hours=18,
        max_gpu_rate=math.ceil(quote['hourly_ceiling']*1000)/1000,training_authorized=True,evaluation_only=False,
        checkpoint_group=768,coding_questions=2000,optimizer_steps=84,direct_ssh=True,
        source_user_instruction='2026-11-03: user says tak to prosim rozjed after recommendation1RTX4090;2000-task supplied OpenCoder SFT from768, live progress to decide continuation;15USD conservative named cap within prior50USD proposal.')
    write(reg/'launch.json',launch);write(reg/'quote.json',quote)
    write(reg/'campaign.json',dict(state='prepared',job_name=name,created_at=time.time(),credit_at_prepare=credit,cap_usd=15,telegram_authorized=True))
    write(REG/'current.json',dict(directory=str(reg)))
    return reg


def sha_id(value):
    import hashlib
    return hashlib.sha256(('17:'+value).encode()).hexdigest()


def notify(reg,key,message):
    try:deliver_once(reg/'notifications.sqlite3',key,message)
    except Exception as e:write(reg/'notification-error.json',dict(key=key,error=type(e).__name__,time=time.time()))


def send_validations(reg,name,campaign,job):
    live=campaign.get('live',{});points=live.get('training/validation-curve.json',{}).get('checkpoints',[])
    for point in points:
        if not point['iteration']:continue
        elapsed=max(0,point['completed_at']-job['created_at'])
        msg=validation_message(dict(current=point['iteration'],total=84,elapsed=elapsed,gpu_count=1),point,points[0])
        msg=msg.replace('Self-replay scaled','OpenCoder SFT · parent768').replace('1× A100-SXM4-80GB','1× RTX 4090 (Vast)').replace('Iterations (otázky)','Updates (optimizer)')
        v=point['metrics']['opencoder'];b=points[0]['metrics']['opencoder']
        msg+='\n\nOpenCoder · oddělené validační úlohy\nAccuracy: '+f"{100*b['accuracy']:.1f}% → {100*v['accuracy']:.1f}% ({100*v['accuracy_delta']:+.1f} p. b.)\nTokeny / odpověď: {b['mean_tokens']:.0f} → {v['mean_tokens']:.0f}"
        msg+='\nCheckpoint uložen; pokles accuracy sám běh nezastavuje.'
        notify(reg,name+':validation:'+str(point['iteration']),msg)


def run(reg):
    launch=read(reg/'launch.json');name=launch['job_name'];campaign=read(reg/'campaign.json')
    if verify(launch['release'])!=launch['payload_sha256']:raise RuntimeError('Sealed payload changed')
    jobs=read(cloud.STATE/'ledger.json')['jobs']
    if name in jobs:command=[sys.executable,str(ROOT/'ops/isolated_cloud.py'),'watch','--release',launch['release'],'--name',name]
    else:
        if vastai.list_pods():raise RuntimeError('Unexpected rental; no duplicate')
        if float(vastai.api('v0/users/current/')['credit'])<17:raise RuntimeError('Credit below15USD+reserve')
        write(cloud.STATE/'authorizations'/(name+'.json'),launch)
        command=[sys.executable,str(ROOT/'ops/isolated_cloud.py'),'run','--release',launch['release'],'--name',name,'--entry',ENTRY,
            '--gpu',GPU,'--gpu-count','1','--hardware-scope',SCOPE,'--hours','1','--max-cost-usd','15','--max-runtime-hours','18',
            '--provider','vast','--vast-offer-id',str(launch['offer_id']),'--vast-disk-gb','80','--data-upload-timeout','1800']
    campaign.update(state='starting');write(reg/'campaign.json',campaign)
    keys=['opencoder-campaign/progress.json','opencoder-campaign/failure.json','opencoder-campaign/completion.json',
          'training/progress.json','training/latest-checkpoint.json','training/validation-curve.json','training/segment-completion.json',
          'gpu-verification.json','full-evaluation/completion.json']
    with (reg/'controller.log').open('a') as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        while process.poll() is None:
            job=read(cloud.STATE/'ledger.json')['jobs'].get(name,{})
            if job.get('state')=='running' and job.get('pod_id') and not job.get('released_at'):
                try:
                    pod=cloud.api('pods/'+job['pod_id'])
                    script="import json;from pathlib import Path;p=Path('/workspace/project/output');names="+repr(keys)+";print(json.dumps({n:json.loads((p/n).read_text()) for n in names if (p/n).exists()}))"
                    live=json.loads(cloud.remote(pod,'python3 -c '+shlex.quote(script),timeout=20))
                    campaign.update(state='running',observed_at=time.time(),live=live);campaign.pop('monitor_error',None)
                    send_validations(reg,name,campaign,job)
                except Exception as e:campaign['monitor_error']=str(e)[:300]
                write(reg/'campaign.json',campaign)
            time.sleep(20)
    job=read(cloud.STATE/'ledger.json')['jobs'].get(name,{})
    output=cloud.STATE/name/'output';done=read(output/'opencoder-campaign/completion.json') if (output/'opencoder-campaign/completion.json').exists() else {}
    complete=bool(job.get('backup_verified') and job.get('released_at') and done.get('complete'))
    for key in keys:
        if (output/key).exists():campaign.setdefault('live',{})[key]=read(output/key)
    if job.get('created_at'):send_validations(reg,name,campaign,job)
    campaign.update(state='complete' if complete else 'stopped',backup_verified=bool(job.get('backup_verified')),released=bool(job.get('released_at')),output=str(output),finished_at=time.time())
    lines=['<b>Project: ThinkingCap / OpenCoder SFT · parent768</b>',
        'Summary: '+('Pilot a plná evaluace dokončeny.' if complete else 'Běh ukončen před dokončením; zachované výstupy a checkpointy ověřit.'),
        'GPU: 1× RTX 4090 (Vast)']
    if complete:
        result=read(output/'full-evaluation/comparison.json');campaign['official_results']=result
        lines += [f"Accuracy proti původní baseline: {100*result['average']['accuracy_delta']:+.2f} p. b.",f"Úspora tokenů: {100*result['average']['token_reduction']:.1f} %",'Plná evaluace: 2759 ×8; baseline z cache.']
        for k,label in [('svamp','Matematika'),('mbpp','Kódování'),('ifeval','Instrukce')]:
            v=result['results'][k];lines+=['',label,f"Accuracy: {100*v['base']['accuracy']:.1f}% → {100*v['candidate']['accuracy']:.1f}%"]
    failure=output/'opencoder-campaign/failure.json'
    if failure.exists():lines+=['Důvod: '+read(failure)['error']]
    lines+=['GPU uvolněna: '+('ano' if job.get('released_at') else 'ne'),'Timestamp: '+datetime.now(ZoneInfo('Europe/Prague')).strftime('%Y-%m-%d %H:%M:%S %Z (Praha)')]
    write(reg/'campaign.json',campaign);notify(reg,name+':terminal','\n'.join(lines))
    subprocess.run([sys.executable,str(ROOT/'experiment-ui/publish.py')],check=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);p.add_argument('--directory',type=Path);a=p.parse_args()
    if a.action=='prepare':print(prepare())
    else:run(a.directory or Path(read(REG/'current.json')['directory']))
