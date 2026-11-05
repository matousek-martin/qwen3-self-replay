"""Prepare/launch/monitor the three named, funded inference-only survey arms."""
import argparse,datetime,json,math,shlex,shutil,subprocess,sys,time
from pathlib import Path
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments import cloud
from experiments.coding_frontier.budget import DATASETS,ENTRY,GPU,PREFIX,SCOPE
from experiments.opencoder_sft.runtime import read,write,sha
from experiments.replay_scaled.notify import deliver_once
from ops import vastai
from ops.isolated_cloud import seal,verify
REG=ROOT/'lab/experiments/coding-frontier'

def prepare():
    if (REG/'current.json').exists():raise RuntimeError('Campaign already registered; reconcile before another launch')
    if vastai.list_pods():raise RuntimeError('Existing Vast workers require reconciliation')
    credit=float(vastai.api('v0/users/current/')['credit'])
    if credit<64:raise RuntimeError('Need60USD campaign funding and4USD account reserve')
    data=REG/'data-v2';inventory=read(data/'complete.json')
    if not inventory['complete'] or set(inventory['datasets'])!=set(DATASETS):raise RuntimeError('Inventory incomplete')
    jobs=read(cloud.STATE/'ledger.json')['jobs']
    excluded={j.get('provider_quote',{}).get('offer',{}).get('machine_id') for j in jobs.values() if j.get('startup_failed_before_launch')}-{None}
    parent=ROOT/'lab/experiments/cloud/replay-scaled-training-20261101t122440z/output/training/checkpoints/group-000768'
    manifest=read(parent/'complete.json')
    for f,h in manifest['files'].items():
        if sha(parent/f)!=h:raise RuntimeError('Parent768 checkpoint corrupt')
    # Reuse tokenizer/config staging from the verified earlier768 experiment.
    old=read(ROOT/'lab/experiments/opencoder-mode-probe/20261103t231900z/launch.json')
    if verify(old['release'])!=old['payload_sha256']:raise RuntimeError('Preserved parent staging corrupt')
    adapter=Path(old['release'])/'lab/experiments/data/opencoder-mode-probe/adapters/parent768'
    if sha(adapter/'adapter_model.safetensors')!=sha(parent/'adapter/adapter_model.safetensors'):raise RuntimeError('Parent768 adapter differs')
    stamp=time.strftime('%Y%m%dt%H%M%Sz',time.gmtime()).lower();reg=REG/stamp;reg.mkdir();campaign_path=reg/'campaign.json'
    campaign=dict(authorized=True,state='prepared',cap_usd=60,credit_at_prepare=credit,created_at=time.time(),arms={},source_user_instruction='2026-11-04: User requests parallel checkpoint768 inference surveys of APPS, TACO and KodCode Light, one GPU each, all difficulties ascending, retain per-answer correctness/token data to select future training.')
    for label in DATASETS:
        quote=vastai.quote(GPU,1,80,min_duration=48*3600,exclude_machine_ids=excluded,
            allowed_countries=('US','CA','GB','DE','FR','NL','PL','FI','SE','CZ','ES','IT','AT','RO','LT','BG'))
        if not quote or quote['hourly_ceiling']>.6:raise RuntimeError('Suitable three-worker4090 capacity unavailable')
        excluded.add(quote['offer']['machine_id']);name=PREFIX+label+'-'+stamp;arm=reg/label;arm.mkdir();release=ROOT/'lab/isolated'/name
        shutil.copytree(ROOT/'experiments',release/'experiments',ignore=shutil.ignore_patterns('__pycache__','.git','.env*','.venv*','*.log'))
        (release/'ThinkingCap-Test').mkdir();stage=release/'lab/experiments/data/coding-frontier';stage.mkdir(parents=True)
        shutil.copytree(adapter,stage/'adapter');shutil.copy2(data/(label+'.jsonl.gz'),stage/'questions.jsonl.gz')
        cfg=dict(scope=SCOPE,dataset=label,parent_group=768,parent_manifest_sha256=sha(parent/'complete.json'),training_authorized=False,
            inference_only=True,official_evaluation=False,questions=inventory['datasets'][label]['eligible_before_reference_check'],
            samples=8,context_tokens=40960,temperature=.6,top_p=.95,top_k=20,enable_thinking=True,
            source_inventory=inventory['datasets'][label],source_files=read(REG/'sources/source-files.json'),ordering=inventory['ordering'],reference_policy='Verify supplied reference in sandbox before model sampling; unverified tasks are separately accounted, not model failures.',
            files_sha256={str(p.relative_to(stage)):sha(p) for p in stage.rglob('*') if p.is_file()})
        write(stage/'plan.json',cfg);write(arm/'plan.json',cfg)
        preload=ROOT/'lab/experiments/replay-scaled-evaluation/20261102t102548z';target=release/'lab/experiments/data/replay-evaluation';target.mkdir()
        shutil.copy2(preload/'model-preload-source-verification.json',target/'model-preload.json');shutil.copy2(preload/'uv-wheel-source.json',target/'uv-wheel-source.json')
        lock=release/'lab/gpu-requirements.lock';lock.write_text((ROOT/'lab/gpu-requirements.lock').read_text()+'\npytest==8.3.5\niniconfig==2.3.1\npluggy==1.7.0\n')
        digest=seal(release)
        launch=dict(job_name=name,release=str(release),payload_sha256=digest,authorized=True,scope=SCOPE,hardware_scope=SCOPE,entry=ENTRY,
            provider='vast',gpu=GPU,gpu_count=1,dataset=label,parent_group=768,samples=8,offer_id=quote['offer_id'],disk_gb=80,
            max_cost_usd=20,max_runtime_hours=48,max_gpu_rate=math.ceil(quote['hourly_ceiling']*1000)/1000,evaluation_only=True,
            training_authorized=False,direct_ssh=True,campaign_path=str(campaign_path),combined_cap_usd=60,
            source_user_instruction=campaign['source_user_instruction'])
        write(arm/'launch.json',launch);write(arm/'quote.json',quote);write(arm/'status.json',dict(state='prepared',job_name=name))
        campaign['arms'][label]=dict(job_name=name,directory=str(arm),max_cost_usd=20,questions=cfg['questions'])
    write(campaign_path,campaign);write(REG/'current.json',dict(directory=str(reg)))
    for label in DATASETS:
        v=read(reg/label/'launch.json');write(cloud.STATE/'authorizations'/(v['job_name']+'.json'),v)
    return reg

def message(label,p,terminal=False,released=False):
    levels=p.get('levels',{});lines=['<b>Project: ThinkingCap / Coding capability · '+label.upper()+'</b>',
        'Summary: '+('Test dokončen.' if p.get('phase')=='complete' else 'Výsledky uloženy; financovaný úsek skončil.' if terminal else 'Nové výsledky testu schopností.'),
        'Checkpoint: 768 · žádný trénink','GPU: 1× RTX 4090 (Vast)',
        f"Otázky: {p.get('visited',0)} / {p.get('total_questions','?')}",f"Odpovědi: {p.get('answers',0)} · 8 na ověřitelnou úlohu",
        f"Neověřitelné testy / jiné technické vyřazení: {p.get('skipped_questions',0)}"]
    for level,v in levels.items():lines+=['',level,f"Accuracy odpovědí: {100*v['accuracy']:.1f}%",f"Vyřešeno alespoň jednou: {v['solved_questions']} / {v['questions']}",f"Tokeny / odpověď: {v['mean_tokens']:.0f}",f"Useknutí kontextem: {v['failures'].get('context_limit',0)}"]
    secs=int(p.get('elapsed',0));lines+=['',f'Time elapsed: {secs//3600:02d}:{secs%3600//60:02d}:{secs%60:02d}',
        'Timestamp: '+datetime.datetime.now(ZoneInfo('Europe/Prague')).strftime('%Y-%m-%d %H:%M:%S %Z (Praha)')]
    if terminal:lines+=['GPU uvolněna: '+('ano' if released else 'ne')]
    return '\n'.join(lines)

def run(reg,label):
    arm=reg/label;launch=read(arm/'launch.json');name=launch['job_name']
    def save_status(value):
        # A finishing predecessor must never overwrite its replacement's live UI.
        target=arm/'status.json' if read(arm/'launch.json')['job_name']==name else arm/('status-'+name+'.json')
        write(target,value)
    if verify(launch['release'])!=launch['payload_sha256']:raise RuntimeError('Sealed payload differs')
    jobs=read(cloud.STATE/'ledger.json')['jobs']
    resume_startup=name in jobs and launch.get('resume_startup') is True and jobs[name].get('state')=='starting' and not jobs[name].get('worker_launch_attempted_at')
    if name in jobs and not resume_startup:cmd=[sys.executable,str(ROOT/'ops/isolated_cloud.py'),'watch','--release',launch['release'],'--name',name]
    else:
        # All three reservations are already named. No extra provider workers may coexist.
        expected={a['job_name'] for a in read(reg/'campaign.json')['arms'].values()}
        known_pods={j.get('pod_id') for n,j in jobs.items() if n in expected and not j.get('released_at')}
        if any(p['id'] not in known_pods for p in vastai.list_pods()):raise RuntimeError('Unexpected live Vast rental')
        if not resume_startup and float(vastai.api('v0/users/current/')['credit'])<60:raise RuntimeError('Campaign funding changed before launch')
        cmd=[sys.executable,str(ROOT/'ops/isolated_cloud.py'),'run','--release',launch['release'],'--name',name,'--entry',ENTRY,
            '--gpu',GPU,'--gpu-count','1','--hardware-scope',SCOPE,'--hours','1','--max-cost-usd',str(launch['max_cost_usd']),'--max-runtime-hours','48',
            '--provider','vast','--vast-disk-gb','80','--data-upload-timeout','1800']
        if launch.get('offer_id') is not None:cmd+=['--vast-offer-id',str(launch['offer_id'])]
        if resume_startup:cmd.append('--resume-startup')
    status=dict(state='starting',job_name=name,started_at=time.time());save_status(status)
    with (arm/'controller.log').open('a') as log:
        proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
        while proc.poll() is None:
            job=read(cloud.STATE/'ledger.json')['jobs'].get(name,{})
            if job.get('state')=='running' and job.get('pod_id') and not job.get('released_at'):
                try:
                    pod=cloud.api('pods/'+job['pod_id'])
                    script="import json;from pathlib import Path;p=Path('/workspace/project/output/coding-frontier');print(json.dumps({n:json.loads((p/n).read_text()) for n in ['progress.json','preflight.json','failure.json'] if (p/n).exists()}))"
                    live=json.loads(cloud.remote(pod,'python3 -c '+shlex.quote(script),timeout=20));status.update(state='running',observed_at=time.time(),live=live);status.pop('monitor_error',None)
                    p=live.get('progress.json',{});questions=p.get('graded_questions',0)
                    if (launch.get('previous_attempts') and live.get('preflight.json',{}).get('passed')
                            and p.get('phase') in ('generation','grading','survey')):
                        recovered=message(label,p).replace('Nové výsledky testu schopností.',
                            'Obnoveno: aktuální test běží; předchozí chyba patřila staršímu pokusu.')
                        deliver_once(reg/'notifications.sqlite3',name+':recovered',recovered)
                    if questions>=32:
                        bucket=32 if questions<128 else questions//128*128
                        deliver_once(reg/'notifications.sqlite3',name+':questions:'+str(bucket),message(label,p))
                except Exception as e:status['monitor_error']=str(e)[:350]
                save_status(status)
            time.sleep(30)
    job=read(cloud.STATE/'ledger.json')['jobs'].get(name,{});out=cloud.STATE/name/'output/coding-frontier'
    done=read(out/'completion.json') if (out/'completion.json').exists() else {};p=read(out/'progress.json') if (out/'progress.json').exists() else {}
    verified=bool(job.get('released_at') and job.get('backup_verified') and (out/'results.json').exists() and sha(out/'results.json')==done.get('results_sha256'))
    status.update(state=('complete' if done.get('complete') else 'budget_checkpoint') if verified else 'failed',finished_at=time.time(),released=bool(job.get('released_at')),
        backup_verified=bool(job.get('backup_verified')),output=str(out),live={'progress.json':p})
    save_status(status)
    text=message(label,p,terminal=True,released=bool(job.get('released_at')))
    if not verified:text=text.replace('Výsledky uloženy; financovaný úsek skončil.','Technická chyba; test nedokončen, kontroluji archiv.')
    if read(arm/'launch.json')['job_name']==name:deliver_once(reg/'notifications.sqlite3',name+':terminal',text)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);p.add_argument('--dataset',choices=DATASETS);p.add_argument('--directory',type=Path);a=p.parse_args()
    if a.action=='prepare':print(prepare())
    else:run(a.directory or Path(read(REG/'current.json')['directory']),a.dataset)
