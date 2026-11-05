"""Bounded pilot preparation and monitoring through the canonical cloud ledger."""
import argparse,json,math,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments import cloud
from experiments.replay_scaled.core import GPU,SCOPE,ENTRY,write
from experiments.replay_scaled.notify import deliver_once,validation_message
from experiments.replay_scaled.generation import validate_settings
from ops import vastai
from ops.isolated_cloud import seal,verify
POINTER=ROOT/'lab/experiments/replay-scaled/current.json'
REG=(ROOT/json.loads(POINTER.read_text())['path']).parent if POINTER.exists() else ROOT/'lab/experiments/replay-scaled/20261101'

def read(p):return json.loads(Path(p).read_text())
def previous_pilot_spend(jobs,names):
    spent=0.
    for name in names:
        job=jobs[name]
        # Canonically reconciled failed creates have no worker artifacts to back up.
        no_worker=(job.get('safely_no_pod') and not job.get('pod_id') and job.get('charged_estimate_usd')==0
                   and job.get('state') in ('create_absent_reconciled','capacity_rejected_absent','allocation_failed'))
        prelaunch_failure=(job.get('state')=='failed' and job.get('startup_failed_before_launch') is True
                           and not job.get('worker_launch_attempted_at'))
        if not job.get('released_at') or not (job.get('backup_verified') or no_worker or prelaunch_failure):
            raise RuntimeError('Previous pilot not archived/released or conclusively reconciled as never allocated')
        spent+=job['charged_estimate_usd']
    return spent

def prepare(mode='pilot', cap=10, repair=False):
    if mode not in ('pilot','training') or not 1<cap<=150:raise ValueError('Invalid bounded launch')
    if mode=='pilot':
        previous=read(REG/'campaign.json').get('pilot_previous_jobs',[])
        spent=previous_pilot_spend(read(cloud.STATE/'ledger.json')['jobs'],previous)
        cap=min(cap,round(10-spent,4))
        if cap<=1:raise RuntimeError('Combined pilot budget exhausted')
    proof=None
    if repair:
        if mode!='training':raise ValueError('Repair is a main-stage operation')
        c=read(REG/'campaign.json');jobs=read(cloud.STATE/'ledger.json')['jobs'];parent=c['job_name']
        if c.get('state')!='stopped' or not c.get('full_training_authorized'):raise RuntimeError('No stopped authorized main stage')
        progress=read(cloud.STATE/parent/'output/training/progress.json')
        if progress.get('optimizer_updates')!=0 or progress.get('current')!=0:
            raise RuntimeError('Trained state exists: use complete resume, never restart from base')
        if list((cloud.STATE/parent/'output/training/checkpoints').glob('*/complete.json')):
            raise RuntimeError('Checkpoint exists: preserve it through explicit resume')
        previous=c.get('main_previous_jobs',[])+[parent]
        if len(set(previous))!=len(previous):raise RuntimeError('Duplicate predecessor')
        spent=previous_pilot_spend(jobs,previous)
        cap=min(cap,round(c['main_authorization']['cap_usd']-spent,4))
        if cap<=1:raise RuntimeError('Funded main-stage budget exhausted')
        old_launch=read(REG/'launch.json')
        from experiments.replay_scaled.core import sha
        old=read(Path(old_launch['release'])/'payload-manifest.json')['files']
        # Repair changes transport and runtime verification, never the model objective/data.
        for file in ['experiments/replay_scaled/'+x for x in ['core.py','metrics.py','pilot.py','generation.py']]+['experiments/rl_training/train.py','experiments/rl_training/objectives.py','experiments/shorterbetter/multidomain/rewards.py']:
            if sha(ROOT/file)!=old[file]:raise RuntimeError('Repair changes protected objective: '+file)
        for file in ['plan.json','train.jsonl','development.jsonl']:
            if sha(REG/file)!=old['lab/experiments/data/replay-scaled/'+file]:raise RuntimeError('Repair changes frozen plan/data')
        archive=REG/'attempts'/parent;archive.mkdir(parents=True,exist_ok=False)
        for file in ['campaign.json','launch.json','quote.json','live.json']:
            if (REG/file).exists():shutil.copy2(REG/file,archive/file)
        c.update(main_previous_jobs=previous,repair_requires_same_worker_preflight=True)
        write(REG/'campaign.json',c)
    elif mode=='training':
        c=read(REG/'campaign.json')
        if c.get('state')!='pilot_passed':raise RuntimeError('Verified GPU pilot required before main preparation')
        proof=read(ROOT/c['verification'])
        if not proof.get('passed') or proof.get('world_size')!=2:raise RuntimeError('GPU pilot did not verify 2 ranks')
        generation=validate_settings(read(REG/'plan.json').get('generation'))
        if proof.get('generation_protocol')!=generation['protocol'] or proof.get('generation_settings')!=generation:
            raise RuntimeError('GPU proof does not cover the registered generation implementation')
        pilot=read(REG/'launch.json')
        old=read(Path(pilot['release'])/'payload-manifest.json')['files']
        from experiments.replay_scaled.core import sha
        for file in ['experiments/replay_scaled/'+x for x in ['worker.py','core.py','metrics.py','check.py','pilot.py','generation.py','communication.py','run.sh']]+['experiments/rl_training/train.py','experiments/rl_training/objectives.py','experiments/shorterbetter/multidomain/rewards.py']:
            if sha(ROOT/file)!=old[file]:raise RuntimeError('Worker changed after GPU test: '+file)
        write(REG/'pilot-launch.json',pilot)
    stamp=time.strftime('%Y%m%dt%H%M%Sz',time.gmtime()).lower();name='replay-scaled-'+mode+'-'+stamp
    release=ROOT/'lab/isolated'/name
    ignore=shutil.ignore_patterns('.git','__pycache__','.venv*','runs','.env*','*.log')
    shutil.copytree(ROOT/'experiments',release/'experiments',ignore=ignore)
    shutil.copytree(ROOT/'ThinkingCap-Test',release/'ThinkingCap-Test',ignore=ignore)
    stage=release/'lab/experiments/data/replay-scaled';stage.mkdir(parents=True)
    for file in ['plan.json','train.jsonl','development.jsonl']:shutil.copy2(REG/file,stage/file)
    write(stage/'launch-mode.json',{'mode':mode,'verify_before_training':repair})
    if proof:write(stage/'gpu-verification.json',proof)
    shutil.copy2(ROOT/'lab/gpu-requirements.lock',release/'lab/gpu-requirements.lock')
    digest=seal(release)
    prior=read(REG/'campaign.json').get('pilot_previous_jobs',[])
    jobs=read(cloud.STATE/'ledger.json')['jobs']
    excluded=[jobs[name]['offer_id'] for name in prior if jobs[name].get('state')=='create_absent_reconciled'
              and jobs[name].get('provider')=='vast' and jobs[name].get('offer_id')
              and time.time()-jobs[name]['created_at']<600]
    excluded_machines=[jobs[name].get('provider_quote',{}).get('offer',{}).get('machine_id') for name in prior
                       if jobs[name].get('startup_failed_before_launch') and jobs[name].get('released_at')
                       and time.time()-jobs[name]['created_at']<3600]
    if repair:
        excluded_machines += [jobs[n].get('provider_quote',{}).get('offer',{}).get('machine_id')
                              for n in read(REG/'campaign.json')['main_previous_jobs']]
        excluded_machines += [jobs[n].get('provider_quote',{}).get('offer',{}).get('machine_id')
                              for n in prior if jobs[n].get('startup_failed_before_launch')]
    quote=vastai.quote(GPU,2,60,min_duration=3*3600,exclude_offer_ids=excluded,
                      exclude_machine_ids=[m for m in excluded_machines if m is not None])
    if not quote:raise RuntimeError('No eligible 2A100 offer')
    rate=math.ceil(quote['hourly_ceiling']/2*1000)/1000
    if rate>2.2:raise RuntimeError('Quote exceeds registered ceiling')
    launch=dict(job_name=name,release=str(release),payload_sha256=digest,provider='vast',gpu=GPU,gpu_count=2,
                hardware_scope=SCOPE,entry=ENTRY,max_cost_usd=cap,max_runtime_hours=2 if mode=='pilot' else 12,max_gpu_rate=rate,
                offer_id=quote['offer_id'],disk_gb=60,direct_ssh=True,phase=mode)
    write(REG/'launch.json',launch);write(REG/'quote.json',quote)
    return launch

def run(observe_only=False):
    launch=read(REG/'launch.json');name=launch['job_name'];provider=launch.get('provider','vast')
    is_pilot=launch['phase']=='pilot'
    campaign=read(REG/'campaign.json')
    if not is_pilot and not campaign.get('full_training_authorized'):
        raise RuntimeError('Main campaign budget must be explicitly registered before running')
    if verify(launch['release'])!=launch['payload_sha256']:raise RuntimeError('Payload changed')
    jobs=read(cloud.STATE/'ledger.json')['jobs']
    if observe_only:
        if name not in jobs or jobs[name].get('released_at') or not jobs[name].get('identity_confirmed'):
            raise RuntimeError('Watch requires an existing identified active job')
    elif name in jobs:raise RuntimeError('Existing job: inspect canonical state, never duplicate')
    instruction=('Prepare larger self-replay test with checkpoints/UI/Telegram; run on2 or4A100 (2026-11-01).'
                 if is_pilot else campaign['main_authorization']['source_user_instruction'])
    if not is_pilot:
        spent=previous_pilot_spend(jobs,campaign.get('main_previous_jobs',[]))
        if launch['max_cost_usd']>round(campaign['main_authorization']['cap_usd']-spent,4):
            raise RuntimeError('Launch exceeds the remaining registered funded training stage')
    auth=dict(launch,authorized=True,scope=('replay_runpod_resume_v1' if provider=='runpod' else 'replay_vast_bounded_v1'),source_user_instruction=instruction)
    if not observe_only:write(cloud.STATE/'authorizations'/(name+'.json'),auth)
    campaign.update(state='pilot_running' if is_pilot else 'training_running',job_name=name,
                    active_cap_usd=launch['max_cost_usd'],active_hours=launch['max_runtime_hours'],
                    released=False,backup_verified=False);write(REG/'campaign.json',campaign)
    if not observe_only:
        write(REG/'live.json',{'progress':{'phase':'initializing','current':launch.get('resume_group',0),'total':4 if is_pilot else campaign['total'],
                                         'pilot':is_pilot,'gpu_count':2},'observed_at':time.time()})
    command=[sys.executable,str(ROOT/'ops/isolated_cloud.py'),'run','--release',launch['release'],'--name',name,'--entry',ENTRY,
             '--gpu',GPU,'--gpu-count','2','--hardware-scope',SCOPE,'--hours','1','--max-cost-usd',str(launch['max_cost_usd']),
             '--max-runtime-hours',str(launch['max_runtime_hours']),'--provider',provider,'--data-upload-timeout','900']
    if provider=='vast':command+=['--vast-offer-id',str(launch['offer_id']),'--vast-disk-gb','60']
    if observe_only:
        command=[sys.executable,str(ROOT/'ops/isolated_cloud.py'),'watch','--release',launch['release'],'--name',name]
    log=(REG/'controller.log').open('a');process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
    folder='pilot-reference' if is_pilot else 'training'
    # Independent small status transfer; checkpoint synchronization stays in canonical controller.
    remote_script='import json,pathlib; p=pathlib.Path("/workspace/project/output/'+folder+'"); out={};\nfor key,file in [("progress","progress.json"),("training","training-series.json"),("validation","validation-curve.json"),("checkpoint","latest-checkpoint.json")]:\n try: out[key]=json.loads((p/file).read_text())\n except (OSError,ValueError): pass\nprint(json.dumps(out))'
    if is_pilot:
        remote_script=remote_script.replace('print(json.dumps(out))',
            'b=p.parent/"generation-benchmark/progress.json"\nif b.exists():\n out["progress"]=json.loads(b.read_text())\n out["progress"].update(pilot=True,phase="generation_benchmark")\nprint(json.dumps(out))')
    else:
        remote_script=remote_script.replace('print(json.dumps(out))',
            'b=p.parent/"preflight-reference/progress.json"\nif not out.get("progress") and b.exists():\n out["progress"]=json.loads(b.read_text())\n out["progress"].update(pilot=True,phase="resume_verification")\nprint(json.dumps(out))')
    import shlex
    while process.poll() is None:
        job=read(cloud.STATE/'ledger.json')['jobs'].get(name,{})
        requested=campaign.get('requested_stop_after_validation')
        if requested and job.get('pod_id') and not job.get('requested_stop_after_validation'):
            cloud.update(name,requested_stop_after_validation=requested,no_auto_resume=True)
        if not is_pilot and job.get('pod_id') and not job.get('released_at'):
            remaining=job['max_cost_usd']-max(job.get('account_spend_observed',0),
                           max(0,time.time()-job['created_at'])/3600*job['budget_rate'])
            credit=job.get('last_account_credit',job.get('initial_account_credit',float('inf')))
            if remaining<=5 or credit<=10:
                from datetime import datetime
                from zoneinfo import ZoneInfo
                stamp=datetime.now(ZoneInfo('Europe/Prague')).strftime('%Y-%m-%d %H:%M:%S %Z (Praha)')
                deliver_once(REG/'notifications.sqlite3',name+':funding-warning:'+str(job['max_cost_usd']),
                    '<b>Project: ThinkingCap / Self-replay scaled</b>\nSummary: Blíží se limit financované etapy.'
                    +(f'\nKredit Vast: {credit:.2f} USD' if provider=='vast' else '')+f'\nZbývající konzervativní rozpočet etapy: {max(0,remaining):.2f} USD'
                    +'\nPokračování vyžaduje kontrolu financování a zvýšení limitu etapy; samotné dobití limit nemění.'
                    +'\nPři dosažení limitu se uloží checkpoint a GPU se uvolní.\nTimestamp: '+stamp)
        if job.get('pod_id') and not job.get('released_at'):
            try:
                pod=cloud.api('pods/'+job['pod_id'])
                if pod.get('ssh'):
                    live=json.loads(cloud.remote(pod,'python3 -c '+shlex.quote(remote_script),timeout=25));live['observed_at']=time.time();write(REG/'live.json',live)
                    points=live.get('validation',{}).get('checkpoints',[])
                    if not is_pilot and points and points[0]['iteration']==0:
                        for point in points[1:]:
                            if point['iteration']<=launch.get('resume_group',0):continue
                            deliver_once(REG/'notifications.sqlite3',name+':validation:'+str(point['iteration']),validation_message(live['progress'],point,points[0]))
            except (RuntimeError,ValueError,OSError,subprocess.SubprocessError):pass
        time.sleep(30)
    job=read(cloud.STATE/'ledger.json')['jobs'].get(name,{})
    if not job.get('released_at'):
        # A polling/SSH failure is not evidence that detached GPU training stopped.
        campaign=read(REG/'campaign.json')
        campaign.update(state='pilot_running' if is_pilot else 'training_running',
                        monitor_error=f'Controller exited {process.returncode}; worker state requires observation',
                        monitor_error_at=time.time())
        write(REG/'campaign.json',campaign)
        raise RuntimeError('Controller exited before confirmed release; resume watch without restarting worker')
    proof=cloud.STATE/name/('output/gpu-verification.json' if is_pilot else 'output/training/completion.json')
    benchmark=cloud.STATE/name/'output/generation-benchmark/generation-benchmark.json'
    passed=bool(proof.exists() and read(proof).get('passed' if is_pilot else 'complete') and job.get('state')=='complete'
                and job.get('backup_verified') and job.get('released_at')
                and (not is_pilot or (benchmark.exists() and read(benchmark).get('complete'))))
    campaign=read(REG/'campaign.json')
    campaign.update(state=('pilot_passed' if passed else 'pilot_failed') if is_pilot else ('completed' if passed else 'stopped'),released=bool(job.get('released_at')),backup_verified=bool(job.get('backup_verified')),
                    verification=str(proof.relative_to(ROOT)) if proof.exists() else None)
    write(REG/'campaign.json',campaign)
    from datetime import datetime
    from zoneinfo import ZoneInfo
    stamp=datetime.now(ZoneInfo('Europe/Prague')).strftime('%Y-%m-%d %H:%M:%S %Z (Praha)')
    summary=('Technický test a přesná obnova checkpointu prošly.' if passed else 'Technický test nedokončen; hlavní trénink nespouštěn.') if is_pilot else ('Trénink dokončen, checkpointy zálohovány.' if passed else 'Trénink ukončen před cílem; zkontrolovat checkpoint a technický stav.')
    output=cloud.STATE/name/'output';details=''
    if not is_pilot:
        progress=read(output/'training/progress.json') if (output/'training/progress.json').exists() else {}
        latest=read(output/'training/latest-checkpoint.json') if (output/'training/latest-checkpoint.json').exists() else {}
        details=f'\nIterations (otázky): {progress.get("current",0)} / {campaign["total"]}\nCheckpoint: '+(str(latest['group']) if latest else 'žádný dokončený')
        if not passed and job.get('state')=='failed':
            logtext=(output/'runner.log').read_text(errors='replace') if (output/'runner.log').exists() else ''
            summary=('Technická chyba komunikace GPU (NCCL timeout).' if 'WorkNCCL' in logtext and 'Timeout' in logtext
                     else 'Technická chyba: worker skončil s chybou; podrobnosti jsou v archivovaném logu.')
            if progress.get('current')==0:summary+=' Trénink ještě neprovedl žádnou iteraci.'
        elif not passed and proof.exists():
            requested=job.get('requested_stop_after_validation')
            validated=output/'training/validations'/f'{requested:06d}.json' if type(requested) is int else None
            if (requested and progress.get('current')==requested and latest.get('group')==requested
                    and validated and validated.exists() and read(validated).get('iteration')==requested
                    and job.get('backup_verified')):
                summary=f'Na vaše přání zastaveno po validaci {requested}; úplný checkpoint zálohován.'
            else:
                summary=('Etapa ukončena kvůli místu pro zálohy; úplný checkpoint uložen.' if job.get('storage_stop_requested_at')
                         else 'Financovaná etapa ukončena; úplný checkpoint uložen pro pokračování.')
    message='<b>Project: ThinkingCap / Self-replay scaled</b>\nSummary: '+summary+'\nGPU: 2× A100-SXM4-80GB'+details+'\nGPU uvolněna: '+('ano' if job.get('released_at') else 'ne')+'\nTimestamp: '+stamp
    deliver_once(REG/'notifications.sqlite3',name+':terminal:'+str(job['released_at']),message)
    if not passed and (is_pilot or not (job.get('backup_verified') and job.get('released_at') and proof.exists())):
        raise RuntimeError('Run incomplete; inspect archived logs')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','prepare-main','prepare-repair','run','watch']);p.add_argument('--cap',type=float);a=p.parse_args()
    if a.action=='prepare':print(json.dumps(prepare(cap=a.cap or 10),indent=2))
    elif a.action in ('prepare-main','prepare-repair'):
        if a.cap is None:raise SystemExit('A concrete main-run cap is required')
        print(json.dumps(prepare('training',a.cap,repair=a.action=='prepare-repair'),indent=2))
    else:run(observe_only=a.action=='watch')
