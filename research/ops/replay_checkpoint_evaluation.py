"""Freeze and evaluate replay a selected checkpoint alongside its ongoing training job."""
import argparse,json,math,shutil,subprocess,sys,time
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments import cloud,study_eval as study,full_replay_eval
from experiments.replay_scaled.core import write,sha
from experiments.replay_scaled.evaluation_budget import SCOPE,ENTRY,GPU,startup_retry_budget
from experiments.replay_scaled.notify import deliver_once
from ops import vastai,runpod
from ops.isolated_cloud import seal,verify
PARENT='replay-scaled-training-20261101t122440z'
REGROOT=ROOT/'lab/experiments/replay-scaled-evaluation'
INSTRUCTION='2026-11-02: User explicitly requests repairing checkpoint576 evaluation and obtaining results. Sequential replacement after reconciled startup failure, within original combined4USD; mandatory cached4090 baseline, no training interruption.'

def read(p):return json.loads(Path(p).read_text())

def prepare(group=576,repair=False,provider='vast'):
    if group not in (128,256,576,768,850,1024,1280):raise ValueError('Checkpoint evaluation not registered for this request')
    if group!=1280 and (group==1024)!=(provider=='runpod'):raise ValueError('RunPod evaluation is registered only for1024')
    if group==1024 and repair:raise ValueError('Inspect failed1024 job and remaining budget before repair')
    parent=('replay-scaled-vast-resume1024-20261105t075310z' if group==1280 else 'replay-scaled-runpod-resume-20261102t162212z' if group==1024 else PARENT)
    jobs=read(cloud.STATE/'ledger.json')['jobs'];training=jobs[parent]
    if group in (1024,1280) and not (training.get('backup_verified') and training.get('released_at')):raise RuntimeError('RunPod training must be archived/released before evaluation')
    predecessors=[]
    concurrent=[]
    for path in REGROOT.glob('*/campaign.json'):
        previous=read(path);job=jobs.get(previous.get('job_name'),{})
        if previous.get('checkpoint_group')==group and repair:
            predecessors.append(previous['job_name'])
            continue
        if previous.get('checkpoint_group')!=group and job.get('released_at'):
            if job.get('backup_verified'):continue
            try:startup_retry_budget([job])
            except ValueError:pass
            else:continue
        if group==768 and previous.get('checkpoint_group')==576:
            if job.get('released_at'):
                if job.get('backup_verified') or (job.get('startup_failed_before_launch') and not job.get('worker_launch_attempted_at')) or (job.get('safely_no_pod') and not job.get('pod_id')):continue
            elif job.get('state')=='running' and job.get('identity_confirmed'):
                concurrent.append(previous['job_name']);continue
        if (job.get('released_at') and job.get('safely_no_pod') and not job.get('pod_id')
                and not job.get('allocation_attempts') and job.get('charged_estimate_usd')==0
                and job.get('state')=='allocation_unavailable'):
            continue  # Verified zero-rental capacity miss; preserve history and retry safely.
        if previous.get('checkpoint_group')==group or not job.get('released_at') or not job.get('backup_verified'):
            raise RuntimeError('Existing evaluation must be reconciled; never duplicate a checkpoint job')
    candidate=f'replay-{group}'
    cap=startup_retry_budget([jobs[n] for n in predecessors],ceiling=6 if group==1280 else 4) if repair else (6 if provider=='runpod' else 4)
    if group==1280 and provider=='vast':cap=min(cap,4)
    pods=cloud.list_pods(provider)
    allowed_pods={training['pod_id']}|{jobs[n]['pod_id'] for n in concurrent}
    if any(p['id'] not in allowed_pods for p in pods):raise RuntimeError('Reconcile other rentals before evaluation')
    checkpoint=cloud.STATE/parent/f'output/training/checkpoints/group-{group:06d}'
    manifest=read(checkpoint/'complete.json')
    if manifest['group']!=group or manifest['pilot'] or any(sha(checkpoint/f)!=h for f,h in manifest['files'].items()):
        raise RuntimeError('Complete checkpoint SHA verification failed')
    train_plan=read(ROOT/'lab/experiments/replay-scaled/20261101-checkpoints-v5/plan.json')
    if manifest['contract']!=sha(ROOT/'lab/experiments/replay-scaled/20261101-checkpoints-v5/plan.json') or train_plan['revision']!=study.BASE_REVISION:
        raise RuntimeError('Checkpoint base model provenance differs')
    account=runpod.account() if provider=='runpod' else vastai.api('v0/users/current/')
    credit=float(account['clientBalance'] if provider=='runpod' else account['credit'])
    if provider=='runpod' and (pods or account['currentSpendPerHr']):raise RuntimeError('Reconcile other RunPod spending before evaluation')
    training_reserve=0 if training.get('released_at') else max(0,training['hard_deadline']-time.time())/3600*training['budget_rate']+1
    other_reserve=sum(max(0,jobs[n]['hard_deadline']-time.time())/3600*jobs[n]['budget_rate'] for n in concurrent)
    if credit<cap+training_reserve+other_reserve+2:raise RuntimeError('Credit cannot safely cover evaluation and remaining training reservation')
    excluded={j.get('provider_quote',{}).get('offer',{}).get('machine_id') for j in jobs.values()
              if j.get('provider')=='vast' and j.get('startup_failed_before_launch')}
    rejected_offers={jobs[n].get('offer_id') for n in predecessors}
    if provider=='runpod':
        import urllib.parse
        quote=runpod.api('catalog/gpus/'+urllib.parse.quote(GPU)+'?include=AVAILABILITY&product=POD&count=1&minCudaVersion=12.8&cloud=SECURE')
        if quote.get('availability') not in ('LOW','MEDIUM','HIGH') or not 0<float(quote['price']['secure'])<=.99:raise RuntimeError('No suitable RunPod4090 capacity')
        rate=float(quote['price']['secure'])
    else:
        quote=vastai.quote(GPU,1,60,min_duration=6.5*3600,exclude_machine_ids=excluded-{None},exclude_offer_ids=rejected_offers-{None},
                          allowed_countries=('US','CA','GB','DE','FR','NL','PL','FI','SE','CZ','ES','IT','AT','RO','LT','BG'))
        if not quote or quote['hourly_ceiling']>.5:raise RuntimeError('No eligible bounded4090 offer')
        rate=quote['hourly_ceiling']
    stamp=time.strftime('%Y%m%dt%H%M%Sz',time.gmtime()).lower();name=f'replay-evaluation-{group}-'+stamp
    reg=REGROOT/stamp;reg.mkdir(parents=True)
    release=ROOT/'lab/isolated'/name
    frozen=ROOT/'lab/isolated/full-evaluation-20261026/teacher-student-v2'
    ignore=shutil.ignore_patterns('.git','.env*','.venv*','__pycache__','runs','*.log')
    shutil.copytree(ROOT/'experiments',release/'experiments',ignore=ignore)
    shutil.copytree(frozen/'ThinkingCap-Test',release/'ThinkingCap-Test',ignore=ignore)
    data=release/'lab/experiments/data';stage=data/'replay-evaluation';stage.mkdir(parents=True)
    if group in (768,850,1024,1280):
        shutil.copy2(REGROOT/'20261102t102548z/model-preload-source-verification.json',stage/'model-preload.json')
        shutil.copy2(REGROOT/'20261102t102548z/uv-wheel-source.json',stage/'uv-wheel-source.json')
    shutil.copy2(frozen/'lab/experiments/data/pilot.jsonl',data/'pilot.jsonl')
    shutil.copy2(frozen/'lab/gpu-requirements.lock',release/'lab/gpu-requirements.lock')
    shutil.copy2(ROOT/'lab/experiments/data/study/splits.json',stage/'splits.json')
    target=stage/'adapters'/candidate;shutil.copytree(checkpoint/'adapter',target)
    adapter_config=read(target/'adapter_config.json')
    if adapter_config['base_model_name_or_path']!=study.BASE_MODEL or adapter_config['revision'] not in (None,study.BASE_REVISION):
        raise RuntimeError('Unrecognized adapter base')
    # PEFT omitted revision metadata; weights stay byte-identical to the verified checkpoint.
    write(target/'adapter_config.json',dict(adapter_config,revision=study.BASE_REVISION))
    if sha(target/'adapter_model.safetensors')!=manifest['files']['adapter/adapter_model.safetensors']:raise RuntimeError('Weights changed')
    hashes=study.check_adapter(target)
    lock=dict(schema_version=1,candidate=candidate,primary_candidate=candidate,adapter_files_sha256=hashes,
              split_manifest_sha256=sha(stage/'splits.json'),locked_at=study.now(),selection=f'User-selected iteration{group}; explored benchmark, not untouched holdout')
    write(stage/'candidate-lock.json',lock)
    with patch.object(study,'REPO',release/'ThinkingCap-Test'):
        config=dict(candidates=['base',candidate],samples=8,replay_scaled_checkpoint=group,baseline_cache_required=True,evaluation_only=True,training_authorized=False,
                    evaluator_files_sha256=study.evaluator_hashes(),training_provenance=dict(job_name=parent,group=group,
                    checkpoint_manifest_sha256=sha(checkpoint/'complete.json'),original_adapter_config=adapter_config,
                    original_weights_sha256=sha(target/'adapter_model.safetensors'),metadata_normalization='Revision pinned from verified training plan; weights unchanged'))
        write(stage/'evaluation.json',config)
        full_replay_eval.inputs(stage/'evaluation.json')
    digest=seal(release)
    launch=dict(job_name=name,release=str(release),payload_sha256=digest,provider=provider,hardware_scope=SCOPE,entry=ENTRY,
                gpu=GPU,gpu_count=1,offer_id=quote.get('offer_id'),disk_gb=60,max_cost_usd=cap,max_runtime_hours=6,
                max_gpu_rate=math.ceil(rate*1000)/1000,phase='evaluation',evaluation_only=True,
                training_authorized=False,checkpoint_group=group,direct_ssh=True,concurrent_training_job=parent,
                account_spending_limit_usd=cap+training_reserve+other_reserve,
                concurrent_evaluation_jobs=concurrent,source_user_instruction=(f'2026-11-05: User requests full checkpoint1280 evaluation promptly, prefer RunPod with Vast fallback; cached baseline, UI/Telegram, training remains stopped.' if group==1280 else f'2026-11-03: User requests full checkpoint1024 evaluation on RunPod if credit suffices, and automatic UI results; training remains stopped.' if group==1024 else f'2026-11-02: User explicitly requests full checkpoint{group} evaluation now, preserving existing jobs and cached baseline.' if group in (768,850) else INSTRUCTION),
                previous_attempts=predecessors,combined_cap_usd=6 if group in (1024,1280) else 4)
    write(reg/'launch.json',launch);write(reg/'quote.json',quote);write(reg/'evaluation.json',config)
    write(reg/'campaign.json',dict(state='prepared',job_name=name,checkpoint_group=group,cap_usd=cap,credit_at_prepare=credit,
                                  previous_attempts=predecessors,combined_cap_usd=6 if group in (1024,1280) else 4,
                                  remaining_training_reserve_usd=training_reserve,remaining_evaluation_reserve_usd=other_reserve,created_at=time.time(),telegram_authorized=True))
    write(REGROOT/'current.json',dict(directory=str(reg)))
    return reg

def run(reg,resume_startup=False):
    launch=read(reg/'launch.json');name=launch['job_name'];campaign=read(reg/'campaign.json')
    group=launch['checkpoint_group'];candidate=f'replay-{group}'
    if verify(launch['release'])!=launch['payload_sha256']:raise RuntimeError('Sealed evaluator changed')
    if name in read(cloud.STATE/'ledger.json')['jobs'] and not resume_startup:raise RuntimeError('Existing evaluation: never allocate twice')
    provider=launch.get('provider','vast')
    write(cloud.STATE/'authorizations'/(name+'.json'),dict(launch,authorized=True,scope='replay_runpod_evaluation_v1' if provider=='runpod' else 'replay_vast_bounded_v1'))
    campaign.update(state='running');write(reg/'campaign.json',campaign)
    command=[sys.executable,str(ROOT/'ops/isolated_cloud.py'),'run','--release',launch['release'],'--name',name,
             '--entry',ENTRY,'--gpu',GPU,'--gpu-count','1','--hardware-scope',SCOPE,'--hours','1','--max-cost-usd',str(launch['max_cost_usd']),
             '--max-runtime-hours','6','--provider',provider,'--data-upload-timeout','1800' if resume_startup or group in (768,850,1024,1280) else '900']
    if provider=='vast':command+=['--vast-offer-id',str(launch['offer_id']),'--vast-disk-gb','60']
    if resume_startup:command.append('--resume-startup')
    with (reg/'controller.log').open('a') as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        while process.poll() is None:
            output=cloud.STATE/name/'output/full-evaluation'
            campaign['models']={label:read(output/'status'/(label+'.json')) for label in ['base',candidate] if (output/'status'/(label+'.json')).exists()}
            campaign['observed_at']=time.time();write(reg/'campaign.json',campaign);time.sleep(30)
    job=read(cloud.STATE/'ledger.json')['jobs'][name];output=cloud.STATE/name/'output/full-evaluation'
    complete=bool(job.get('backup_verified') and job.get('released_at') and job.get('state')=='complete'
                  and (output/'completion.json').exists() and read(output/'completion.json').get('complete'))
    campaign.update(state='complete' if complete else 'failed',backup_verified=bool(job.get('backup_verified')),
                    released=bool(job.get('released_at')),output=str(output),finished_at=time.time())
    from datetime import datetime
    from zoneinfo import ZoneInfo
    lines=['<b>Project: ThinkingCap / Self-replay scaled</b>',
           'Summary: '+(f'Plná evaluace checkpointu{group} dokončena.' if complete else f'Plná evaluace checkpointu{group} selhala; archiv a logy zachovány.'),
           f'Evaluace: 2 759 otázek × 8 odpovědí pro checkpoint{group}; baseline z cache','GPU: 1× RTX 4090']
    if complete:
        result=read(output/'comparison.json')
        if result['ungraded_benchmarks'] or any(result['full_evaluation'].values()):raise RuntimeError('Incomplete official comparison')
        average=result['average'];campaign['average']=average
        lines += [f'Změna accuracy: {100*average["accuracy_delta"]:+.2f} p. b.',f'Oficiální úspora tokenů: {100*average["token_reduction"]:.1f} %']
        from experiments.full_replay_campaign import benchmark_metrics
        base=benchmark_metrics(read(output/'runs/base/results.json'));candidate_metrics=benchmark_metrics(read(output/'runs'/candidate/'results.json'))
        for domain,label in [('svamp','Matematika'),('mbpp','Kódování'),('bfcl','Nástroje'),('ifeval','Instrukce')]:
            b,c=base[domain],candidate_metrics[domain]
            lines += ['',label,f'Accuracy: {100*b["accuracy"]:.1f}% → {100*c["accuracy"]:.1f}%',f'Tokeny / odpověď: {b["mean_output_tokens"]:.0f} → {c["mean_output_tokens"]:.0f}']
    lines += ['','GPU uvolněna: '+('ano' if job.get('released_at') else 'ne'),
              'Timestamp: '+datetime.now(ZoneInfo('Europe/Prague')).strftime('%Y-%m-%d %H:%M:%S %Z (Praha)')]
    write(reg/'campaign.json',campaign)
    deliver_once(reg/'notifications.sqlite3',name+':terminal','\n'.join(lines))
    subprocess.run([sys.executable,str(ROOT/'experiment-ui/publish.py')],check=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);p.add_argument('--directory',type=Path);p.add_argument('--checkpoint',type=int,choices=[128,256,576,768,850,1024,1280],default=576);p.add_argument('--provider',choices=['vast','runpod'],default='vast');p.add_argument('--repair-startup',action='store_true');p.add_argument('--resume-startup',action='store_true');a=p.parse_args()
    if a.action=='prepare':print(prepare(a.checkpoint,repair=a.repair_startup,provider=a.provider))
    else:run(a.directory or Path(read(REGROOT/'current.json')['directory']),resume_startup=a.resume_startup)
