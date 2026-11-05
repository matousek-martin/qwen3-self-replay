"""Checkpoint256 curricula using the canonical allocation, lease, backup and release."""
import argparse,datetime,json,math,shlex,shutil,subprocess,sys,time
from pathlib import Path
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments import cloud
from experiments.opencoder_sft.runtime import read,write
from experiments.coding_curriculum.budget import SCOPE,PREFIX,ENTRY,GPU,CAP,ticket
from ops import vastai
from ops.isolated_cloud import seal,verify
REG=ROOT/'lab/experiments/coding-curriculum'

def prepare(data):
    if (REG/'current.json').exists():raise RuntimeError('Campaign already registered; inspect before allocating')
    jobs=read(cloud.STATE/'ledger.json')['jobs'];now=time.time();pods=vastai.list_pods()
    active={n:j for n,j in jobs.items() if not j.get('released_at') and j.get('pod_id')}
    if {p['id'] for p in pods}!={j['pod_id'] for j in active.values()}:raise RuntimeError('Provider/ledger mismatch')
    reserve=sum(max(0,j['max_cost_usd']-(now-j['created_at'])/3600*j['budget_rate']) for j in active.values())
    credit=float(vastai.api('v0/users/current/')['credit'])
    if credit<CAP+reserve+3:raise RuntimeError('Insufficient credit after protecting existing reservations')
    stamp=time.strftime('%Y%m%dt%H%M%Sz',time.gmtime()).lower();reg=REG/stamp;reg.mkdir()
    name=PREFIX+stamp;release=ROOT/'lab/isolated'/name
    instruction='2026-11-04 user: prepare and launch checkpoint256 KodCode or TACO easiest-to-hardest replay experiments, at least128iterations,2xA100; monitor and ensure real training.'
    campaign=dict(authorized=True,state='prepared',job_name=name,datasets=['kodcode','taco'],cap_usd=CAP,
        previous_attempts=[],credit_at_prepare=credit,protected_jobs=list(active),protected_reserve_usd=reserve,
        source_user_instruction=instruction,created_at=now,iterations_per_branch=128,coding_per_branch=32)
    write(reg/'campaign.json',campaign)
    shutil.copytree(ROOT/'experiments',release/'experiments',ignore=shutil.ignore_patterns('__pycache__','.git','.env*','.venv*','*.log'))
    (release/'ThinkingCap-Test').mkdir()
    shutil.copytree(data,release/'lab/experiments/data/coding-curriculum')
    preload=ROOT/'lab/experiments/replay-scaled-evaluation/20261102t102548z';target=release/'lab/experiments/data/replay-evaluation';target.mkdir()
    shutil.copy2(preload/'model-preload-source-verification.json',target/'model-preload.json')
    shutil.copy2(preload/'uv-wheel-source.json',target/'uv-wheel-source.json')
    (release/'lab/gpu-requirements.lock').write_text((ROOT/'lab/gpu-requirements.lock').read_text()+'\npytest==8.3.5\niniconfig==2.3.1\npluggy==1.6.0\n')
    digest=seal(release)
    launch=dict(job_name=name,release=str(release),payload_sha256=digest,authorized=True,scope=SCOPE,hardware_scope=SCOPE,entry=ENTRY,
        provider='vast',gpu=GPU,gpu_count=2,parent_group=256,offer_id=None,disk_gb=80,max_cost_usd=CAP,max_runtime_hours=10,
        max_gpu_rate=1.05,training_authorized=True,direct_ssh=True,campaign_path=str(reg/'campaign.json'),combined_cap_usd=CAP,
        account_spending_limit_usd=CAP+reserve,source_user_instruction=instruction)
    write(reg/'launch.json',launch);write(cloud.STATE/'authorizations'/(name+'.json'),launch)
    if not ticket(cloud.STATE,name,launch):raise RuntimeError('Invalid curriculum ticket')
    write(REG/'current.json',dict(directory=str(reg)))
    write(reg/'status.json',dict(state='prepared',job_name=name))
    print(json.dumps(dict(directory=str(reg),job_name=name,credit=credit,protected_reserve_usd=reserve,cap_usd=CAP)),flush=True)
    return reg

def live_read(pod):
    script="""import json,collections
from pathlib import Path
p=Path('/workspace/project/output');result={}
for name in ['current-branch.json','gpu-verification.json','gpu.txt']:
 if (p/name).exists():result[name]=(p/name).read_text() if name.endswith('.txt') else json.loads((p/name).read_text())
for branch in ['kodcode','taco']:
 result[branch]={}
 for name in ['progress.json','validation-curve.json','latest-checkpoint.json','parent-initialization.json','completion.json']:
  f=p/branch/name
  if f.exists():result[branch][name]=json.loads(f.read_text())
 f=p/branch/'training-records.jsonl'
 if f.exists():
  groups=collections.defaultdict(list)
  for line in f.read_text().splitlines():
   try:r=json.loads(line)
   except ValueError:continue
   if r['task']=='coding':groups[r['group']].append(r)
  complete=[rs for rs in groups.values() if len(rs)==4]
  result[branch]['coding-diagnostics.json']=dict(questions=len(complete),
   correct_per_group=dict(collections.Counter(sum(r['correct'] for r in rs) for rs in complete)),
   truncated_answers=sum(r['truncated'] for rs in complete for r in rs),answers=len(complete)*4)
print(json.dumps(result))
"""
    return json.loads(cloud.remote(pod,'python3 -c '+shlex.quote(script),timeout=25))

def repair(reg):
    c=read(reg/'campaign.json');old=read(reg/'launch.json');jobs=read(cloud.STATE/'ledger.json')['jobs'];j=jobs[old['job_name']]
    if not (j.get('released_at') and j.get('backup_verified')):raise RuntimeError('Previous worker not archived/released')
    if any(p['id']==j.get('pod_id') for p in vastai.list_pods()):raise RuntimeError('Previous instance still present')
    oldout=cloud.STATE/old['job_name']/'output'
    if any((oldout/b/'parent-initialization.json').exists() for b in ('kodcode','taco')):
        raise RuntimeError('Branch may have trained; preserve and resume complete state, never restart parent')
    previous=c.get('previous_attempts',[])+[old['job_name']]
    spent=sum(jobs[n]['charged_estimate_usd'] for n in previous)
    name=PREFIX+reg.name+'-retry'+str(len(previous));release=ROOT/'lab/isolated'/name
    shutil.copytree(ROOT/'experiments',release/'experiments',ignore=shutil.ignore_patterns('__pycache__','.git','.env*','.venv*','*.log'))
    shutil.copytree(Path(old['release'])/'ThinkingCap-Test',release/'ThinkingCap-Test')
    shutil.copytree(Path(old['release'])/'lab/experiments/data/coding-curriculum',release/'lab/experiments/data/coding-curriculum')
    shutil.copytree(Path(old['release'])/'lab/experiments/data/replay-evaluation',release/'lab/experiments/data/replay-evaluation')
    shutil.copy2(Path(old['release'])/'lab/gpu-requirements.lock',release/'lab/gpu-requirements.lock')
    launch=dict(old,job_name=name,release=str(release),payload_sha256=seal(release),max_cost_usd=round(CAP-spent,4))
    c.update(job_name=name,previous_attempts=previous,state='prepared',repair_reason='Communication preflight scoped GPU authorization dispatch; no training occurred')
    write(reg/('launch-attempt'+str(len(previous)-1)+'.json'),old);write(reg/'campaign.json',c)
    write(reg/'launch.json',launch);write(cloud.STATE/'authorizations'/(name+'.json'),launch)
    if not ticket(cloud.STATE,name,launch):raise RuntimeError('Repair ticket invalid')
    write(reg/'status.json',dict(state='prepared',job_name=name))
    print(json.dumps(dict(job_name=name,remaining_cap=launch['max_cost_usd'],prior_charge=spent)))

def notify(reg,name,live,terminal=False):
    from experiments.replay_scaled.notify import deliver_once
    for branch in ('kodcode','taco'):
        data=live.get(branch,{});p=data.get('progress.json',{})
        for point in data.get('validation-curve.json',{}).get('checkpoints',[]):
            if not point['iteration']:continue
            lines=['<b>Project: ThinkingCap / Replay256 · '+branch+'</b>',
                'Summary: Dokončena development validace.','GPU: 2× A100-SXM4-80GB',
                f"Iterace: {p.get('current',0)} / 128",f"Validace: {point['iteration']}",
                f"Elapsed: {p.get('elapsed',0)/3600:.2f} h",
                'Timestamp: '+datetime.datetime.now(ZoneInfo('Europe/Prague')).strftime('%Y-%m-%d %H:%M:%S %Z')]
            for d in ['coding','math','tools','instructions']:
                m=point['metrics'][d];lines += [d,f"Accuracy: {100*m['accuracy']:.2f}%",f"Tokeny / odpověď: {m['mean_tokens']:.0f}"]
            deliver_once(reg/'notifications.sqlite3',name+':'+branch+':validation:'+str(point['iteration']),'\n'.join(lines))
    if terminal:
        j=read(cloud.STATE/'ledger.json')['jobs'][name]
        summary='; '.join(b+': '+str(live.get(b,{}).get('progress.json',{}).get('current',0))+'/128' for b in ('kodcode','taco'))
        deliver_once(reg/'notifications.sqlite3',name+':terminal','<b>Project: ThinkingCap / Replay256 curricula</b>\nSummary: '+summary+
            '\nState: '+j['state']+'\nGPU: 2× A100-SXM4-80GB\nBackup verified: '+str(bool(j.get('backup_verified')))+
            '\nReleased: '+str(bool(j.get('released_at')))+'\nTimestamp: '+datetime.datetime.now(ZoneInfo('Europe/Prague')).strftime('%Y-%m-%d %H:%M:%S %Z'))

def run(reg):
    l=read(reg/'launch.json');name=l['job_name']
    if verify(l['release'])!=l['payload_sha256']:raise RuntimeError('Source seal mismatch')
    jobs=read(cloud.STATE/'ledger.json')['jobs'];cmd=[sys.executable,str(ROOT/'ops/isolated_cloud.py')]
    if name in jobs:
        if jobs[name].get('released_at'):proc=None
        else:cmd+=['watch','--release',l['release'],'--name',name];proc=True
    else:
        # Recheck actual remaining funding immediately before the only allocation.
        credit=float(vastai.api('v0/users/current/')['credit']);now=time.time()
        reserve=sum(max(0,j['max_cost_usd']-(now-j['created_at'])/3600*j['budget_rate']) for j in jobs.values()
                    if not j.get('released_at') and j.get('pod_id'))
        if credit<l['max_cost_usd']+reserve+3:raise RuntimeError('Protected concurrent funding unavailable')
        cmd+=['run','--release',l['release'],'--name',name,'--entry',ENTRY,'--gpu',GPU,'--gpu-count','2','--hardware-scope',SCOPE,
            '--hours','1','--max-cost-usd',str(l['max_cost_usd']),'--max-runtime-hours','10','--provider','vast','--vast-disk-gb','80','--data-upload-timeout','1800']
        proc=True
    state=read(reg/'status.json') if (reg/'status.json').exists() else {};state.update(job_name=name,state='starting')
    if proc:
        with (reg/'controller.log').open('a') as log:
            proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
            while proc.poll() is None:
                j=read(cloud.STATE/'ledger.json')['jobs'].get(name,{})
                state.update(state=j.get('state','starting'),updated_at=time.time())
                if j.get('pod_id') and not j.get('released_at'):
                    try:
                        state['live']=live_read(cloud.api('pods/'+j['pod_id']));state['observed_at']=time.time();state.pop('error',None)
                        notify(reg,name,state['live'])
                    except Exception as exc:state['error']=str(exc)[:350]
                write(reg/'status.json',state);time.sleep(30)
    j=read(cloud.STATE/'ledger.json')['jobs'].get(name,{})
    if not j.get('released_at'):
        state.update(state='monitor_error',error='Controller exited without confirmed release; watch existing worker only')
        write(reg/'status.json',state);raise RuntimeError(state['error'])
    out=cloud.STATE/name/'output';live={}
    for b in ('kodcode','taco'):
        live[b]={f.name:read(f) for f in (out/b).glob('*.json') if f.name in ['progress.json','validation-curve.json','latest-checkpoint.json','completion.json']}
    complete=all(live[b].get('completion.json',{}).get('complete') for b in live)
    state.update(state='complete' if complete and j.get('backup_verified') else 'incomplete',live=live,
        released=True,backup_verified=bool(j.get('backup_verified')),finished_at=time.time())
    write(reg/'status.json',state);c=read(reg/'campaign.json');c['state']=state['state'];write(reg/'campaign.json',c);notify(reg,name,live,terminal=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','repair']);p.add_argument('--data',type=Path);p.add_argument('--directory',type=Path);a=p.parse_args()
    if a.action=='prepare':prepare(a.data)
    elif a.action=='repair':repair(a.directory or Path(read(REG/'current.json')['directory']))
    else:run(a.directory or Path(read(REG/'current.json')['directory']))
