"""Named ABC registration and canonical controller; no separate rental lifecycle."""
import argparse,datetime,json,math,shlex,shutil,subprocess,sys,time
from pathlib import Path
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments import cloud
from experiments.opencoder_sft.runtime import read,write,sha
from experiments.coding_abc.budget import SCOPE,PREFIX,ENTRY,GPU,ARMS
from experiments.coding_abc.prepare import prepare as prepare_data
from experiments.replay_scaled.notify import deliver_once
from ops import vastai
from ops.isolated_cloud import seal,verify
REG=ROOT/'lab/experiments/coding-abc'

def prepare():
    if (REG/'current.json').exists():raise RuntimeError('ABC already registered; inspect existing campaign')
    if vastai.list_pods():raise RuntimeError('Wait for existing survey release')
    credit=float(vastai.api('v0/users/current/')['credit'])
    if credit<42:raise RuntimeError('Need36USD pilot plus6USD reserve')
    stamp=time.strftime('%Y%m%dt%H%M%Sz',time.gmtime()).lower();reg=REG/stamp;reg.mkdir(parents=True)
    common=prepare_data(reg/'data');write(reg/'common-plan.json',common)
    instruction='User2026-11-04 requests pause APPS/TACO/KodCode surveys and run A/B/C from768, own answers only, no length preference, three parallel4090 pilots.'
    campaign=dict(authorized=True,state='prepared',cap_usd=36,credit_at_prepare=credit,created_at=time.time(),arms={},source_user_instruction=instruction)
    excluded=set()
    for label in ARMS:
        quote=vastai.quote(GPU,1,80,min_duration=18*3600,exclude_machine_ids=excluded,
            allowed_countries=('US','CA','GB','DE','FR','NL','PL','FI','SE','CZ','ES','IT','AT','RO','LT','BG','IE'))
        if not quote or quote['hourly_ceiling']>.6:raise RuntimeError('Suitable4090 capacity unavailable')
        excluded.add(quote['offer']['machine_id']);name=PREFIX+label+'-'+stamp;arm=reg/label;arm.mkdir()
        release=ROOT/'lab/isolated'/name
        shutil.copytree(ROOT/'experiments',release/'experiments',ignore=shutil.ignore_patterns('__pycache__','.git','.env*','.venv*','*.log'))
        (release/'ThinkingCap-Test').mkdir();stage=release/'lab/experiments/data/coding-abc';shutil.copytree(reg/'data',stage)
        plan=dict(common,scope=SCOPE,arm=label);write(stage/'plan.json',plan);write(arm/'plan.json',plan)
        preload=ROOT/'lab/experiments/replay-scaled-evaluation/20261102t102548z';target=release/'lab/experiments/data/replay-evaluation';target.mkdir()
        shutil.copy2(preload/'model-preload-source-verification.json',target/'model-preload.json');shutil.copy2(preload/'uv-wheel-source.json',target/'uv-wheel-source.json')
        (release/'lab/gpu-requirements.lock').write_text((ROOT/'lab/gpu-requirements.lock').read_text()+'\npytest==8.3.5\niniconfig==2.3.1\npluggy==1.6.0\n')
        digest=seal(release)
        launch=dict(job_name=name,release=str(release),payload_sha256=digest,authorized=True,scope=SCOPE,hardware_scope=SCOPE,entry=ENTRY,
            provider='vast',gpu=GPU,gpu_count=1,arm=label,parent_group=768,offer_id=quote['offer_id'],disk_gb=80,max_cost_usd=12,
            max_runtime_hours=18,max_gpu_rate=math.ceil(quote['hourly_ceiling']*1000)/1000,training_authorized=True,
            direct_ssh=True,campaign_path=str(reg/'campaign.json'),combined_cap_usd=36,source_user_instruction=instruction)
        write(arm/'launch.json',launch);write(arm/'quote.json',quote);write(arm/'status.json',dict(state='prepared',job_name=name))
        campaign['arms'][label]=dict(job_name=name,directory=str(arm),max_cost_usd=12)
    write(reg/'campaign.json',campaign);write(REG/'current.json',dict(directory=str(reg)))
    for label in ARMS:
        v=read(reg/label/'launch.json');write(cloud.STATE/'authorizations'/(v['job_name']+'.json'),v)
    return reg

def message(label,p,point=None,terminal=False,released=False,error=None):
    names={'a':'A · pouze RL','b':'B · pouze SFT','c':'C · RL + replay'}
    seconds=int(p.get('elapsed',0));lines=['<b>Project: ThinkingCap / Coding ABC · '+names[label]+'</b>',
        'Summary: '+('Chyba: '+error if error else 'Pilot ukončen; výsledky a checkpointy uloženy.' if terminal else 'Dokončena development validace.'),
        'GPU: 1× RTX 4090 (Vast)',f"Kroky (otázky): {p.get('current',0)} / {p.get('total',48)}",f"Updates (optimizer): {p.get('optimizer_updates',0)}",
        f'Time elapsed: {seconds//3600:02d}:{seconds%3600//60:02d}:{seconds%60:02d}',
        'Timestamp: '+datetime.datetime.now(ZoneInfo('Europe/Prague')).strftime('%Y-%m-%d %H:%M:%S %Z (Praha)')]
    if point:
        lines+=['',f"Validace: krok {point['iteration']} · development64×2, není plná evaluace"]
        for d,title in [('overall','Celkem'),('coding','Kódování'),('math','Matematika'),('tools','Nástroje'),('instructions','Instrukce')]:
            m=point['metrics'][d];lines+=['',title,f"Accuracy: {100*m['accuracy']:.1f}% ({100*m.get('accuracy_delta',0):+.1f} p. b. proti768)",
                f"Tokeny / odpověď: {m['mean_tokens']:.0f}",f"Useknuté odpovědi: {100*m['truncation']:.1f}%"]
    if terminal:lines+=['','GPU uvolněna: '+('ano' if released else 'ne')]
    return '\n'.join(lines)

def run(reg,label):
    arm=reg/label;launch=read(arm/'launch.json');name=launch['job_name']
    if verify(launch['release'])!=launch['payload_sha256']:raise RuntimeError('Sealed ABC source changed')
    jobs=read(cloud.STATE/'ledger.json')['jobs']
    cmd=[sys.executable,str(ROOT/'ops/isolated_cloud.py')]
    if name in jobs:cmd+=['watch','--release',launch['release'],'--name',name]
    else:
        cmd+=['run','--release',launch['release'],'--name',name,'--entry',ENTRY,'--gpu',GPU,'--gpu-count','1','--hardware-scope',SCOPE,
            '--hours','1','--max-cost-usd',str(launch['max_cost_usd']),'--max-runtime-hours',str(launch['max_runtime_hours']),'--provider','vast','--vast-disk-gb','80',
            '--data-upload-timeout','1800']
        if launch.get('offer_id') is not None:cmd+=['--vast-offer-id',str(launch['offer_id'])]
    state=dict(state='starting',job_name=name,started_at=time.time());write(arm/'status.json',state)
    with (arm/'controller.log').open('a') as log:
        proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
        while proc.poll() is None:
            j=read(cloud.STATE/'ledger.json')['jobs'].get(name,{})
            if j.get('state')=='running' and j.get('pod_id') and not j.get('released_at'):
                try:
                    pod=cloud.api('pods/'+j['pod_id'])
                    script="import json;from pathlib import Path;p=Path('/workspace/project/output/training');print(json.dumps({n:json.loads((p/n).read_text()) for n in ['progress.json','validation-curve.json','latest-checkpoint.json','preflight.json','failure.json'] if (p/n).exists()}))"
                    live=json.loads(cloud.remote(pod,'python3 -c '+shlex.quote(script),timeout=20))
                    state.update(state='running',observed_at=time.time(),live=live);state.pop('error',None)
                    for point in live.get('validation-curve.json',{}).get('checkpoints',[]):
                        if point['iteration']>0:deliver_once(reg/'notifications.sqlite3',name+':validation:'+str(point['iteration']),message(label,live.get('progress.json',{}),point))
                except Exception as e:state['error']=str(e)[:350]
                write(arm/'status.json',state)
            time.sleep(30)
    j=read(cloud.STATE/'ledger.json')['jobs'].get(name,{});out=cloud.STATE/name/'output/training'
    live={n:read(out/n) for n in ['progress.json','validation-curve.json','latest-checkpoint.json','preflight.json','failure.json','completion.json'] if (out/n).exists()}
    good=bool(j.get('backup_verified') and j.get('released_at'));complete=live.get('completion.json',{}).get('complete')
    official=read(out.parent/'full-evaluation/completion.json') if (out.parent/'full-evaluation/completion.json').exists() else {}
    comparison=read(out.parent/'full-evaluation/comparison.json') if (out.parent/'full-evaluation/comparison.json').exists() else {}
    needs_official=read(arm/'plan.json').get('final_official_evaluation',False)
    success=good and complete and j.get('state')=='complete' and (not needs_official or official.get('complete'))
    state.update(state='complete' if success else 'stopped' if good and 'completion.json' in live else 'failed',
        live=live,finished_at=time.time(),released=bool(j.get('released_at')),backup_verified=bool(j.get('backup_verified')),official_results=comparison if official.get('complete') else None);write(arm/'status.json',state)
    points=live.get('validation-curve.json',{}).get('checkpoints',[])
    terminal=message(label,live.get('progress.json',{}),points[-1] if points else None,terminal=True,released=bool(j.get('released_at')),error=live.get('failure.json',{}).get('error') or ('Běh nedokončen; kontroluji archiv.' if state['state']=='failed' else None))
    if official.get('complete') and comparison:
        av=comparison['average'];terminal+='\n\nPlná oficiální evaluace: 2759×8 · baseline z cache\nAccuracy proti původní baseline: '+f"{100*av['accuracy_delta']:+.2f} p. b."+'\nÚspora tokenů: '+f"{100*av['token_reduction']:.1f}%"
        for key,title in [('mbpp','Kódování'),('svamp','Matematika'),('ifeval','Instrukce')]:
            m=comparison['results'][key];terminal+='\n'+title+': '+f"{100*m['base']['accuracy']:.1f}% → {100*m['candidate']['accuracy']:.1f}%"
    elif needs_official:terminal+='\nPlná oficiální evaluace zatím není dokončena.'
    deliver_once(reg/'notifications.sqlite3',name+':terminal',terminal)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);p.add_argument('--arm',choices=ARMS);p.add_argument('--directory',type=Path);a=p.parse_args()
    if a.action=='prepare':print(prepare())
    else:run(a.directory or Path(read(REG/'current.json')['directory']),a.arm)
