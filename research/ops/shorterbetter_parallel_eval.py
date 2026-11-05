"""Four bounded evaluation workers for the twelve preserved alpha2/alpha5 snapshots."""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments import cloud,study_eval as study
from experiments.shorterbetter.paired_eval import bundle
from ops.shorterbetter_paired import read,sha,jobs
from ops.isolated_cloud import seal,verify
from ops import vastai
GROUPS={'control-early':['control-010','control-005','control-015'],
        'control-late':['control-020','control-025','control-030'],
        'accuracy-early':['accuracy-010','accuracy-005','accuracy-015'],
        'accuracy-late':['accuracy-020','accuracy-025','accuracy-030']}
INSTRUCTION='2026-10-31: User requests all alpha2/alpha5 checkpoint evaluations in parallel as soon as possible; four concurrent workers, no new training, within105USD campaign cap.'

def prior_cost(c):
 names={a['job_name'] for a in c['arms'].values()}|{e['job_name'] for e in c.get('early_evaluations',[])}
 if c.get('evaluation',{}).get('job_name'):names.add(c['evaluation']['job_name'])
 ledger=jobs()
 if any(not ledger[n].get('released_at') for n in names):raise RuntimeError('Earlier job still active')
 return sum(ledger[n]['charged_estimate_usd'] for n in names)

def prepare():
 cp=ROOT/read(ROOT/'lab/experiments/shorterbetter-paired/current.json')['path'];c=read(cp)
 if c.get('parallel_evaluations'):raise RuntimeError('Parallel campaign already registered; inspect existing workers')
 if vastai.list_pods():raise RuntimeError('Existing provider instance requires reconciliation')
 spent=prior_cost(c)
 if spent+22>c['cap_usd']:raise RuntimeError('Four5.5USD reservations exceed joint cap')
 if vastai.api('v0/users/current/')['credit']<25:raise RuntimeError('Insufficient funded credit')
 if shutil.disk_usage(ROOT).free<4*2**30:raise RuntimeError('Insufficient space for evaluation archives')
 for a in c['arms'].values():
  j=jobs()[a['job_name']]
  if not j.get('backup_verified'):raise RuntimeError('Training archive not verified')
 stamp=datetime.now(timezone.utc).strftime('%Y%m%dt%H%M%Sz').lower();directory=ROOT/'lab/experiments/shorterbetter-parallel-evaluation'/stamp;directory.mkdir(parents=True)
 base=ROOT/'lab/isolated/full-evaluation-20261026/teacher-student-v2';registered=[]
 for group,labels in GROUPS.items():
  name=f'shorterbetter-scaled-parallel-{group}-{stamp}';release=ROOT/'lab/isolated'/name;reg=directory/group;reg.mkdir()
  for folder in ['experiments','ThinkingCap-Test']:
   shutil.copytree(base/folder,release/folder,copy_function=os.link,ignore=shutil.ignore_patterns('__pycache__','.git','.env*','.venv*','runs'))
  for f in ['experiments/setup_gpu.sh','experiments/gpu_policy.py','experiments/study_eval.py','experiments/baseline_cache.py','experiments/full_replay_eval.py','experiments/shorterbetter/full_eval.py','experiments/shorterbetter/paired_eval.py','experiments/shorterbetter/hardware.py']:
   p=release/f;p.parent.mkdir(parents=True,exist_ok=True)
   if p.exists():p.unlink()
   shutil.copy2(ROOT/f,p)
  data=release/'lab/experiments/data';stage=data/'parallel-evaluation';stage.mkdir(parents=True)
  shutil.copy2(base/'lab/experiments/data/pilot.jsonl',data/'pilot.jsonl');shutil.copy2(base/'lab/gpu-requirements.lock',release/'lab/gpu-requirements.lock')
  repo=release/'ThinkingCap-Test';files=[repo/'evaluate.py',repo/'compare.py']+sorted((repo/'utils/eval').rglob('*.py'));candidates=[]
  for label in labels:
   arm,iteration=label.split('-');training=c['arms'][arm]['job_name'];source=cloud.STATE/training/'output/training/snapshots'/f'iteration-{iteration}'
   m=read(source/'snapshot-manifest.json')
   if m['optimizer_step']!=int(iteration)*4 or any(sha(source/f)!=h for f,h in m['files_sha256'].items()):raise RuntimeError('Snapshot integrity failure')
   target=stage/label/'candidate';shutil.copytree(source,target,copy_function=os.link);original=read(source/'config.json');(target/'config.json').unlink();cloud.write(target/'config.json',dict(original,dtype='bfloat16'))
   conf={'job_name':name,'evaluation_only':True,'training_authorized':False,'paired_short_evaluation':True,'samples':8,'evaluation_seed':9001,
      'candidates':['base','shorterbetter-final'],'model':study.BASE_MODEL,'revision':study.BASE_REVISION,'max_cost_usd':5.5,
      'cohort_sha256':sha(repo/'questions/questions.jsonl'),'questions_manifest_sha256':sha(repo/'questions/manifest.json'),
      'evaluator_files_sha256':{str(f.relative_to(repo)):sha(f) for f in files},
      'candidate':{'path':'candidate','files_sha256':{f.name:sha(f) for f in target.iterdir() if f.is_file()},'original_weights_sha256':sha(source/'model.safetensors'),
       'optimizer_step':int(iteration)*4,'original_model_config':original,'source_training_job':training,'alpha':c['arms'][arm]['alpha'],
       'selection':'Every5-rollout checkpoint, all12 included; exploratory repeatedly inspected benchmark.'}}
   cloud.write(stage/label/'evaluation.json',conf);candidates.append({'label':label,'config':f'{label}/evaluation.json'})
  config={'job_name':name,'evaluation_only':True,'training_authorized':False,'parallel_snapshot_shard':True,'registered_labels':labels,
    'candidates':candidates,'max_cost_usd':5.5,'same_worker_baseline':False,'baseline_policy':'gpu-model-and-protocol-cache-v1','fresh_holdout':False,'baseline_loading':'pinned_local',
    'protocol_version':'parallel-local-loading-v2','source_user_instruction':INSTRUCTION}
  cloud.write(stage/'bundle.json',config);cloud.write(reg/'bundle.json',config)
  with patch.object(study,'REPO',repo):bundle(stage/'bundle.json')
  entry=release/'experiments/shorterbetter/run-scaled.sh'
  if entry.exists():entry.unlink()
  entry.write_text('''#!/usr/bin/env bash
set -euo pipefail
source /workspace/project/experiments/setup_gpu.sh
export NLTK_DATA=/root/thinkingcap_nltk_data
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m experiments.shorterbetter.paired_eval --config data/parallel-evaluation/bundle.json --out output/paired-evaluation
''')
  payload=seal(release)
  launch={'job_name':name,'release':str(release),'payload_sha256':payload,'provider':'vast','disk_gb':60,'gpu':'NVIDIA GeForce RTX 4090','gpu_count':1,
    'phase':'evaluation','hours':1,'max_runtime_hours':8,'max_cost_usd':5.5,'max_gpu_rate':.55,'evaluation_only':True,'full_training_authorized':False,
    'telegram_authorized':True,'direct_ssh':True,'paired_evaluation_account_guard':True,'paired_campaign_path':str(cp)}
  cloud.write(reg/'launch.json',launch)
  registered.append({'group':group,'job_name':name,'labels':labels,'cap_usd':5.5,'directory':str(reg),'state':'prepared'})
 c.update(parallel_evaluations=registered,state='parallel_evaluation_prepared',parallel_evaluation_directory=str(directory),parallel_evaluation_instruction=INSTRUCTION)
 cloud.write(cp,c);cloud.write(directory/'campaign.json',{'parent_campaign':str(cp),'groups':registered,'prior_charged_usd':spent,'reserved_usd':22,'joint_cap_usd':105,'protocol_version':'parallel-local-loading-v2'})
 cloud.write(directory.parent/'current.json',{'directory':str(directory),'state':'prepared'})
 return directory

def quote_for(directory,group):
 reg=directory/group;l=read(reg/'launch.json')
 if l.get('offer_id'):return l
 excluded={60701}  # Measured unusable SSH throughput; advertised location/bandwidth not sufficient.
 for path in directory.glob('*/quote.json'):excluded.add(read(path)['offer']['machine_id'])
 q={'verified':{'eq':True},'rentable':{'eq':True},'rented':{'eq':False},'gpu_name':{'eq':'RTX 4090'},'num_gpus':{'eq':1},'gpu_ram':{'gte':24000},
    'cuda_max_good':{'gte':12.8},'disk_space':{'gte':60},'reliability2':{'gte':.99},'type':'on-demand','order':[['dph_total','asc']],'limit':1000}
 rows=vastai.api('v0/bundles/',params={'q':json.dumps(q)})['offers'];europe={'SE','NO','FI','DK','DE','NL','EE','GB','IE','ES','FR','PL','CZ','BG','RO','LT','LV','IT','AT','BE','CH'}
 rows=[r for r in rows if r.get('geolocation','').split(',')[-1].strip() in (europe|{'US','CA'}) and r['machine_id'] not in excluded and float(r.get('direct_port_count',0))>=1 and float(r.get('inet_down',0))>=700
       and float(r.get('duration',0))>=8.5*3600 and float(r['dph_total'])+60*float(r['storage_cost'])/720<=.55
       and not cloud.unhealthy_placement(jobs(),l['gpu'],'vast',{'offer':r,'offer_id':r['id']})]
 rows.sort(key=lambda r:(r.get('geolocation','').split(',')[-1].strip() not in europe,r['machine_id'] not in (38546,40176,31087,54653),-float(r.get('inet_down',0)),float(r['dph_total'])))
 for r in rows:
  offer=vastai.quote(l['gpu'],1,60,r['id'],8.5*3600)
  if offer and offer['hourly_ceiling']<=.55:
   l.update(offer_id=offer['offer_id'],max_gpu_rate=math.ceil(offer['hourly_ceiling']*1000)/1000)
   cloud.write(reg/'launch.json',l);cloud.write(reg/'quote.json',offer)
   cloud.write(cloud.STATE/'authorizations'/(l['job_name']+'.json'),dict(l,authorized=True,scope='shorterbetter_vast_bounded_v1',source_user_instruction=INSTRUCTION))
   return l
 raise RuntimeError('No distinct eligible European4090 capacity within cap')

def worker(directory,group):
 l=read(directory/group/'launch.json');n=l['job_name']
 if verify(Path(l['release']))!=l['payload_sha256']:raise RuntimeError('Sealed payload changed')
 if n in jobs():
  if not jobs()[n].get('released_at'):cloud.watch(n)
  return
 cmd=[sys.executable,str(ROOT/'ops/isolated_cloud.py'),'run','--release',l['release'],'--name',n,'--entry','experiments/shorterbetter/run-scaled.sh',
      '--gpu',l['gpu'],'--gpu-count','1','--hardware-scope','shorterbetter-scaled-v1','--hours','1','--max-cost-usd',str(l['max_cost_usd']),
      '--max-runtime-hours','8','--provider','vast','--vast-offer-id',str(l['offer_id']),'--vast-disk-gb','60','--data-upload-timeout','1800']
 result=subprocess.run(cmd,check=False)
 if result.returncode:raise RuntimeError(f'Canonical evaluator launcher exited {result.returncode}')

def comparison_equal(a,b):
 """Allow only floating-point roundoff across controller/worker Python versions."""
 if isinstance(a,dict) and isinstance(b,dict):return a.keys()==b.keys() and all(comparison_equal(a[k],b[k]) for k in a)
 if isinstance(a,list) and isinstance(b,list):return len(a)==len(b) and all(comparison_equal(x,y) for x,y in zip(a,b))
 if isinstance(a,float) or isinstance(b,float):return isinstance(a,(int,float)) and isinstance(b,(int,float)) and math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12)
 return type(a)==type(b) and a==b

def publish_pair(directory,row,label):
 """Verify the complete immutable pair before publishing it while other tests run."""
 out=cloud.STATE/row['job_name']/'output/paired-evaluation';comparison=out/'comparisons'/f'{label}.json'
 receipt=directory/'live-results'/row['job_name']/(label+'.json')
 if not comparison.exists():return False
 try:
  value=read(comparison)
  if value['ungraded_benchmarks'] or any(value['full_evaluation'].values()):return False
  if not receipt.exists():
   loading=read(out/'baseline-loading.json')
   if loading['revision']!=study.BASE_REVISION or loading['model']!=study.BASE_MODEL or Path(loading['path']).name!=study.BASE_REVISION:raise ValueError('Unpinned baseline')
   release=Path(read(Path(row['directory'])/'launch.json')['release']);repo=release/'ThinkingCap-Test';rows=study.read_jsonl(repo/'questions/questions.jsonl');cohort=sha(repo/'questions/questions.jsonl');proof={}
   for side in ['base',label]:
    run=out/'runs'/side;digest=sha(run/'records.jsonl')
    if digest!=read(out/'status'/(side+'.json'))['records_sha256']:return False
    model=loading['path'] if side=='base' else '/workspace/project/data/parallel-evaluation/'+label+'/candidate'
    study.validate_records(run,rows,8,cohort,'full',model=model,revision=None)
    proof[side]={'records_sha256':digest,'results_sha256':sha(run/'results.json')}
   receipt.parent.mkdir(parents=True,exist_ok=True);recomputed=receipt.with_suffix('.comparison.json')
   with receipt.with_suffix('.log').open('w') as log:
    subprocess.run([sys.executable,str(repo/'compare.py'),str(out/'runs/base'),str(out/'runs'/label),'--out',str(recomputed)],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
   computed=read(recomputed)
   for side,suffix in [('base','base'),('candidate',label)]:
    expected='/workspace/project/output/paired-evaluation/runs/'+suffix
    if value[side]['run']!=expected:raise ValueError('Unexpected worker run path')
    computed[side]['run']=expected
   if not comparison_equal(computed,value):raise ValueError('Recomputed comparison differs')
   provenance=read(out/'provenance.json')
   if provenance['job_name']!=row['job_name']:raise ValueError('Wrong worker provenance')
   cloud.write(receipt,{'verified':True,'source_job':row['job_name'],'label':label,'comparison_sha256':sha(comparison),'pairs':proof,
     'responses_per_model':22072,'numeric_comparison_tolerance':1e-12,'gpu_identity':provenance['gpu_identity'],'protocol_version':'parallel-local-loading-v2','verified_at':time.time()})
  if read(receipt)['comparison_sha256']!=sha(comparison):return False
  from ops.notify_shorterbetter_scaled import send_html
  from zoneinfo import ZoneInfo
  sys.path.insert(0,'/srv/playground');from project_alerts import Alerts
  arm,iteration=label.split('-');alpha=2 if arm=='control' else 5
  lines=['<b>Project: ThinkingCap / ShorterBetter</b>',f'Experiment: α={alpha}',f'Summary: Plná evaluace checkpointu {int(iteration)} ověřena.',
   'Evaluace: 2 759 otázek × 8 odpovědí pro baseline i checkpoint','GPU: 1× RTX4090 (vast)',f'Checkpoint: {int(iteration)} / 30',
   f"Accuracy: {value['average']['accuracy_delta']*100:+.2f} p. b.",f"Úspora tokenů: {value['average']['token_reduction']*100:.1f} %",
   'Výsledné soubory: zálohované a ověřené','Stav: úplné srovnání tohoto checkpointu',f'Timestamp: {datetime.now(ZoneInfo("Europe/Prague")):%Y-%m-%d %H:%M:%S %Z} (Praha)']
  stamp=lines.pop()
  for domain,title in [('svamp','Matematika'),('mbpp','Kódování'),('ifeval','Instrukce'),('bfcl','Nástroje')]:
   parts=[v for k,v in value.get('results',{}).items() if k==domain or (domain=='bfcl' and k.startswith('bfcl_'))]
   count=sum(v['paired_rows'] for v in parts)
   if not count:continue
   def mean(side,key):return sum(v['paired_rows']*v[side][key] for v in parts)/count
   lines+=['',title,f"Accuracy: {mean('base','accuracy'):.1%} → {mean('candidate','accuracy'):.1%}",f"Tokeny / odpověď: {mean('base','mean_output_tokens'):.0f} → {mean('candidate','mean_output_tokens'):.0f}"]
  job=jobs().get(row['job_name'],{});elapsed=int((job.get('released_at') or time.time())-job.get('created_at',time.time()));h,rest=divmod(max(0,elapsed),3600);m,sec=divmod(rest,60)
  lines+=['',f"GPU uvolněna: {'ano' if job.get('released_at') else 'ne, evaluace pokračuje'}",f'Time elapsed: {h:02d}:{m:02d}:{sec:02d}',stamp]
  alerts=Alerts(directory/'notifications.sqlite3','ThinkingCap / ShorterBetter',sender=send_html,formatter=lambda *args:'\n'.join(lines));key='checkpoint:'+label
  if not alerts.db.execute('SELECT 1 FROM incidents WHERE key=?',(key,)).fetchone():alerts.emit(key,'Results','Complete checkpoint comparison verified')
  return True
 except (OSError,ValueError,KeyError,RuntimeError,subprocess.SubprocessError) as exc:
  cloud.write(directory/(row['group']+'-publication-error.json'),{'label':label,'error':str(exc),'at':time.time()})
  return False

def notify_group(directory,row,j):
 from ops.notify_shorterbetter_scaled import send_html
 from zoneinfo import ZoneInfo
 sys.path.insert(0,'/srv/playground');from project_alerts import Alerts
 out=cloud.STATE/row['job_name']/'output/paired-evaluation';count=0
 lines=['<b>Project: ThinkingCap / ShorterBetter</b>',f"Experiment: α={2 if row['group'].startswith('control') else 5}",
  'Summary: Plná evaluace skupiny checkpointů dokončila běh.','Evaluace: 2 759 otázek × 8 odpovědí na model','GPU: 1× RTX4090 (vast)','GPU uvolněna: ano',f"Záloha ověřena: {'ano' if j.get('backup_verified') else 'ne'}"]
 if j.get('backup_verified'):
  cloud.verify_backup(row['job_name'])
  for label in row['labels']:
   p=out/'comparisons'/f'{label}.json'
   if not p.exists():continue
   v=read(p)
   if v['ungraded_benchmarks'] or any(v['full_evaluation'].values()):continue
   count+=1;a=v['average'];lines += ['',f"<b>Iterace {int(label.split('-')[1])}</b>",f"Accuracy: {a['accuracy_delta']*100:+.2f} p. b.",f"Úspora tokenů: {a['token_reduction']*100:.1f} %"]
 lines+=['',f'Hotová srovnání: {count} / 3',f'Timestamp: {datetime.now(ZoneInfo("Europe/Prague")):%Y-%m-%d %H:%M:%S %Z} (Praha)']
 alerts=Alerts(directory/'notifications.sqlite3','ThinkingCap / ShorterBetter',sender=send_html,formatter=lambda *args:'\n'.join(lines));key=row['group']
 if count<3 and not alerts.db.execute('SELECT 1 FROM incidents WHERE key=?',(key,)).fetchone():alerts.emit(key,'Technical','Parallel evaluation worker released incompletely')
 return count

def run(directory):
 registration=read(directory/'campaign.json');cp=Path(registration['parent_campaign'])
 for row in registration['groups']:
  l=quote_for(directory,row['group']);unit=l['job_name'].replace('shorterbetter-scaled-','thinkingcap-shorterbetter-')
  if row['job_name'] not in jobs() and subprocess.run(['systemctl','is-active','--quiet',unit]).returncode:
   subprocess.run(['systemd-run','--unit='+unit,'--property=WorkingDirectory='+str(ROOT),'--property=UMask=0077',sys.executable,str(Path(__file__).resolve()),'worker','--directory',str(directory),'--group',row['group']],check=True)
 c=read(cp);c['state']='parallel_evaluation_running';cloud.write(cp,c)
 cloud.write(directory.parent/'current.json',{'directory':str(directory),'state':'running'})
 done={}
 while len(done)<len(registration['groups']):
  ledger=jobs()
  for row in registration['groups']:
   n=row['job_name'];j=ledger.get(n,{})
   for label in row['labels']:publish_pair(directory,row,label)
   if n in done or not j.get('released_at'):continue
   if any(p['id']==j.get('pod_id') for p in vastai.list_pods()):continue
   done[n]=notify_group(directory,row,j)
   c=read(cp)
   for item in c['parallel_evaluations']:
    if item['job_name']==n:item.update(state='completed_released' if done[n]==3 else 'incomplete_released',comparisons=done[n])
   cloud.write(cp,c)
  if len(done)<len(registration['groups']):time.sleep(30)
 c=read(cp);c['state']='completed' if sum(done.values())==12 else 'parallel_evaluation_incomplete';cloud.write(cp,c)
 cloud.write(directory/'completion.json',{'complete':sum(done.values())==12,'comparisons':sum(done.values()),'groups':done})
 cloud.write(directory.parent/'current.json',{'directory':str(directory),'state':c['state']})
 from ops.notify_shorterbetter_scaled import send_html
 from zoneinfo import ZoneInfo
 sys.path.insert(0,'/srv/playground');from project_alerts import Alerts
 text='\n'.join(['<b>Project: ThinkingCap / ShorterBetter</b>',f"Summary: Paralelní evaluace skončily; ověřeno {sum(done.values())} / 12 checkpointů.",'Experimenty: α=2 a α=5','GPU uvolněny: ano',f'Timestamp: {datetime.now(ZoneInfo("Europe/Prague")):%Y-%m-%d %H:%M:%S %Z} (Praha)'])
 alerts=Alerts(directory/'notifications.sqlite3','ThinkingCap / ShorterBetter',sender=send_html,formatter=lambda *args:text)
 if not alerts.db.execute('SELECT 1 FROM incidents WHERE key=?',('all-final',)).fetchone():alerts.emit('all-final','Results','Parallel evaluations released')


if __name__=='__main__':
 os.umask(0o077);p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','worker']);p.add_argument('--directory',type=Path);p.add_argument('--group');a=p.parse_args()
 if a.action=='prepare':print(json.dumps({'directory':str(prepare())}))
 elif a.action=='run':run(a.directory.resolve())
 else:worker(a.directory.resolve(),a.group)
