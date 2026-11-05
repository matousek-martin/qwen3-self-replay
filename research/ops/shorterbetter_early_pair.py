"""One bounded full evaluation of alpha2 iterations10/20 and a fresh shared baseline."""
import argparse
from datetime import datetime,timezone
import json
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
from ops import vastai
from ops.isolated_cloud import seal,verify

def prepare():
 campaign_path=ROOT/read(ROOT/'lab/experiments/shorterbetter-paired/current.json')['path'];c=read(campaign_path);ledger=jobs()
 previous=c.get('early_evaluations',[])
 if not all(ledger[e['job_name']].get('released_at') for e in previous):raise RuntimeError('Previous evaluator still active')
 cap=round(10-sum(ledger[e['job_name']]['charged_estimate_usd'] for e in previous),4)
 if cap<5:raise RuntimeError('Insufficient bounded evaluation budget')
 training=c['arms']['control']['job_name'];train_job=ledger[training]
 active_training={ledger[a['job_name']]['pod_id'] for a in c['arms'].values() if a['job_name'] in ledger and not ledger[a['job_name']].get('released_at')}
 if {p['id'] for p in vastai.list_pods()}!=active_training:raise RuntimeError('Unexpected concurrent provider instance')
 if any((cloud.STATE/e['job_name']/'output/paired-evaluation/completion.json').exists() for e in previous if e.get('early_control_pair')):raise RuntimeError('A previous pair completed; do not duplicate it')
 source_root=cloud.STATE/training/'output/training/snapshots';proof={}
 for i in (10,20):
  source=source_root/f'iteration-{i:03d}';m=read(source/'snapshot-manifest.json')
  if m['optimizer_step']!=i*4 or not all(sha(source/f)==h for f,h in m['files_sha256'].items()):raise RuntimeError('Snapshot mismatch')
  if train_job.get('released_at'):
   if not train_job.get('backup_verified'):raise RuntimeError('Released source archive is not verified')
  else:
   remote=json.loads(cloud.remote(cloud.api('pods/'+train_job['pod_id']),f'cat /workspace/project/output/training/snapshots/iteration-{i:03d}/snapshot-manifest.json',timeout=20))
   if remote!=m:raise RuntimeError('Live source manifest differs')
  proof[str(i)]=m
 stamp=datetime.now(timezone.utc).strftime('%Y%m%dt%H%M%Sz').lower();name='shorterbetter-scaled-early-pair-'+stamp
 directory=ROOT/'lab/experiments/shorterbetter-full-evaluation'/stamp;directory.mkdir(parents=True)
 release=ROOT/'lab/isolated'/name;base=ROOT/'lab/isolated/full-evaluation-20261026/teacher-student-v2'
 for folder in ('experiments','ThinkingCap-Test'):
  shutil.copytree(base/folder,release/folder,copy_function=os.link,ignore=shutil.ignore_patterns('__pycache__','.git','.env*','.venv*','runs'))
 for f in ['experiments/setup_gpu.sh','experiments/gpu_policy.py','experiments/study_eval.py','experiments/baseline_cache.py','experiments/full_replay_eval.py','experiments/shorterbetter/full_eval.py','experiments/shorterbetter/paired_eval.py','experiments/shorterbetter/hardware.py']:
  target=release/f;target.parent.mkdir(parents=True,exist_ok=True)
  if target.exists():target.unlink()
  shutil.copy2(ROOT/f,target)
 data=release/'lab/experiments/data';stage=data/'early-pair';stage.mkdir(parents=True)
 shutil.copy2(base/'lab/experiments/data/pilot.jsonl',data/'pilot.jsonl');shutil.copy2(base/'lab/gpu-requirements.lock',release/'lab/gpu-requirements.lock')
 repo=release/'ThinkingCap-Test';files=[repo/'evaluate.py',repo/'compare.py']+sorted((repo/'utils/eval').rglob('*.py'));candidates=[]
 for i in (10,20):
  label=f'control-{i:03d}';source=source_root/f'iteration-{i:03d}';target=stage/label/'candidate';shutil.copytree(source,target,copy_function=os.link)
  original=read(source/'config.json');(target/'config.json').unlink();cloud.write(target/'config.json',dict(original,dtype='bfloat16'))
  config={'job_name':name,'evaluation_only':True,'training_authorized':False,'paired_short_evaluation':True,'samples':8,'evaluation_seed':9001,
   'candidates':['base','shorterbetter-final'],'model':study.BASE_MODEL,'revision':study.BASE_REVISION,'max_cost_usd':cap,
   'cohort_sha256':sha(repo/'questions/questions.jsonl'),'questions_manifest_sha256':sha(repo/'questions/manifest.json'),
   'evaluator_files_sha256':{str(p.relative_to(repo)):sha(p) for p in files},
   'candidate':{'path':'candidate','files_sha256':{p.name:sha(p) for p in target.iterdir() if p.is_file()},'original_weights_sha256':sha(source/'model.safetensors'),
   'optimizer_step':4*i,'original_model_config':original,'source_training_job':training,'alpha':2,
   'selection':'User-requested early10/20 comparison while training continues; both snapshots hash-verified against source worker.'}}
  cloud.write(stage/label/'evaluation.json',config);candidates.append({'label':label,'config':f'{label}/evaluation.json'})
 config={'job_name':name,'evaluation_only':True,'training_authorized':False,'early_control_pair':True,'candidates':candidates,'max_cost_usd':cap,
  'same_worker_baseline':True,'fresh_holdout':False,'campaign_path':str(campaign_path),'source_user_instruction':'2026-10-30: full evaluation of iteration10, then 20 iter checkpoint is also available run the full test suite; keep alpha2 to30 and alpha5 queued.'}
 cloud.write(stage/'bundle.json',config);cloud.write(directory/'bundle.json',config);cloud.write(directory/'snapshot-verification.json',proof)
 with patch.object(study,'REPO',repo):bundle(stage/'bundle.json')
 entry=release/'experiments/shorterbetter/run-scaled.sh'
 if entry.exists():entry.unlink()
 entry.write_text('''#!/usr/bin/env bash
set -euo pipefail
source /workspace/project/experiments/setup_gpu.sh
export NLTK_DATA=/root/thinkingcap_nltk_data
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m experiments.shorterbetter.paired_eval --config data/early-pair/bundle.json --out output/paired-evaluation
''')
 payload=seal(release)
 gpu='NVIDIA GeForce RTX 4090';hours=10;rate=.6;disk=60
 query={'verified':{'eq':True},'rentable':{'eq':True},'rented':{'eq':False},'gpu_name':{'eq':'RTX 4090'},'num_gpus':{'eq':1},'gpu_ram':{'gte':24000},'cuda_max_good':{'gte':12.8},'disk_space':{'gte':disk},'direct_port_count':{'gte':1},'reliability2':{'gte':.99},'type':'on-demand','order':[['dph_total','asc']],'limit':1000}
 offers=vastai.api('v0/bundles/',params={'q':json.dumps(query)})['offers'];quote=None
 offers=[r for r in offers if float(r['dph_total'])+disk*float(r['storage_cost'])/720<=rate and float(r.get('inet_down',0))>=600 and float(r.get('direct_port_count',0))>0 and float(r.get('duration',0))>=(hours+.5)*3600 and not cloud.unhealthy_placement(ledger,gpu,'vast',{'offer':r,'offer_id':r['id']})]
 offers.sort(key=lambda r:(r['machine_id'] not in (38546,38674),-float(r.get('reliability2',0)),float(r['dph_total'])))
 for r in offers[:15]:
  q=vastai.quote(gpu,1,disk,r['id'],(hours+.5)*3600)
  if q and q['hourly_ceiling']<=rate:quote=q;break
 if not quote:raise RuntimeError('No eligible direct-SSH offer; sealed payload preserved')
 launch=dict(job_name=name,release=str(release),payload_sha256=payload,provider='vast',offer_id=quote['offer_id'],disk_gb=disk,gpu=gpu,gpu_count=1,
  phase='evaluation',hours=1,max_runtime_hours=hours,max_cost_usd=cap,max_gpu_rate=rate,evaluation_only=True,full_training_authorized=False,telegram_authorized=True,direct_ssh=True,
  paired_evaluation_account_guard=True,paired_campaign_path=str(campaign_path))
 cloud.write(directory/'launch.json',launch);cloud.write(directory/'quote.json',quote)
 cloud.write(cloud.STATE/'authorizations'/(name+'.json'),dict(launch,authorized=True,scope='shorterbetter_vast_bounded_v1',source_user_instruction=config['source_user_instruction']))
 from experiments.shorterbetter.budget import ticket
 if not ticket(cloud.STATE,name,dict(launch,entry='experiments/shorterbetter/run-scaled.sh',hardware_scope='shorterbetter-scaled-v1')):raise RuntimeError('Budget ticket invalid')
 c['early_evaluations'].append(dict(job_name=name,iteration=20,iterations=[10,20],alpha=2,early_control_pair=True,cap_usd=cap,directory=str(directory),state='prepared'))
 cloud.write(campaign_path,c);cloud.write(directory.parent/'current.json',{'directory':str(directory),'job_name':name,'service':'thinkingcap-shorterbetter-early-pair-'+stamp,'status':'sealed_not_launched','evaluation_only':True})
 return directory,launch

def notify(directory,launch):
 from ops.notify_shorterbetter_scaled import send_html
 sys.path.insert(0,'/srv/playground');from project_alerts import Alerts
 from zoneinfo import ZoneInfo
 name=launch['job_name'];j=jobs()[name]
 if not j.get('released_at') or any(p['id']==j['pod_id'] for p in vastai.list_pods()):raise RuntimeError('GPU not released')
 out=cloud.STATE/name/'output/paired-evaluation';complete=bool(j.get('backup_verified') and read(out/'completion.json').get('complete')) if (out/'completion.json').exists() else False
 if j.get('backup_verified'):cloud.verify_backup(name)
 lines=['<b>Project: ThinkingCap / ShorterBetter</b>','Experiment: α=2',
  'Summary: '+('Plná evaluace iterací10 a20 dokončena.' if complete else 'Plná evaluace iterací10/20 není kompletní; zachované výsledky v UI.'),
  'Evaluace: 2 759 otázek × 8 odpovědí na model','GPU: 1× RTX4090 (vast)','GPU uvolněna: ano',f"Záloha ověřena: {'ano' if j.get('backup_verified') else 'ne'}"]
 for i in (10,20):
  path=out/'comparisons'/f'control-{i:03d}.json'
  if not path.exists():continue
  r=read(path);lines+=['',f'<b>Iterace {i}</b>',f"Accuracy: {r['average']['accuracy_delta']*100:+.2f} p. b.",f"Úspora tokenů: {r['average']['token_reduction']*100:.1f} %"]
  for domain,label in [('svamp','Matematika'),('mbpp','Kódování'),('ifeval','Instrukce'),('bfcl','Nástroje')]:
   parts=[v for k,v in r['results'].items() if k==domain or (domain=='bfcl' and k.startswith('bfcl_'))];n=sum(v['paired_rows'] for v in parts)
   def value(side,key):return sum(v['paired_rows']*v[side][key] for v in parts)/n
   lines += [label,f"Accuracy: {value('base','accuracy'):.1%} → {value('candidate','accuracy'):.1%}",f"Tokeny / odpověď: {value('base','mean_output_tokens'):.0f} → {value('candidate','mean_output_tokens'):.0f}"]
 elapsed=int(j['released_at']-j['created_at']);h,rest=divmod(elapsed,3600);m,s=divmod(rest,60)
 lines+=['',f'Time elapsed: {h:02d}:{m:02d}:{s:02d}',f'Timestamp: {datetime.now(ZoneInfo("Europe/Prague")):%Y-%m-%d %H:%M:%S %Z} (Praha)']
 alerts=Alerts(directory/'notifications.sqlite3','ThinkingCap / ShorterBetter',sender=send_html,formatter=lambda *args:'\n'.join(lines))
 if not alerts.db.execute('SELECT 1 FROM incidents WHERE key=?',('final',)).fetchone():alerts.emit('final','Results' if complete else 'Technical','Early10/20 evaluator finished')
 cloud.write(directory/'completion-status.json',{'success':complete,'released':True,'backup_verified':bool(j.get('backup_verified'))})
 pointer=directory.parent/'current.json';p=read(pointer)
 if p.get('job_name')==name:p['status']='completed_released' if complete else 'failed_released';cloud.write(pointer,p)

def main():
 os.umask(0o077);p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);p.add_argument('--directory',type=Path);a=p.parse_args()
 if a.action=='prepare':
  d,l=prepare();print(json.dumps({'directory':str(d),'job_name':l['job_name'],'cap_usd':l['max_cost_usd'],'service':'thinkingcap-shorterbetter-early-pair-'+d.name}));return
 d=a.directory.resolve();l=read(d/'launch.json');name=l['job_name']
 if verify(Path(l['release']))!=l['payload_sha256']:raise RuntimeError('Immutable payload changed')
 if name in jobs():
  if not jobs()[name].get('released_at'):cloud.watch(name)
 else:
  command=[sys.executable,str(ROOT/'ops/isolated_cloud.py'),'run','--release',l['release'],'--name',name,'--entry','experiments/shorterbetter/run-scaled.sh','--gpu',l['gpu'],'--gpu-count','1','--hardware-scope','shorterbetter-scaled-v1','--hours','1','--max-cost-usd',str(l['max_cost_usd']),'--max-runtime-hours',str(l['max_runtime_hours']),'--provider','vast','--vast-offer-id',str(l['offer_id']),'--vast-disk-gb',str(l['disk_gb']),'--data-upload-timeout','900']
  subprocess.run(command,check=False)
 notify(d,l)
if __name__=='__main__':main()
