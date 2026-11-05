"""Bounded automatic full evaluation after both paired training archives exist."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from unittest.mock import patch
from ops.shorterbetter_paired import ROOT,read,sha,jobs,remaining,complete,choose_quote
from experiments import cloud,study_eval as study
from experiments.shorterbetter.paired_eval import bundle
from ops.isolated_cloud import seal,verify

def prepare(directory):
 c=read(directory/'campaign.json');reg=directory/'evaluation';reg.mkdir(exist_ok=True)
 if (reg/'launch.json').exists():return read(reg/'launch.json')
 for arm in c['arms'].values():complete(arm['job_name'])
 cap=round(min(15.,remaining(c)),4)
 if cap<9:raise RuntimeError('Insufficient joint budget for the registered snapshot evaluation')
 name='shorterbetter-scaled-paired-eval-'+c['id'];gpu='NVIDIA GeForce RTX 4090';hours=17
 q=choose_quote(gpu,1,80,hours,.6)
 release=ROOT/'lab/isolated'/name
 base=ROOT/'lab/isolated/full-evaluation-20261026/teacher-student-v2'
 if release.exists():raise RuntimeError('Partial evaluation payload requires inspection')
 for folder in ('experiments','ThinkingCap-Test'):
  shutil.copytree(base/folder,release/folder,copy_function=os.link,ignore=shutil.ignore_patterns('__pycache__','.git','.env*','.venv*','runs'))
 for f in ['experiments/setup_gpu.sh','experiments/gpu_policy.py','experiments/study_eval.py','experiments/baseline_cache.py','experiments/full_replay_eval.py','experiments/shorterbetter/full_eval.py','experiments/shorterbetter/paired_eval.py','experiments/shorterbetter/hardware.py']:
  target=release/f;target.parent.mkdir(parents=True,exist_ok=True)
  if target.exists():target.unlink()
  shutil.copy2(ROOT/f,target)
 data=release/'lab/experiments/data';stage=data/'paired-evaluation';stage.mkdir(parents=True)
 shutil.copy2(base/'lab/experiments/data/pilot.jsonl',data/'pilot.jsonl')
 shutil.copy2(base/'lab/gpu-requirements.lock',release/'lab/gpu-requirements.lock')
 repo=release/'ThinkingCap-Test';evaluator=[repo/'evaluate.py',repo/'compare.py']+sorted((repo/'utils/eval').rglob('*.py'))
 candidates=[]
 for arm in ('control','accuracy'):
  training=c['arms'][arm]['job_name']
  for i in range(5,31,5):
   label=f'{arm}-{i:03d}';source=cloud.STATE/training/'output/training/snapshots'/f'iteration-{i:03d}'
   target=stage/label/'candidate';shutil.copytree(source,target,copy_function=os.link)
   original=read(source/'config.json');(target/'config.json').unlink();cloud.write(target/'config.json',dict(original,dtype='bfloat16'))
   config={'job_name':name,'evaluation_only':True,'training_authorized':False,'paired_short_evaluation':True,'samples':8,'evaluation_seed':9001,
    'candidates':['base','shorterbetter-final'],'model':study.BASE_MODEL,'revision':study.BASE_REVISION,'max_cost_usd':cap,
    'cohort_sha256':sha(repo/'questions/questions.jsonl'),'questions_manifest_sha256':sha(repo/'questions/manifest.json'),
    'evaluator_files_sha256':{str(p.relative_to(repo)):sha(p) for p in evaluator},
    'candidate':{'path':'candidate','files_sha256':{p.name:sha(p) for p in target.iterdir() if p.is_file()},
     'original_weights_sha256':sha(source/'model.safetensors'),'optimizer_step':4*i,'original_model_config':original,
     'source_training_job':training,'alpha':c['arms'][arm]['alpha'],'selection':'All registered snapshots every5rollouts; exploratory comparison on repeatedly inspected benchmarks.'}}
   cloud.write(stage/label/'evaluation.json',config);candidates.append({'label':label,'config':f'{label}/evaluation.json'})
 config={'job_name':name,'evaluation_only':True,'training_authorized':False,'candidates':candidates,'max_cost_usd':cap,
         'same_worker_baseline':False,'baseline_policy':'gpu-model-and-protocol-cache-v1','fresh_holdout':False,'source_user_instruction':c['source_user_instruction']}
 cloud.write(stage/'bundle.json',config);cloud.write(reg/'bundle.json',config)
 with patch.object(study,'REPO',repo):bundle(stage/'bundle.json')
 entry=release/'experiments/shorterbetter/run-scaled.sh'
 if entry.exists():entry.unlink()
 entry.write_text('''#!/usr/bin/env bash
set -euo pipefail
source /workspace/project/experiments/setup_gpu.sh
export NLTK_DATA=/root/thinkingcap_nltk_data
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m experiments.shorterbetter.paired_eval --config data/paired-evaluation/bundle.json --out output/paired-evaluation
''')
 payload=seal(release)
 launch=dict(job_name=name,release=str(release),payload_sha256=payload,provider='vast',offer_id=q['offer_id'],disk_gb=80,
  gpu=gpu,gpu_count=1,phase='evaluation',hours=1,max_runtime_hours=hours,max_cost_usd=cap,max_gpu_rate=.6,evaluation_only=True,
  full_training_authorized=False,telegram_authorized=True)
 cloud.write(reg/'launch.json',launch);cloud.write(reg/'quote.json',q)
 cloud.write(cloud.STATE/'authorizations'/(name+'.json'),dict(launch,authorized=True,scope='shorterbetter_vast_bounded_v1',source_user_instruction=c['source_user_instruction']))
 c['evaluation']={'job_name':name,'state':'prepared','cap_usd':cap,'snapshots':12};cloud.write(directory/'campaign.json',c)
 return launch

def notify(directory,success,count):
 from ops.notify_shorterbetter_scaled import send_html
 sys.path.insert(0,'/srv/playground')
 from project_alerts import Alerts
 from datetime import datetime
 from zoneinfo import ZoneInfo
 text='\n'.join(['<b>Project: ThinkingCap / ShorterBetter</b>',
  'Summary: '+('Oba experimenty a plná evaluace checkpointů dokončeny.' if success else 'Evaluace checkpointů není úplná; dostupné výsledky jsou v UI.'),
  f'Vyhodnocené checkpointy: {count} / 12','Evaluace: 2 759 otázek × 8 odpovědí na model; společný čerstvý baseline',
  'GPU uvolněna: ano','Záloha: ověřena',f'Timestamp: {datetime.now(ZoneInfo("Europe/Prague")):%Y-%m-%d %H:%M:%S %Z} (Praha)'])
 alerts=Alerts(directory/'notifications.sqlite3','ThinkingCap / ShorterBetter',sender=send_html,formatter=lambda *args:text)
 key='paired-evaluation-final'
 if not alerts.db.execute('SELECT 1 FROM incidents WHERE key=?',(key,)).fetchone():alerts.emit(key,'Results','Paired evaluation finished')

def run(directory):
 l=prepare(directory);name=l['job_name'];j=jobs().get(name)
 if verify(Path(l['release']))!=l['payload_sha256']:raise RuntimeError('Evaluation payload changed')
 if j:
  if not j.get('released_at'):cloud.watch(name)
 else:
  from ops import vastai
  if vastai.list_pods():raise RuntimeError('Existing Vast instance before evaluation')
  command=[sys.executable,str(ROOT/'ops/isolated_cloud.py'),'run','--release',l['release'],'--name',name,'--entry','experiments/shorterbetter/run-scaled.sh',
   '--gpu',l['gpu'],'--gpu-count','1','--hardware-scope','shorterbetter-scaled-v1','--hours','1','--max-cost-usd',str(l['max_cost_usd']),
   '--max-runtime-hours',str(l['max_runtime_hours']),'--provider','vast','--vast-offer-id',str(l['offer_id']),'--vast-disk-gb','80','--data-upload-timeout','900']
  subprocess.run(command,check=False)
 j=jobs()[name]
 if not j.get('released_at') or not j.get('backup_verified'):raise RuntimeError('Evaluation archive/release incomplete')
 cloud.verify_backup(name);out=cloud.STATE/name/'output/paired-evaluation';count=len(list((out/'comparisons').glob('*.json')))
 success=(out/'completion.json').is_file() and count==12
 c=read(directory/'campaign.json');c['state']='completed' if success else 'evaluation_incomplete';c['evaluation'].update(state=c['state'],comparisons=count,released=True)
 cloud.write(directory/'campaign.json',c);notify(directory,success,count)
