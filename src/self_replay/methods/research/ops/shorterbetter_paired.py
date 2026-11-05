"""Two registered sequential 30-rollout runs, bounded jointly with evaluation."""
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments import cloud
from ops import vastai
from ops.isolated_cloud import seal,verify
from experiments.shorterbetter.multidomain.full_run import validate,technical_protocol
from experiments.shorterbetter.config import training_arguments
BASE=ROOT/'lab/isolated/shorterbetter-scaled-full-vast-a100x4-20261029t115451z'
CAMPAIGNS=ROOT/'lab/experiments/shorterbetter-paired'
GPU='NVIDIA A100-SXM4-80GB'
INSTRUCTION='2026-10-30: i topped up 100 pls run both; add both experiments and polling live training graphs to UI. Two30-rollout seed17 runs alpha2/alpha5; weights every5; quality diagnostics monitoring-only.'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 with Path(p).open('rb') as s:return hashlib.file_digest(s,'sha256').hexdigest()
def jobs():return read(cloud.STATE/'ledger.json')['jobs']
def remaining(c):
 ledger=jobs()
 training=sum(ledger.get(a['job_name'],{}).get('charged_estimate_usd',0) for a in c['arms'].values())
 early=sum(ledger.get(a['job_name'],{}).get('charged_estimate_usd',0) if ledger.get(a['job_name'],{}).get('released_at')
           else a['cap_usd'] for a in c.get('early_evaluations',[]))
 return c['cap_usd']-training-early

def check_other_instances(launch):
 c=read(launch['paired_campaign_path'])
 names={e['job_name'] for e in c.get('early_evaluations',[])}
 allowed={j['pod_id'] for name,j in jobs().items() if name in names and j.get('pod_id') and not j.get('released_at')}
 if any(p['id'] not in allowed for p in vastai.list_pods()):
  raise RuntimeError('An unregistered existing Vast instance requires reconciliation')
def choose_quote(gpu=GPU,count=4,disk=80,hours=10,max_rate=1.1):
 q=vastai.quote(gpu,count,disk,min_duration=(hours+.5)*3600)
 if q and q['hourly_ceiling']/count<=max_rate and not cloud.unhealthy_placement(jobs(),gpu,'vast',q):return q
 query={'verified':{'eq':True},'rentable':{'eq':True},'rented':{'eq':False},'gpu_name':{'eq':vastai.GPU_NAMES[gpu]},
   'num_gpus':{'eq':count},'gpu_ram':{'gte':24000 if '4090' in gpu else 80000},'cuda_max_good':{'gte':12.8},
   'disk_space':{'gte':disk},'reliability2':{'gte':.99},'type':'on-demand','order':[['dph_total','asc']],'limit':1000}
 rows=vastai.api('v0/bundles/',params={'q':json.dumps(query)})['offers']
 for row in rows:
  if (float(row.get('duration',0))<(hours+.5)*3600 or float(row.get('inet_down',0))<300
      or (float(row['dph_total'])+disk*float(row['storage_cost'])/720)/count>max_rate
      or cloud.unhealthy_placement(jobs(),gpu,'vast',{'offer':row,'offer_id':row['id']})):continue
  q=vastai.quote(gpu,count,disk,row['id'],(hours+.5)*3600)
  if q and q['hourly_ceiling']/count<=max_rate:return q
 raise RuntimeError('No eligible offer within the registered rate; no rental created')

def prepare(directory,arm,cap):
 c=read(directory/'campaign.json');a=c['arms'][arm];name=a['job_name'];reg=directory/arm;reg.mkdir(exist_ok=True)
 if (reg/'launch.json').exists():return read(reg/'launch.json')
 base=BASE if arm=='control' else Path(read(directory/'control/launch.json')['release'])
 release=ROOT/'lab/isolated'/name
 if release.exists():raise RuntimeError('Unsealed partial payload requires inspection')
 p=read(base/'experiments/shorterbetter/launch/protocol.json')
 p.update(id=name,question='Early shortening versus stronger correctness reward on unchanged four-domain data.',scope='One of two matched30-rollout experiments; same base/data/seed; only alpha differs.')
 p['execution'].update(paired_short_experiment=True,preserve_every_rollouts=5,save_total_limit=1)
 p['training'].update(rollout_batches=30,optimizer_updates_total=120,training_completions_per_arm=15360)
 p['arms']['shorterbetter']['alpha']=p['reward']['alpha']=a['alpha']
 p['stopping']['quality_stopping_enabled']=False
 p['authorization'].update(source_user_instruction=INSTRUCTION,quality_stopping_disabled_by_user=True,full_training_authorized=True)
 p['budget'].update(proposed_total_pilot_cap_usd=cap,max_runtime_hours=10,scope='shorterbetter_vast_bounded_v1')
 p['full_run_target']={'launch_authorized':True,'rollout_batches':30,'optimizer_updates':120}
 p['selection']={'rule':'Preserve every5-rollout inference snapshot; evaluate independently. No quality-based stop or reward adaptation.'}
 p['phases']=['Isolated two-rollout same-worker resume and precision checks','Fresh pinned baseline;30 training rollouts; fixed validation every10','SHA verified snapshots and complete final optimizer/RNG checkpoint, release']
 p['lineage']={'base_job':BASE.name,'campaign':c['id'],'arm':arm,'same_data_and_seed':True}
 validate(p);technical_protocol(p)
 assert training_arguments(p,'shorterbetter',17,'output/training')['max_steps']==120
 cloud.write(reg/'protocol.json',p)
 shutil.copytree(base,release,copy_function=os.link,ignore=shutil.ignore_patterns('payload-manifest.json','__pycache__'))
 if arm=='control':
  files=[f for f in (ROOT/'experiments/shorterbetter').rglob('*.py') if not any(part.startswith('.') for part in f.relative_to(ROOT).parts)]+[ROOT/'experiments/cloud.py']
  for src in files:
   dest=release/src.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True)
   if dest.exists():dest.unlink()
   shutil.copy2(src,dest)
 proof={'status':'passed','test_evidence':str(c['test_evidence']),'gpu_checks':'pending_on_this_worker',
        'source_sha256':{str(f.relative_to(release)):sha(f) for f in (release/'experiments/shorterbetter').rglob('*.py')}}
 cloud.write(reg/'cpu-verification.json',proof)
 ready=read(base/'experiments/shorterbetter/launch/readiness.json')
 ready.update(protocol_sha256=sha(reg/'protocol.json'),cpu_verification_path=str((reg/'cpu-verification.json').relative_to(ROOT)),cpu_verification_sha256=sha(reg/'cpu-verification.json'))
 for key in ('budget_reservation','pilot_cap_enforced','trainer_gpu_smoke','distributed_gpu_check','resume_equivalence','matched_validation','export_reload','cost_projection'):ready[key]=False
 cloud.write(reg/'readiness.json',ready)
 for f in ('protocol.json','readiness.json','cpu-verification.json'):
  dest=release/'experiments/shorterbetter/launch'/f;dest.unlink();shutil.copy2(reg/f,dest)
 payload=seal(release);q=choose_quote();cloud.write(reg/'quote.json',q)
 launch=dict(job_name=name,release=str(release),payload_sha256=payload,provider='vast',offer_id=q['offer_id'],disk_gb=80,
             gpu=GPU,gpu_count=4,phase='training',hours=1,max_runtime_hours=10,max_cost_usd=cap,max_gpu_rate=1.1,
             rollout_batches=30,full_training_authorized=True,telegram_authorized=True,paired_short_experiment=True,
             paired_campaign_path=str(directory/'campaign.json'))
 cloud.write(reg/'launch.json',launch)
 cloud.write(cloud.STATE/'authorizations'/(name+'.json'),dict(launch,authorized=True,scope='shorterbetter_vast_bounded_v1',source_user_instruction=INSTRUCTION))
 registry=ROOT/'lab/experiments/shorterbetter-scaled/notifications/jobs.json';r=read(registry)
 r[name]={'telegram_authorized':True,'directory':str(reg)};cloud.write(registry,r)
 return launch

def complete(name):
 j=jobs().get(name,{})
 if not j.get('released_at') or not j.get('backup_verified'):raise RuntimeError('Previous worker not archived and released')
 cloud.verify_backup(name)
 out=cloud.STATE/name/'output/training'
 if read(out/'training-finished.json')['optimizer_steps']!=120:raise RuntimeError('Training ended before30rollouts')
 for i in range(5,31,5):
  folder=out/'snapshots'/f'iteration-{i:03d}';m=read(folder/'snapshot-manifest.json')
  if m['optimizer_step']!=4*i:raise RuntimeError('Wrong checkpoint iteration')
  for f,h in m['files_sha256'].items():
   if sha(folder/f)!=h:raise RuntimeError('Snapshot checksum mismatch')
 # Export is tensor-identical, but safetensors bytes can differ; link only exact matches.
 model=out/'model/model.safetensors';last=out/'snapshots/iteration-030/model.safetensors'
 if model.is_file() and model.stat().st_ino!=last.stat().st_ino and sha(model)==sha(last):
  tmp=model.with_suffix('.dedup');os.link(last,tmp);os.replace(tmp,model)

def launch_training(l):
 if verify(Path(l['release']))!=l['payload_sha256']:raise RuntimeError('Sealed payload changed')
 name=l['job_name'];j=jobs().get(name)
 if j:
  if not j.get('released_at'):cloud.watch(name)
 else:
  check_other_instances(l)
  if shutil.disk_usage(ROOT).free<21*2**30:raise RuntimeError('Insufficient local archive space')
  command=[sys.executable,str(ROOT/'ops/isolated_cloud.py'),'run','--release',l['release'],'--name',name,
    '--entry','experiments/shorterbetter/run-scaled.sh','--gpu',l['gpu'],'--gpu-count','4','--hardware-scope','shorterbetter-scaled-v1',
    '--hours','1','--max-cost-usd',str(l['max_cost_usd']),'--max-runtime-hours','10','--provider','vast',
    '--vast-offer-id',str(l['offer_id']),'--vast-disk-gb','80','--data-upload-timeout','900','--direct-pinned-wheels']
  subprocess.run(command,check=False)
 complete(name)

def run(directory):
 with (directory/'controller.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  for arm in ('control','accuracy'):
   c=read(directory/'campaign.json');name=c['arms'][arm]['job_name']
   if jobs().get(name,{}).get('released_at'):
    complete(name);continue
   cap=round(min(55.,remaining(c)-c['evaluation_reserve_usd']),4)
   if cap<35:raise RuntimeError('Insufficient remaining campaign budget for the next training run')
   c['state']='running_'+arm;c['arms'][arm]['state']='running';c['arms'][arm]['cap_usd']=cap;cloud.write(directory/'campaign.json',c)
   l=prepare(directory,arm,cap);launch_training(l)
   c=read(directory/'campaign.json');c['arms'][arm]['state']='completed_released';cloud.write(directory/'campaign.json',c)
  c=read(directory/'campaign.json');c['state']='training_complete_evaluation_pending';cloud.write(directory/'campaign.json',c)
  from ops.shorterbetter_paired_evaluation import run as evaluate
  evaluate(directory)

def main():
 os.umask(0o077);parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run']);parser.add_argument('--directory',type=Path);a=parser.parse_args()
 if a.action=='prepare':
  if vastai.list_pods():raise RuntimeError('Existing Vast instance')
  account=vastai.api('v0/users/current/')
  if account['credit']<105:raise RuntimeError('New campaign credit not funded')
  stamp=datetime.now(timezone.utc).strftime('%Y%m%dt%H%M%Sz').lower();directory=CAMPAIGNS/stamp;directory.mkdir(parents=True)
  c={'id':stamp,'authorized':True,'source_user_instruction':INSTRUCTION,'cap_usd':105.,'evaluation_reserve_usd':10.,'initial_credit_usd':account['credit'],
     'state':'prepared','test_evidence':'lab/experiments/shorterbetter-paired/cpu-tests.txt',
     'arms':{arm:{'job_name':f'shorterbetter-scaled-paired-{arm}-{stamp}','alpha':alpha,'state':'queued'} for arm,alpha in [('control',2.),('accuracy',5.)]}}
  cloud.write(directory/'campaign.json',c);cloud.write(CAMPAIGNS/'current.json',{'path':str((directory/'campaign.json').relative_to(ROOT))})
  prepare(directory,'control',55.);print(json.dumps({'directory':str(directory),'service':'thinkingcap-shorterbetter-paired-'+stamp}))
 else:
  try:run(a.directory.resolve())
  except Exception as exc:
   c=read(a.directory/'campaign.json');c.update(state='needs_inspection',error=str(exc),error_at=time.time());cloud.write(a.directory/'campaign.json',c)
   raise
if __name__=='__main__':main()
