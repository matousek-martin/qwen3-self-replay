"""Register the authorized all-answer overnight comparison without duplicate GPUs."""
import json,math,shutil,time,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments import cloud
from experiments.opencoder_sft.runtime import read,write,sha
from experiments.coding_abc.budget import GPU,SCOPE,PREFIX,ENTRY,ARMS,ticket
from experiments.coding_abc.prepare_expanded import prepare as data_prepare
from experiments.coding_abc.logic import check_contract
from experiments.opencoder_sft.runtime import rows
from ops import vastai
from ops.isolated_cloud import seal

def prepare():
 oldreg=Path(read(ROOT/'lab/experiments/coding-abc/current.json')['directory'])
 if read(oldreg/'common-plan.json').get('variant')=='expanded-v2':raise RuntimeError('Expanded comparison already registered')
 if vastai.list_pods():raise RuntimeError('Wait for pilot backup/release')
 credit=float(vastai.api('v0/users/current/')['credit'])
 if credit<40:raise RuntimeError('Insufficient protected36USD plus4reserve')
 stamp=time.strftime('%Y%m%dt%H%M%Sz',time.gmtime()).lower();reg=ROOT/'lab/experiments/coding-abc'/stamp;reg.mkdir()
 cfg=data_prepare(reg/'data');write(reg/'common-plan.json',cfg)
 remaining_hours=(cfg['overall_deadline']-time.time())/3600
 if not 2<remaining_hours<=18:raise RuntimeError('Insufficient overnight window')
 jobs=read(cloud.STATE/'ledger.json')['jobs'];instruction='2026-11-04: User requests all436ownanswers forB, onlineA/C, results bynextmorning, accepts3RTX4090and16384cap, explicitly sayslaunchwhatcanrun.'
 campaign=dict(authorized=True,state='prepared',cap_usd=36,credit_at_prepare=credit,created_at=time.time(),arms={},source_user_instruction=instruction,
  variant='expanded-v2',prior_campaign=str(oldreg),coding_answers=436,coding_questions=cfg['coding_questions'],target_steps=cfg['steps'],main_scale_question_pending=False)
 for arm in ARMS:
  name=PREFIX+arm+'-expanded-'+stamp;d=reg/arm;d.mkdir();release=ROOT/'lab/isolated'/name
  previous=[n for n in jobs if n.startswith(PREFIX+arm+'-')]
  if any(not jobs[n].get('released_at') for n in previous):raise RuntimeError('Unreleased predecessor')
  spent=sum(jobs[n]['charged_estimate_usd'] for n in previous);cap=round(12-spent,4)
  if cap<3:raise RuntimeError('Insufficient arm budget')
  shutil.copytree(ROOT/'experiments',release/'experiments',ignore=shutil.ignore_patterns('__pycache__','.git','.env*','.venv*','*.log'))
  frozen=ROOT/'lab/isolated/full-evaluation-20261026/teacher-student-v2'
  shutil.copytree(frozen/'ThinkingCap-Test',release/'ThinkingCap-Test',ignore=shutil.ignore_patterns('__pycache__','.git','.env*','.venv*','runs','*.log'))
  stage=release/'lab/experiments/data/coding-abc';shutil.copytree(reg/'data',stage);plan=dict(cfg,arm=arm,scope=SCOPE)
  check_contract(plan,rows(stage/'train.jsonl'),rows(stage/'development.jsonl'));write(stage/'plan.json',plan);write(d/'plan.json',plan)
  ev=release/'lab/experiments/data/replay-evaluation';ev.mkdir();src=ROOT/'lab/experiments/replay-scaled-evaluation/20261102t102548z'
  shutil.copy2(src/'model-preload-source-verification.json',ev/'model-preload.json');shutil.copy2(src/'uv-wheel-source.json',ev/'uv-wheel-source.json')
  shutil.copy2(ROOT/'lab/experiments/data/study/splits.json',ev/'splits.json')
  (release/'lab/gpu-requirements.lock').write_text((ROOT/'lab/gpu-requirements.lock').read_text()+'\npytest==8.3.5\niniconfig==2.3.1\npluggy==1.6.0\n')
  digest=seal(release)
  from experiments.baseline_cache import check_launch
  check_launch(release,ENTRY,ROOT/'lab/experiments/baseline-cache',GPU)
  launch=dict(job_name=name,release=str(release),payload_sha256=digest,authorized=True,scope=SCOPE,hardware_scope=SCOPE,entry=ENTRY,
   provider='vast',gpu=GPU,gpu_count=1,arm=arm,parent_group=768,offer_id=None,disk_gb=80,max_cost_usd=cap,max_runtime_hours=round(remaining_hours-.02,4),
   max_gpu_rate=.6,training_authorized=True,direct_ssh=True,campaign_path=str(reg/'campaign.json'),combined_cap_usd=36,
   source_user_instruction=instruction,previous_attempts=previous,variant='expanded-v2',final_official_evaluation=True)
  write(d/'launch.json',launch);write(d/'status.json',dict(state='prepared',job_name=name));campaign['arms'][arm]=dict(job_name=name,directory=str(d),max_cost_usd=12,previous_attempts=previous)
  write(cloud.STATE/'authorizations'/(name+'.json'),launch)
 write(reg/'campaign.json',campaign)
 for arm in ARMS:
  l=read(reg/arm/'launch.json')
  if not ticket(cloud.STATE,l['job_name'],l):raise RuntimeError('Invalid bounded arm ticket')
 write(ROOT/'lab/experiments/coding-abc/current.json',dict(directory=str(reg)))
 print(json.dumps(dict(directory=str(reg),coding_questions=cfg['coding_questions'],coding_answers=436,retention_questions=cfg['retention_questions'],steps=cfg['steps'],credit=credit)))
 return reg
if __name__=='__main__':prepare()
