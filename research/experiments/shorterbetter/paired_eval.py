"""Evaluate twelve preserved early snapshots against one fresh same-worker baseline."""
import argparse
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
from experiments import study_eval as study
from .full_eval import inputs,digest,tokenizer_parity
from .hardware import verify_worker

def bundle(path):
 path=Path(path).resolve();config=json.loads(path.read_text())
 expected=[f'{arm}-{i:03d}' for arm in ('control','accuracy') for i in range(5,31,5)]
 if config.get('early_control_pair') is True:
  expected=['control-010','control-020']
 if config.get('parallel_snapshot_shard') is True:
  registered=[f'{arm}-{i:03d}' for arm in ('control','accuracy') for i in range(5,31,5)]
  expected=config.get('registered_labels',[])
  if not expected or len(expected)>3 or len(expected)!=len(set(expected)) or any(x not in registered for x in expected):
   raise ValueError('Invalid parallel snapshot shard')
 if (config.get('evaluation_only') is not True or config.get('training_authorized') is not False
     or [c['label'] for c in config['candidates']]!=expected):raise ValueError('Unregistered paired snapshot evaluation')
 result=[]
 for row in config['candidates']:
  p=study.inside(path.parent,row['config']);c,source,questions,model=inputs(p)
  arm,iteration=row['label'].split('-')
  if (not c.get('paired_short_evaluation') or c['candidate']['optimizer_step']!=4*int(iteration)
      or not c['candidate']['source_training_job'].startswith('shorterbetter-scaled-paired-'+arm+'-')
      or c['job_name']!=config['job_name'] or c['max_cost_usd']!=config['max_cost_usd']):
   raise ValueError('Candidate registration differs from bundle')
  result.append((row['label'],c,source,questions,model,p))
 return config,result

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);a=parser.parse_args()
 signal.signal(signal.SIGTERM,study.interrupt)
 config,candidates=bundle(a.config);lease=json.loads((study.ROOT/'worker-lease.json').read_text())
 if lease['job_name']!=config['job_name'] or lease['max_cost_usd']!=config['max_cost_usd']:raise ValueError('Wrong evaluation lease')
 verify_worker(subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True).splitlines(),lease)
 gpu=['nvidia-smi','--query-gpu=uuid,name,driver_version','--format=csv,noheader'];identity=subprocess.check_output(gpu,text=True).strip()
 out=a.out.resolve()
 if out.exists():raise ValueError('No implicit restart into previous evaluation output')
 out.mkdir(parents=True)
 study.write_json(out/'provenance.json',dict(config,gpu_identity=identity,lease=lease,matched_baseline_regenerated=None,same_physical_worker_required=False,fresh_holdout=False,started_at=study.now()))
 _,first,source,rows,_,_=candidates[0];study.preflight_nltk()
 for label,c,_,_,model,_ in candidates:tokenizer_parity(rows,model,out/'checks'/label)
 study.write_json(out/'phase.json',{'phase':'grader_smoke','updated_at':time.time()})
 study.run_process([sys.executable,study.REPO/'evaluate.py','--questions',source,'--out',out/'grader-smoke','--smoke','--workers',4],out/'logs/grader-smoke.log',None)
 if json.loads((out/'grader-smoke/results.json').read_text()).get('failed_benchmarks'):raise ValueError('Official grader smoke failed')
 sequence=[('base',first,source,rows,None,None),*candidates]
 if config.get('baseline_loading') == 'pinned_local':
  from huggingface_hub import snapshot_download
  baseline=Path(snapshot_download(study.BASE_MODEL,revision=study.BASE_REVISION,
      allow_patterns=['*.json','*.safetensors','*.txt','*.jinja'])).resolve()
  study.write_json(out/'baseline-loading.json',{'model':study.BASE_MODEL,'revision':study.BASE_REVISION,
      'path':str(baseline),'loading':'local_path_like_candidates',
      'files_sha256':{p.name:digest(p) for p in baseline.iterdir() if p.is_file()}})
  sequence[0]=('base',first,source,rows,baseline,None)
 for label,c,source,rows,model,path in sequence:
  if subprocess.check_output(gpu,text=True).strip()!=identity:raise RuntimeError('Physical GPU changed')
  if path:inputs(path)
  if time.time()>=lease['hard_deadline']-900:raise RuntimeError('Evaluation budget/runtime reached; preserving completed comparisons')
  study.write_json(out/'phase.json',{'phase':label,'updated_at':time.time()})
  run=out/'runs'/label;command=[sys.executable,study.REPO/'evaluate.py','--questions',source,'--out',run,'--samples',8,'--seed',9001,'--workers',4]
  if model:command+=['--model',model]
  study.run_process(command,out/'logs'/f'{label}.log',max(1,lease['hard_deadline']-time.time()-600))
  result=study.validate_records(run,rows,8,c['cohort_sha256'],'full',model=str(model) if model else study.BASE_MODEL,revision=None if model else study.BASE_REVISION)
  if result.get('adapter') or result['selection']!='full':raise ValueError('Wrong candidate coverage')
  from experiments.baseline_cache import status_fields
  study.write_json(out/'status'/f'{label}.json',{'state':'complete','finished_at':study.now(),'records_sha256':digest(run/'records.jsonl'),**(status_fields(run) if label=='base' else {})})
  if label != 'base':
   comparison=out/'comparisons'/f'{label}.json'
   comparison.parent.mkdir(parents=True,exist_ok=True)
   study.run_process([sys.executable,study.REPO/'compare.py',out/'runs/base',run,'--out',comparison],out/'logs'/f'compare-{label}.log',600)
   value=json.loads(comparison.read_text())
   if value['ungraded_benchmarks'] or any(value['full_evaluation'].values()):raise ValueError('Incomplete full comparison')
 study.write_json(out/'completion.json',{'complete':True,'finished_at':study.now(),'candidates':len(candidates),'responses_per_model':22072})
 study.write_json(out/'phase.json',{'phase':'finished','updated_at':time.time()})
if __name__=='__main__':main()
