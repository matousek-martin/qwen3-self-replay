import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from experiments.shorterbetter.paired_eval import bundle
class BundleTests(unittest.TestCase):
 def test_all_twelve_registered_snapshots_share_lease_and_correct_steps(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);path=root/'bundle.json';labels=[f'{arm}-{i:03d}' for arm in ('control','accuracy') for i in range(5,31,5)]
   c={'evaluation_only':True,'training_authorized':False,'job_name':'evaluation','max_cost_usd':15,'candidates':[{'label':s,'config':s+'.json'} for s in labels]};path.write_text(json.dumps(c))
   def inputs(p):
    arm,i=p.stem.split('-');return ({'paired_short_evaluation':True,'job_name':'evaluation','max_cost_usd':15,'candidate':{'optimizer_step':int(i)*4,'source_training_job':'shorterbetter-scaled-paired-'+arm+'-test'}},root,[],root,p)
   def adapted(p):return inputs(p)[:4]
   with patch('experiments.shorterbetter.paired_eval.inputs',side_effect=adapted):
    self.assertEqual(len(bundle(path)[1]),12)
    c['early_control_pair']=True
    c['candidates']=[{'label':x,'config':x+'.json'} for x in ['control-010','control-020']]
    path.write_text(json.dumps(c));self.assertEqual(len(bundle(path)[1]),2)
    c['candidates'][0]['label']='control-006';path.write_text(json.dumps(c))
    with self.assertRaises(ValueError):bundle(path)
    c.pop('early_control_pair');c['parallel_snapshot_shard']=True;c['registered_labels']=['accuracy-010','accuracy-020'];c['candidates']=[{'label':x,'config':x+'.json'} for x in c['registered_labels']]
    path.write_text(json.dumps(c));self.assertEqual(len(bundle(path)[1]),2)
    c['registered_labels']=['accuracy-010','accuracy-010'];path.write_text(json.dumps(c))
    with self.assertRaises(ValueError):bundle(path)
if __name__=='__main__':unittest.main()

class ExecutionTests(unittest.TestCase):
 def test_comparison_directory_and_local_baseline_are_used(self):
  import sys
  from experiments.shorterbetter import paired_eval as runner
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);out=root/'out';cfg=root/'bundle.json';cfg.write_text('{}');base=root/'base';base.mkdir();(base/'config.json').write_text('{}')
   model=root/'candidate';model.mkdir();config={'job_name':'evaluation','max_cost_usd':5.5,'baseline_loading':'pinned_local'}
   (root/'worker-lease.json').write_text(json.dumps(dict(config,hard_deadline=10**12)))
   candidate=('control-010',{'cohort_sha256':'x'},root,[],model,None);commands=[]
   def run(command,*args):
    command=list(map(str,command));commands.append(command);target=Path(command[command.index('--out')+1])
    if command[1].endswith('compare.py'):
     # The real CLI does not create this parent: the previous production failure.
     target.write_text(json.dumps({'ungraded_benchmarks':[],'full_evaluation':{}}))
    else:
     target.mkdir(parents=True);(target/'results.json').write_text('{}');(target/'records.jsonl').write_text('{}\n')
   with patch.object(sys,'argv',['paired_eval','--config',str(cfg),'--out',str(out)]),patch.object(runner,'bundle',return_value=(config,[candidate])),patch.object(runner.study,'ROOT',root),patch.object(runner,'verify_worker'),patch.object(runner.subprocess,'check_output',return_value='GPU-test'),patch.object(runner.study,'preflight_nltk'),patch.object(runner,'tokenizer_parity'),patch.object(runner.study,'run_process',side_effect=run),patch.object(runner.study,'validate_records',return_value={'selection':'full'}),patch.dict(sys.modules,{'huggingface_hub':__import__('types').SimpleNamespace(snapshot_download=lambda *a,**k:str(base))}):
    runner.main()
   self.assertTrue((out/'completion.json').exists());self.assertTrue((out/'comparisons/control-010.json').exists())
   baseline_command=next(c for c in commands if str(out/'runs/base') in c)
   self.assertEqual(baseline_command[baseline_command.index('--model')+1],str(base))
