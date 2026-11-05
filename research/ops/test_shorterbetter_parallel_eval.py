import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from ops import shorterbetter_parallel_eval as runner

class PublicationTests(unittest.TestCase):
 def test_only_roundoff_is_accepted_in_recomputed_comparison(self):
  self.assertTrue(runner.comparison_equal({'value':[.2]}, {'value':[.2+1e-15]}))
  self.assertFalse(runner.comparison_equal({'value':[.2]}, {'value':[.201]}))
  self.assertFalse(runner.comparison_equal({'model':'a'}, {'model':'b'}))

 def test_full_pair_receipt_requires_hashes_and_matching_recomputed_metrics(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);state=root/'cloud';directory=root/'registration';reg=directory/'group';release=root/'release';label='control-010';name='shorterbetter-scaled-parallel-test';out=state/name/'output/paired-evaluation'
   def put(p,v):runner.cloud.write(p,v)
   row={'job_name':name,'group':'control-early','directory':str(reg)}
   put(state/'ledger.json',{'jobs':{name:{'created_at':1}}})
   put(reg/'launch.json',{'release':str(release)});q=release/'ThinkingCap-Test/questions/questions.jsonl';q.parent.mkdir(parents=True);q.write_text('{}\n')
   path='/cache/snapshots/'+runner.study.BASE_REVISION
   put(out/'baseline-loading.json',{'model':runner.study.BASE_MODEL,'revision':runner.study.BASE_REVISION,'path':path})
   put(out/'provenance.json',{'job_name':name,'gpu_identity':'GPU-verified'})
   remote={'base':{'run':'/workspace/project/output/paired-evaluation/runs/base'},'candidate':{'run':'/workspace/project/output/paired-evaluation/runs/'+label},'ungraded_benchmarks':[],'full_evaluation':{},'average':{'accuracy_delta':-.01,'token_reduction':.2}}
   put(out/'comparisons'/(label+'.json'),remote)
   for side in ['base',label]:
    records=out/'runs'/side/'records.jsonl';records.parent.mkdir(parents=True);records.write_text('{}\n');put(records.parent/'results.json',{})
    put(out/'status'/(side+'.json'),{'records_sha256':runner.sha(records)})
   receipt=directory/'live-results'/name/(label+'.json')
   sent=SimpleNamespace(db=SimpleNamespace(execute=lambda *args:SimpleNamespace(fetchone=lambda:True)))
   def recompute(cmd,**kwargs):
    v=copy.deepcopy(remote);v['base']['run']=cmd[2];v['candidate']['run']=cmd[3];put(Path(cmd[-1]),v)
   with patch.object(runner.cloud,'STATE',state),patch.object(runner.study,'validate_records') as validate,patch.object(runner.subprocess,'run',side_effect=recompute),patch.dict('sys.modules',{'project_alerts':SimpleNamespace(Alerts=lambda *args,**kwargs:sent)}):
    self.assertTrue(runner.publish_pair(directory,row,label));self.assertTrue(json.loads(receipt.read_text())['verified']);self.assertEqual(validate.call_count,2)
    self.assertEqual(validate.call_args_list[0].kwargs,{'model':path,'revision':None})
    receipt.unlink();put(out/'status'/(label+'.json'),{'records_sha256':'wrong'})
    self.assertFalse(runner.publish_pair(directory,row,label));self.assertFalse(receipt.exists())

if __name__=='__main__':unittest.main()
