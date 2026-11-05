import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from experiments.shorterbetter.multidomain.test_full_run import FullRunTests
from experiments.shorterbetter.multidomain.full_run import validate,technical_protocol
from experiments.shorterbetter.config import training_arguments
from experiments.shorterbetter.live_metrics import publish_series,preserve_snapshot,DOMAINS

class PairedTests(unittest.TestCase):
 def test_registered_variants_and_engineering_separation(self):
  p=FullRunTests().protocol();p['execution'].update(paired_short_experiment=True,preserve_every_rollouts=5)
  p['training'].update(rollout_batches=30,optimizer_updates_total=120)
  p['stopping']={'quality_stopping_enabled':False}
  p['authorization']['quality_stopping_disabled_by_user']=True
  for alpha in [2.,5.]:
   p['arms']['shorterbetter']['alpha']=p['reward']['alpha']=alpha
   validate(p);a=training_arguments(p,'shorterbetter',17,'/tmp/test')
   self.assertEqual((a['max_steps'],a['save_steps'],a['eval_steps']),(120,20,40))
   q=technical_protocol(p);self.assertNotIn('paired_short_experiment',q['execution'])
   self.assertTrue(q['stopping']['quality_stopping_enabled']);validate(q)
  for change in [('reward',{'alpha':3}),('training',{'rollout_batches':40}),('execution',{'preserve_every_rollouts':10})]:
   q=copy.deepcopy(p);q[change[0]].update(change[1])
   with self.assertRaises(ValueError):validate(q)
 def test_complete_rollout_only_and_correct_aggregation(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp)
   for rank in range(4):
    folder=root/f'rank-{rank:03d}';folder.mkdir();rows=[]
    for i in range(16):
     for j in range(8):rows.append(dict(optimizer_step=0,split='train',prompt_id=f'{rank}-{i}',task=DOMAINS[i%4],total_tokens=j+1,correct=j%2==0))
    (folder/'rollouts.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n{')
   self.assertEqual(publish_series(root,3)['series']['overall'],[])
   row=publish_series(root,4)['series']['overall'][0]
   self.assertEqual((row['iteration'],row['median_tokens'],row['accuracy']),(1,4.5,.5))
   (root/'rank-003/rollouts.jsonl').unlink();self.assertEqual(publish_series(root,4)['series']['overall'],[])
 def test_snapshot_survives_rotation_and_excludes_optimizer(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);cp=root/'checkpoint-20';cp.mkdir()
   for name,content in {'model.safetensors':b'weights','optimizer.pt':b'private-state','config.json':b'{}','tokenizer_config.json':b'{}','trainer_state.json':b'{"global_step":20}'}.items():(cp/name).write_bytes(content)
   preserve_snapshot(root,20);snap=root/'snapshots/iteration-005'
   self.assertEqual((cp/'model.safetensors').stat().st_ino,(snap/'model.safetensors').stat().st_ino)
   shutil.rmtree(cp);self.assertEqual((snap/'model.safetensors').read_bytes(),b'weights');self.assertFalse((snap/'optimizer.pt').exists())
   m=json.loads((snap/'snapshot-manifest.json').read_text());self.assertEqual(m['rollout'],5)
   for name,h in m['files_sha256'].items():self.assertEqual(hashlib.sha256((snap/name).read_bytes()).hexdigest(),h)

if __name__=='__main__':unittest.main()

class CampaignGuards(unittest.TestCase):
 def test_joint_cap_and_predecessor_archive_are_required(self):
  from experiments.shorterbetter.budget import ticket
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);state=root/'cloud';(state/'authorizations').mkdir(parents=True)
   directory=root/'shorterbetter-paired/run';directory.mkdir(parents=True);path=directory/'campaign.json'
   names=['shorterbetter-scaled-paired-control-test','shorterbetter-scaled-paired-accuracy-test']
   c={'authorized':True,'cap_usd':105,'evaluation_reserve_usd':10,'arms':{'control':{'job_name':names[0]},'accuracy':{'job_name':names[1]}}};path.write_text(json.dumps(c))
   ledger={'jobs':{names[0]:{'charged_estimate_usd':45,'released_at':1,'backup_verified':True}}};(state/'ledger.json').write_text(json.dumps(ledger))
   auth={'authorized':True,'job_name':names[1],'scope':'shorterbetter_vast_bounded_v1','provider':'vast','gpu':'NVIDIA A100-SXM4-80GB','gpu_count':4,'offer_id':1,'disk_gb':80,'max_cost_usd':50,'max_runtime_hours':10,'max_gpu_rate':1.1,'source_user_instruction':'explicit','paired_short_experiment':True,'paired_campaign_path':str(path),'rollout_batches':30}
   ap=state/'authorizations'/(names[1]+'.json');ap.write_text(json.dumps(auth));job=dict(auth,entry='experiments/shorterbetter/run-scaled.sh',hardware_scope='shorterbetter-scaled-v1')
   self.assertIsNotNone(ticket(state,names[1],job))
   auth['max_cost_usd']=job['max_cost_usd']=51;ap.write_text(json.dumps(auth));self.assertIsNone(ticket(state,names[1],job))
   auth['max_cost_usd']=job['max_cost_usd']=50;ap.write_text(json.dumps(auth));ledger['jobs'][names[0]]['backup_verified']=False;(state/'ledger.json').write_text(json.dumps(ledger));self.assertIsNone(ticket(state,names[1],job))
 def test_engineering_backup_exclusions_only_before_final_marker(self):
  from experiments import cloud
  from unittest.mock import patch
  from contextlib import contextmanager
  @contextmanager
  def ledger():yield {'jobs':{'test':{'preserve_hardlinks':True}}}
  with tempfile.TemporaryDirectory() as tmp,patch.object(cloud,'STATE',Path(tmp)),patch.object(cloud,'ledger',ledger),patch.object(cloud,'sync') as sync,patch.object(cloud,'remote',return_value='active') as remote:
   cloud.backup({},'test');self.assertTrue(sync.call_args.kwargs['preserve_hardlinks']);self.assertIn('/verification-*/checkpoint-*',sync.call_args.kwargs['exclusions'])
   remote.return_value='finished';cloud.backup({},'test');self.assertNotIn('exclusions',sync.call_args.kwargs)
