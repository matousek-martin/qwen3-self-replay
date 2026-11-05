import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from ops import shorterbetter_paired as p
class SequentialTests(unittest.TestCase):
 def test_second_uses_remaining_cap_and_only_follows_first_release(self):
  with tempfile.TemporaryDirectory() as tmp:
   directory=Path(tmp);path=directory/'campaign.json';path.write_text(json.dumps({'cap_usd':105,'evaluation_reserve_usd':10,'arms':{'control':{'job_name':'a'},'accuracy':{'job_name':'b'}}}))
   jobs={};events=[]
   def prepare(d,arm,cap):events.append(('prepare',arm,cap));return {'job_name':'a' if arm=='control' else 'b'}
   def launch(l):
    n=l['job_name']
    if n=='b':self.assertTrue(jobs['a']['released_at'])
    events.append(('launch',n));jobs[n]={'released_at':1,'backup_verified':True,'charged_estimate_usd':45}
   with patch.object(p,'jobs',return_value=jobs),patch.object(p,'prepare',side_effect=prepare),patch.object(p,'launch_training',side_effect=launch),patch('ops.shorterbetter_paired_evaluation.run',side_effect=lambda d:events.append(('eval',))):p.run(directory)
   self.assertEqual(events,[('prepare','control',55),('launch','a'),('prepare','accuracy',50),('launch','b'),('eval',)])
   with patch.object(p,'jobs',return_value=jobs),patch.object(p,'complete') as complete,patch.object(p,'launch_training') as launch,patch('ops.shorterbetter_paired_evaluation.run'):
    p.run(directory);self.assertEqual(complete.call_count,2);launch.assert_not_called()
if __name__=='__main__':unittest.main()

class EarlyEvaluationTests(unittest.TestCase):
 def test_early_evaluation_reservation_then_actual_charge_is_subtracted(self):
  c={'cap_usd':105,'arms':{'control':{'job_name':'a'}},'early_evaluations':[{'job_name':'e','cap_usd':5}]}
  ledger={'a':{'charged_estimate_usd':30},'e':{}}
  with patch.object(p,'jobs',return_value=ledger):
   self.assertEqual(p.remaining(c),70)
   ledger['e']={'released_at':1,'charged_estimate_usd':1.5};self.assertEqual(p.remaining(c),73.5)
 def test_only_registered_evaluation_may_overlap_next_training(self):
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp)/'campaign.json';path.write_text(json.dumps({'early_evaluations':[{'job_name':'e'}]}))
   with patch.object(p,'jobs',return_value={'e':{'pod_id':'vast:1'}}),patch.object(p.vastai,'list_pods',return_value=[{'id':'vast:1'}]) as pods:
    p.check_other_instances({'paired_campaign_path':str(path)})
    pods.return_value.append({'id':'vast:2'})
    with self.assertRaises(RuntimeError):p.check_other_instances({'paired_campaign_path':str(path)})
