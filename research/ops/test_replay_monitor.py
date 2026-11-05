import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch,Mock
from ops import replay_scaled as m

class MonitorRecoveryTest(unittest.TestCase):
 def test_exited_controller_does_not_publish_worker_terminal(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);state=root/'cloud';reg=root/'registration';reg.mkdir();state.mkdir()
   name='existing-training'
   m.write(reg/'launch.json',dict(job_name=name,phase='main',release=str(root/'release'),payload_sha256='proof',max_cost_usd=75,max_runtime_hours=36,offer_id=1))
   m.write(reg/'campaign.json',dict(full_training_authorized=True,total=3840,main_authorization=dict(cap_usd=75,source_user_instruction='Continue')))
   m.write(state/'ledger.json',dict(jobs={name:dict(identity_confirmed=True,state='running',pod_id='vast:existing')}))
   process=Mock(returncode=1);process.poll.return_value=1
   with patch.object(m,'REG',reg),patch.object(m.cloud,'STATE',state),patch.object(m,'verify',return_value='proof'),patch.object(m.subprocess,'Popen',return_value=process),patch.object(m,'deliver_once') as send:
    with self.assertRaisesRegex(RuntimeError,'before confirmed release'):m.run(observe_only=True)
   self.assertEqual(m.read(reg/'campaign.json')['state'],'training_running');send.assert_not_called()

if __name__=='__main__':unittest.main()
