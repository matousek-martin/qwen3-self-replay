import json
from pathlib import Path
import tempfile
import unittest

from .scheduled_stop import decision,shortened_lease,run,write
from .lease import read_lease


class ScheduledStopTests(unittest.TestCase):
    def test_only_target_validation_triggers_exact_stop(self):
        for phase in ['training','validation']:
            self.assertEqual(decision(dict(current=1023,phase=phase),1024),'waiting')
        self.assertEqual(decision(dict(current=1024,phase='validation'),1024),'stop_after_validation')
        self.assertEqual(decision(dict(current=1024,phase='training'),1024),'late_boundary_stop')
        self.assertEqual(decision(dict(current=1025,phase='training'),1024),'late_boundary_stop')

    def test_lease_expiry_stops_next_iteration_without_extending_budget(self):
        lease=dict(job_name='replay-scaled-runpod-resume-20261102t162212z',pod_id='test',provider='runpod',hardware_scope='replay-scaled-a100-v1',
            gpu='NVIDIA A100-SXM4-80GB',gpu_count=2,created_at=100,max_cost_usd=65,budget_rate=4.6,max_runtime_hours=24,hard_deadline=40000)
        request=dict(identity=lease,target=1024)
        new=shortened_lease(lease,request,1000)
        self.assertTrue(1000>new['hard_deadline']-600)
        self.assertEqual(new['max_cost_usd'],65)
        self.assertLessEqual(new['hard_deadline'],lease['hard_deadline'])
        self.assertEqual(shortened_lease(dict(lease,hard_deadline=1200),request,1000)['hard_deadline'],1200)
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'lease.json';write(p,new)
            self.assertEqual(read_lease(p,lease),new)
        with self.assertRaises(RuntimeError):shortened_lease(lease,dict(request,identity=dict(lease,pod_id='wrong')),1000)

    def test_finished_worker_outputs_are_not_modified(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);write(root/'output/exit.json',{'exit_code':0});write(root/'request.json',{'target':1024,'identity':{'job_name':'replay-scaled-runpod-resume-20261102t162212z'}})
            run(root,root/'request.json')
            self.assertFalse((root/'output/training/scheduled-stop.json').exists())
            self.assertEqual(json.loads((root/'control/replay-stop/status.json').read_text())['state'],'worker_exited')


if __name__=='__main__':unittest.main()
