import tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from .core import write
from . import evaluation_budget as b
from experiments.shorterbetter.budget import evaluation_account_limit

class EvaluationTests(unittest.TestCase):
    def test_1280_providers_share_repair_cap_and_require_released_parent(self):
        with tempfile.TemporaryDirectory() as d:
            state=Path(d);name='replay-evaluation-1280-test'
            parent=dict(released_at=123,backup_verified=True,hardware_scope='replay-scaled-a100-v1',max_cost_usd=28)
            jobs={b.PARENT1280:parent};write(state/'ledger.json',{'jobs':jobs})
            v=dict(job_name=name,authorized=True,scope='replay_runpod_evaluation_v1',provider='runpod',hardware_scope=b.SCOPE,entry=b.ENTRY,
                gpu=b.GPU,gpu_count=1,max_cost_usd=6,max_runtime_hours=6,max_gpu_rate=.89,phase='evaluation',evaluation_only=True,
                training_authorized=False,checkpoint_group=1280,concurrent_training_job=b.PARENT1280,source_user_instruction='Evaluate1280',
                combined_cap_usd=6,previous_attempts=[],account_spending_limit_usd=6)
            path=state/'authorizations'/(name+'.json');write(path,v)
            self.assertIsNotNone(b.ticket(state,name,v))
            prior='replay-evaluation-1280-failed'
            jobs[prior]=dict(state='failed',released_at=123,startup_failed_before_launch=True,charged_estimate_usd=2.5)
            write(state/'ledger.json',{'jobs':jobs})
            v.update(provider='vast',scope='replay_vast_bounded_v1',offer_id=123,disk_gb=60,max_gpu_rate=.5,previous_attempts=[prior],max_cost_usd=4)
            write(path,v);self.assertIsNone(b.ticket(state,name,v))
            v['max_cost_usd']=3.5;write(path,v);self.assertIsNotNone(b.ticket(state,name,v))
            jobs[prior]['released_at']=None;write(state/'ledger.json',{'jobs':jobs})
            self.assertIsNone(b.ticket(state,name,v))

    def test_runpod1024_is_separate_bounded_evaluation_of_released_parent(self):
        with tempfile.TemporaryDirectory() as d:
            state=Path(d);name='replay-evaluation-1024-test';parent='replay-scaled-runpod-resume-20261102t162212z'
            write(state/'ledger.json',{'jobs':{parent:{'released_at':123,'backup_verified':True}}})
            v=dict(job_name=name,authorized=True,scope='replay_runpod_evaluation_v1',provider='runpod',hardware_scope=b.SCOPE,entry=b.ENTRY,
                gpu=b.GPU,gpu_count=1,max_cost_usd=6,max_runtime_hours=6,max_gpu_rate=.89,phase='evaluation',evaluation_only=True,
                training_authorized=False,checkpoint_group=1024,concurrent_training_job=parent,source_user_instruction='Evaluate1024 onRunPod')
            write(state/'authorizations'/(name+'.json'),v)
            self.assertIsNotNone(b.ticket(state,name,v))
            reserved={k:x for k,x in v.items() if k!='gpu'};reserved['requested_gpu']=b.GPU
            self.assertIsNotNone(b.ticket(state,name,reserved))
            for change in [dict(provider='vast'),dict(max_cost_usd=7),dict(gpu_count=2),dict(entry='experiments/replay_scaled/run.sh')]:
                self.assertIsNone(b.ticket(state,name,dict(v,**change)))
            write(state/'ledger.json',{'jobs':{parent:{'backup_verified':True}}})
            self.assertIsNone(b.ticket(state,name,v))
            args=SimpleNamespace(hardware_scope=b.SCOPE,name=name,entry=b.ENTRY,gpu=b.GPU,gpu_count=1,max_cost_usd=6,provider='runpod')
            self.assertEqual(b.allocation_spec(args)['max_gpu_rate'],.99)
            args.provider='vast'
            with self.assertRaises(ValueError):b.allocation_spec(args)
    def test_startup_retry_preserves_combined_cap_and_rejects_live_or_started_jobs(self):
        failed=dict(state='failed',released_at=123,startup_failed_before_launch=True,charged_estimate_usd=.118)
        self.assertEqual(b.startup_retry_budget([failed]),3.882)
        absent=dict(state='create_absent_reconciled',released_at=123,safely_no_pod=True,charged_estimate_usd=0,
                    absence_reconciliation=dict(successful_full_reads=12,elapsed_seconds=301),late_arrival_watch_until=1000)
        self.assertEqual(b.startup_retry_budget([absent]),4)
        with self.assertRaises(ValueError):b.startup_retry_budget([{**absent,'absence_reconciliation':{}}])
        for change in [dict(released_at=None),dict(worker_launch_attempted_at=100),
                       dict(charged_estimate_usd=None),dict(state='complete'),dict(charged_estimate_usd=3.1)]:
            with self.assertRaises(ValueError):b.startup_retry_budget([{**failed,**change}])
    def test_separate_bounded_evaluation_ticket_and_account_guard(self):
        with tempfile.TemporaryDirectory() as d:
            state=Path(d);name='replay-evaluation-128-test';parent='replay-scaled-training-20261101t122440z'
            write(state/'ledger.json',{'jobs':{parent:{'hardware_scope':'replay-scaled-a100-v1','max_cost_usd':27.6779}}})
            v=dict(job_name=name,authorized=True,scope='replay_vast_bounded_v1',provider='vast',hardware_scope=b.SCOPE,entry=b.ENTRY,
                   gpu=b.GPU,gpu_count=1,offer_id=123,disk_gb=60,max_cost_usd=4,max_runtime_hours=6,max_gpu_rate=.4,
                   phase='evaluation',evaluation_only=True,training_authorized=False,checkpoint_group=128,
                   source_user_instruction='Evaluate128',concurrent_training_job=parent,account_spending_limit_usd=14)
            write(state/'authorizations'/(name+'.json'),v)
            self.assertIsNotNone(b.ticket(state,name,v))
            self.assertEqual(evaluation_account_limit(state,name,v),14)
            self.assertIsNone(b.ticket(state,name,dict(v,max_cost_usd=5)))
            self.assertIsNone(b.ticket(state,name,dict(v,entry='experiments/replay_scaled/run.sh')))
            next_name='replay-evaluation-256-test'
            next_ticket=dict(v,job_name=next_name,checkpoint_group=256,source_user_instruction='Evaluate256')
            write(state/'authorizations'/(next_name+'.json'),next_ticket)
            self.assertIsNotNone(b.ticket(state,next_name,next_ticket))
            latest_name='replay-evaluation-576-test'
            latest=dict(next_ticket,job_name=latest_name,checkpoint_group=576,source_user_instruction='Evaluate latest576')
            write(state/'authorizations'/(latest_name+'.json'),latest)
            self.assertIsNotNone(b.ticket(state,latest_name,latest))
            for group in (768,850):
                new_name=f'replay-evaluation-{group}-test'
                new_ticket=dict(v,job_name=new_name,checkpoint_group=group)
                write(state/'authorizations'/(new_name+'.json'),new_ticket)
                self.assertIsNotNone(b.ticket(state,new_name,new_ticket))
            write(state/'authorizations'/(next_name+'.json'),dict(next_ticket,checkpoint_group=512))
            self.assertIsNone(b.ticket(state,next_name,next_ticket))
            write(state/'authorizations'/(name+'.json'),dict(v,training_authorized=True))
            self.assertIsNone(b.ticket(state,name,v))
    def test_gpu_scope_cannot_train_or_use_extra_gpus(self):
        args=SimpleNamespace(hardware_scope=b.SCOPE,name='replay-evaluation-test',entry=b.ENTRY,gpu=b.GPU,gpu_count=1,max_cost_usd=4)
        self.assertEqual(b.allocation_spec(args)['count'],1)
        args.gpu_count=2
        with self.assertRaises(ValueError):b.allocation_spec(args)
        lease=dict(hardware_scope=b.SCOPE,job_name='replay-evaluation-test',gpu=b.GPU,gpu_count=1)
        self.assertEqual(b.verify_worker([b.GPU],lease),[b.GPU])
        with self.assertRaises(RuntimeError):b.verify_worker(['NVIDIA A100-SXM4-80GB'],lease)
