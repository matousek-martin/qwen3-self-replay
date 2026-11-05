import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from . import length_budget as b
from .core import write
from .length_probe import cut, summarize


class LengthProbeTests(unittest.TestCase):
    def test_bootstrap_download_failure_falls_back_to_verified_upload(self):
        from unittest.mock import Mock
        import subprocess
        from ops.isolated_cloud import probe_uv_bootstrap
        cloud=Mock();cloud.remote.side_effect=[subprocess.TimeoutExpired('download',125),'']
        sync=Mock();meta={'binary_sha256':'abc','mtime':123}
        probe_uv_bootstrap(cloud,sync,{'id':'test'},'/local/uv','/workspace/project/uv','download_script',meta)
        self.assertEqual(sync.call_count,1)
        self.assertIn('timeout --kill-after=5 110',cloud.remote.call_args_list[0].args[1])
        self.assertIn('hashlib.sha256',cloud.remote.call_args_list[1].args[1])
        self.assertIn('verified_local_binary_upload',cloud.remote.call_args_list[1].args[1])

    def test_repair_deducts_all_prior_costs_and_rejects_live_or_started_jobs(self):
        empty=dict(state='allocation_unavailable',released_at=1,safely_no_pod=True,charged_estimate_usd=0)
        failed=dict(state='failed',released_at=2,startup_failed_before_launch=True,charged_estimate_usd=.0625)
        self.assertEqual(b.remaining_budget([empty,failed]),3.9375)
        for change in [dict(released_at=None),dict(worker_launch_attempted_at=1),dict(charged_estimate_usd=float('nan'))]:
            with self.assertRaises(ValueError):b.remaining_budget([failed|change])
    def test_cloud_accepts_probe_plan_without_unrelated_pilot(self):
        from experiments import cloud
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);entry=root/b.ENTRY;entry.parent.mkdir(parents=True);entry.write_text('true')
            write(root/'lab/experiments/data/coding-length-probe/plan.json',{})
            args=SimpleNamespace(entry=b.ENTRY,hardware_scope=b.SCOPE,resume_startup=False)
            with patch.object(cloud,'ROOT',root),patch.object(cloud.subprocess,'run',return_value=SimpleNamespace(returncode=0)),patch.object(cloud,'allocate',side_effect=RuntimeError('allocation boundary')):
                with self.assertRaisesRegex(RuntimeError,'allocation boundary'):cloud.run(args)

    def test_eos_at_cap_is_complete_and_later_eos_remains_truncated(self):
        self.assertEqual(cut([1,2,9,0],3,{9}),([1,2,9],False))
        self.assertEqual(cut([1,2,9,0],2,{9}),([1,2],True))
        self.assertEqual(cut([1,9,0,0],4,{9}),([1,9],False))

    def test_paired_question_counts_and_rescue(self):
        rs=[dict(id='a',sample=s,cap=c,difficulty='introductory',correct=c>=4096,truncated=c<4096,total_tokens=min(c,3000))
            for s in range(4) for c in [2048,4096,8192]]
        result=summarize(rs,[dict(cap_seconds={'2048':10,'4096':15,'8192':15})])
        metrics=[r for r in result['metrics'] if r['difficulty']=='all']
        self.assertEqual([r['correct'] for r in metrics],[0,4,4])
        self.assertEqual([r['four_of_four'] for r in metrics],[0,1,1])
        self.assertEqual(result['paired_deltas'][0]['accuracy_delta'],1)
        self.assertEqual(result['paired_deltas'][1]['accuracy_delta'],0)

    def test_scope_rejects_training_hardware_or_budget_expansion(self):
        args=dict(hardware_scope=b.SCOPE,name=b.PREFIX+'test',entry=b.ENTRY,gpu=b.GPU,gpu_count=1,max_cost_usd=4)
        self.assertEqual(b.allocation_spec(SimpleNamespace(**args))['count'],1)
        for change in [dict(entry='experiments/replay_scaled/run.sh'),dict(gpu_count=2),dict(max_cost_usd=5)]:
            with self.assertRaises(ValueError):b.allocation_spec(SimpleNamespace(**(args|change)))
        with tempfile.TemporaryDirectory() as temp:
            state=Path(temp);name=args['name']
            v=dict(args,job_name=name,authorized=True,scope=b.SCOPE,provider='vast',phase='evaluation',
                evaluation_only=True,training_authorized=False,checkpoint_group=850,caps=[2048,4096,8192],samples=4,
                questions=47,disk_gb=60,offer_id=1,max_runtime_hours=6,max_gpu_rate=.4,source_user_instruction='test')
            write(state/'authorizations'/(name+'.json'),v)
            self.assertIsNotNone(b.ticket(state,name,v))
            for change in [dict(entry='experiments/replay_scaled/run.sh'),dict(gpu_count=2),dict(max_cost_usd=5),dict(max_runtime_hours=7)]:
                self.assertIsNone(b.ticket(state,name,v|change))
            for change in [dict(training_authorized=True),dict(questions=48),dict(caps=[2048,8192]),dict(max_cost_usd=float('nan'))]:
                write(state/'authorizations'/(name+'.json'),v|change)
                self.assertIsNone(b.ticket(state,name,v))


if __name__=='__main__': unittest.main()
