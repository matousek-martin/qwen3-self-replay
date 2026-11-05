import json,tempfile,unittest
from pathlib import Path
from experiments.coding_frontier.grading import check,grade
from experiments.coding_frontier.worker import seed,summary

class SandboxTests(unittest.TestCase):
    def test_resume_preserves_answers_and_rejects_tampering(self):
        import gzip
        from experiments.coding_frontier.resume import restore,validate
        from experiments.opencoder_sft.runtime import sha
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'source';(source/'responses').mkdir(parents=True)
            chunk=source/'responses/a.jsonl.gz'
            with gzip.open(chunk,'wt') as f:
                for i in range(8):f.write(json.dumps(dict(id='a',sample=i,total_tokens=10,batch_generation_seconds=2))+'\n')
            (source/'questions.jsonl').write_text(json.dumps(dict(id='a',samples=8,chunk=chunk.name,chunk_sha256=sha(chunk)))+'\n')
            target=root/'out';counts=restore(source,target)
            self.assertEqual(counts['answers'],8);self.assertEqual(counts['generated_tokens'],80)
            self.assertEqual(counts['generation_seconds'],2)
            self.assertEqual(restore(source,target),counts)
            (target/'responses/a.jsonl.gz').write_bytes(b'corrupt')
            with self.assertRaises(RuntimeError):validate(target)
    def test_interface_exposes_signature_without_answer_or_test_values(self):
        from experiments.coding_frontier.interface import interface_stub
        result=interface_stub('def string_to_number(s):\n return int(s)', 'from solution import string_to_number\ndef test_f(): assert string_to_number("42")==42')
        self.assertIn('def string_to_number(s):',result)
        self.assertNotIn('int(s)',result)
        self.assertNotIn('42',result)
        with self.assertRaises(ValueError):interface_stub('def f(): pass','from solution import missing')
    def test_io_answers_not_available(self):
        spec=dict(kind='io',inputs=['1'],outputs=['2'])
        self.assertEqual(check('print(int(input())+1)',spec)['status'],'passed')
        self.assertEqual(check('print(3)',spec)['status'],'test_failure')
    def test_pytest_and_wrong_solution(self):
        spec=dict(kind='python_tests',test='import pytest\nfrom solution import f\n@pytest.mark.parametrize("x,y", [(1,2),(2,3)])\ndef test_f(x,y): assert f(x)==y')
        self.assertEqual(check('def f(x): return x+1',spec)['tests_passed'],2)
        self.assertEqual(check('def f(x): return x',spec)['status'],'test_failure')
    def test_no_hidden_skip(self):
        spec=dict(kind='python_tests',test='import pytest\nfrom solution import f\n@pytest.mark.skip\ndef test_f(): assert f()==2')
        self.assertEqual(check('def f(): return 2',spec)['status'],'unsupported_test')
    def test_chroot_and_network(self):
        spec=dict(kind='python_tests',test='from solution import f\ndef test_f(): assert f() is True')
        self.assertEqual(check('def f():\n import os\n return os.getuid()==65534 and not os.path.exists("/etc/passwd")',spec)['status'],'passed')
        self.assertEqual(check('def f():\n import socket\n socket.socket()\n return True',spec)['status'],'test_failure')
    def test_format_and_context_are_distinct(self):
        spec=dict(kind='io',inputs=[''],outputs=['2'])
        text='think step\n</think>\nHere is code:\n```python\nprint(2)\n```'
        self.assertTrue(grade(text,'stop',spec)['correct'])
        self.assertEqual(grade(text,'length',spec)['status'],'context_limit')
        self.assertEqual(grade('print(2)','stop',spec)['status'],'incomplete_reasoning')
    def test_seeds(self):
        self.assertEqual(seed('a',0),seed('a',0));self.assertNotEqual(seed('a',0),seed('a',1))
    def test_candidate_bad_return_and_process_exit_are_not_infrastructure(self):
        spec=dict(kind='io',fn_name='f',inputs=[[]],outputs=[1])
        self.assertEqual(check('def f(): return {1,2}',spec)['status'],'execution_error')
        self.assertEqual(check('import os\nos._exit(1)',spec)['status'],'execution_error')

class BudgetTests(unittest.TestCase):
    def test_named_renewal_uses_own_cap_without_legacy_veto(self):
        import time
        from unittest.mock import patch
        from experiments import cloud
        from experiments.coding_frontier import budget
        with tempfile.TemporaryDirectory() as d:
            state=Path(d);name='coding-frontier-apps-fixture';now=time.time()
            job=dict(hardware_scope=budget.SCOPE,created_at=now-1800,deadline=now+100,hard_deadline=now+20000,
                     reserved_usd=20,budget_rate=.7,max_cost_usd=20)
            (state/'ledger.json').write_text(json.dumps(dict(jobs={name:job})))
            with patch.object(cloud,'STATE',state),patch.object(cloud,'budget_allows',return_value=False),patch.object(budget,'ticket',return_value={'max_cost_usd':20}):
                self.assertTrue(cloud.renew_budget(name))
            job['reserved_usd']=21
            (state/'ledger.json').write_text(json.dumps(dict(jobs={name:job})))
            with patch.object(cloud,'STATE',state),patch.object(cloud,'budget_allows',return_value=False),patch.object(budget,'ticket',return_value={'max_cost_usd':20}):
                self.assertFalse(cloud.renew_budget(name))
            job['reserved_usd']=20
            (state/'ledger.json').write_text(json.dumps(dict(jobs={name:job})))
            with patch.object(cloud,'STATE',state),patch.object(cloud,'budget_allows',return_value=True),patch.object(budget,'ticket',return_value=None):
                self.assertFalse(cloud.renew_budget(name))
    def test_gpu_policy_integration(self):
        from experiments.shorterbetter.hardware import verify_worker
        from experiments.coding_frontier.budget import GPU,SCOPE
        names=verify_worker([GPU],dict(gpu=GPU,gpu_count=1,hardware_scope=SCOPE,job_name='coding-frontier-apps-fixture'))
        self.assertEqual(', '.join(names),GPU)
    def test_named_three_arm_funding_and_no_duplicate(self):
        from experiments.coding_frontier.budget import ticket,SCOPE,ENTRY,GPU
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);state=root/'cloud';(state/'authorizations').mkdir(parents=True);reg=root/'coding-frontier/campaign';reg.mkdir(parents=True)
            name='coding-frontier-apps-fixture';path=reg/'campaign.json'
            arms={k:dict(job_name='coding-frontier-'+k+'-fixture',max_cost_usd=20) for k in ['apps','taco','kodcode']}
            path.write_text(json.dumps(dict(authorized=True,cap_usd=60,arms=arms)))
            v=dict(job_name=name,authorized=True,scope=SCOPE,hardware_scope=SCOPE,entry=ENTRY,provider='vast',gpu=GPU,gpu_count=1,
                evaluation_only=True,training_authorized=False,parent_group=768,samples=8,disk_gb=80,combined_cap_usd=60,dataset='apps',
                offer_id=1,max_cost_usd=20,max_runtime_hours=48,max_gpu_rate=.6,campaign_path=str(path),source_user_instruction='fixture')
            auth=state/'authorizations'/(name+'.json');auth.write_text(json.dumps(v));(state/'ledger.json').write_text(json.dumps(dict(jobs={})))
            self.assertIsNotNone(ticket(state,name,v))
            self.assertIsNone(ticket(state,name,dict(v,max_cost_usd=21)))
            self.assertIsNone(ticket(state,name,dict(v,gpu='NVIDIA A100')))
            (state/'ledger.json').write_text(json.dumps(dict(jobs={'coding-frontier-apps-other':dict(pod_id='vast:9')})))
            self.assertIsNone(ticket(state,name,v))
            old='coding-frontier-apps-old';arms['apps']['previous_attempts']=[old]
            path.write_text(json.dumps(dict(authorized=True,cap_usd=60,arms=arms)))
            previous=dict(released_at=1,safely_no_pod=True,charged_estimate_usd=0,allocation_attempts=[{'outcome':'ambiguous'}],
                absence_reconciled_at=1,absence_reconciliation=dict(successful_full_reads=11,elapsed_seconds=301))
            (state/'ledger.json').write_text(json.dumps(dict(jobs={old:previous})))
            self.assertIsNone(ticket(state,name,v))
            previous['absence_reconciliation']['successful_full_reads']=12
            (state/'ledger.json').write_text(json.dumps(dict(jobs={old:previous})))
            self.assertIsNotNone(ticket(state,name,v))

if __name__=='__main__':unittest.main()
