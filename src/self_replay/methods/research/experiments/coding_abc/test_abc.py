import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from experiments.coding_abc.logic import rewards,choose_correct,check_contract
from experiments.coding_abc.budget import allocation_spec,verify_worker,ticket,SCOPE,GPU,ENTRY

class ABC(unittest.TestCase):
    def test_correctness_advantages(self):
        self.assertEqual(rewards([True,False,False,False]),([1.,0.,0.,0.],[.75,-.25,-.25,-.25]))
        self.assertEqual(rewards([True]*4)[1],[0.]*4)
        self.assertEqual(rewards([False]*4)[1],[0.]*4)
    def test_selection_not_shortest(self):
        rs=[dict(correct=True,truncated=False,text='x',total_tokens=1),dict(correct=True,truncated=False,text='long',total_tokens=1000),dict(correct=False,text='bad')]
        selected={choose_correct(rs,str(i))['text'] for i in range(40)}
        self.assertEqual(selected,{'x','long'})
        for i in range(10):self.assertEqual(choose_correct(rs,str(i)),choose_correct(rs,str(i)))
    def test_hardware(self):
        args=SimpleNamespace(hardware_scope=SCOPE,name='coding-abc-a-test',entry=ENTRY,gpu=GPU,gpu_count=1,provider='vast',max_cost_usd=12)
        self.assertEqual(allocation_spec(args)['count'],1)
        args.gpu='NVIDIA A100-SXM4-80GB'
        with self.assertRaises(ValueError):allocation_spec(args)
        with self.assertRaises(RuntimeError):verify_worker([GPU],dict(gpu=GPU,gpu_count=1,hardware_scope='other',job_name='coding-abc-a-test'))
    def test_no_overlap_or_teacher(self):
        cfg=dict(arm='a',parent_group=768,length_coefficient=0,steps=48)
        train=[dict(id=str(i),task='coding' if i<36 else 'math',split='train',own_answers=[dict(correct=True)]) for i in range(48)]
        check_contract(cfg,train,[dict(id='dev')])
        with self.assertRaises(ValueError):check_contract(cfg,train,[dict(id='0')])
        train[0]['own_answers']=[]
        with self.assertRaises(ValueError):check_contract(cfg,train,[dict(id='dev')])
    def test_ticket_rejects_unregistered(self):
        with tempfile.TemporaryDirectory() as d:self.assertIsNone(ticket(d,'coding-abc-a-test',{}))
    def test_retry_requires_verified_absence_and_keeps_cap(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);state=root/'cloud';(state/'authorizations').mkdir(parents=True);reg=root/'coding-abc/run';reg.mkdir(parents=True)
            name='coding-abc-b-retry';old='coding-abc-b-first';path=reg/'campaign.json'
            arms={a:dict(job_name=name if a=='b' else 'coding-abc-'+a+'-test',max_cost_usd=12,previous_attempts=[old] if a=='b' else []) for a in ('a','b','c')}
            path.write_text(json.dumps(dict(authorized=True,cap_usd=36,arms=arms)))
            v=dict(job_name=name,authorized=True,scope=SCOPE,hardware_scope=SCOPE,entry=ENTRY,provider='vast',gpu=GPU,gpu_count=1,
                training_authorized=True,parent_group=768,combined_cap_usd=36,disk_gb=80,arm='b',offer_id=1,
                max_cost_usd=12,max_runtime_hours=18,max_gpu_rate=.6,campaign_path=str(path),source_user_instruction='fixture')
            (state/'authorizations'/(name+'.json')).write_text(json.dumps(v))
            oldjob=dict(released_at=1,safely_no_pod=True,charged_estimate_usd=0,absence_reconciled_at=1,
                absence_reconciliation=dict(successful_full_reads=12,elapsed_seconds=300),allocation_attempts=[dict(outcome='ambiguous')])
            (state/'ledger.json').write_text(json.dumps(dict(jobs={old:oldjob})))
            self.assertIsNotNone(ticket(state,name,v))
            oldjob['absence_reconciliation']['successful_full_reads']=11
            (state/'ledger.json').write_text(json.dumps(dict(jobs={old:oldjob})))
            self.assertIsNone(ticket(state,name,v))
            # A released host that never received a worker has no model output
            # to archive; an attempted launch still requires verified backup.
            oldjob=dict(released_at=1,pod_id='vast:123',charged_estimate_usd=.1,startup_failed_before_launch=True)
            v['max_cost_usd']=11.9
            (state/'authorizations'/(name+'.json')).write_text(json.dumps(v))
            (state/'ledger.json').write_text(json.dumps(dict(jobs={old:oldjob})))
            self.assertIsNotNone(ticket(state,name,v))
            oldjob['worker_launch_attempted_at']=1
            (state/'ledger.json').write_text(json.dumps(dict(jobs={old:oldjob})))
            self.assertIsNone(ticket(state,name,v))

if __name__=='__main__':unittest.main()
