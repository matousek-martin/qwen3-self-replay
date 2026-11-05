import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from .coding_budget import SCOPE,PREFIX,ENTRY,GPU,allocation_spec,verify_worker,ticket,admit,select


class CodingReplayTests(unittest.TestCase):
    def test_admit_one_correct_only(self):
        b={};samples=[dict(correct=False,truncated=False,ids=[1]),dict(correct=True,truncated=False,ids=[2,3])]
        admit(b,'q',[7],samples)
        self.assertEqual(b,{'q':([7],[2,3])})
        admit(b,'q',[7],[dict(correct=True,truncated=True,ids=[8])])
        self.assertEqual(b['q'][1],[2,3])

    def test_hardware(self):
        a=SimpleNamespace(hardware_scope=SCOPE,name=PREFIX+'test',entry=ENTRY,gpu=GPU,gpu_count=1,provider='vast',max_cost_usd=18)
        self.assertEqual(allocation_spec(a)['count'],1)
        a.gpu_count=2
        with self.assertRaises(ValueError):allocation_spec(a)
        lease=dict(hardware_scope=SCOPE,job_name=PREFIX+'test',gpu=GPU,gpu_count=1)
        with self.assertRaises(RuntimeError):verify_worker(['NVIDIA A100-SXM4-80GB'],lease)

    def test_selection_and_minimum(self):
        rows=[dict(id=str(i),task='coding',split='train') for i in range(128)]
        retention=[dict(id='r'+str(i),task='math',split='train') for i in range(32)]
        records=[dict(id=str(i),sample=s,correct=i<32 and s==0) for i in range(128) for s in range(4)]
        schedule=select(rows,records,retention)
        self.assertEqual(len(schedule),128)
        self.assertEqual(sum(r['task']=='coding' for r in schedule),96)
        self.assertEqual({r['id'] for r in schedule if r['task']=='coding'},{str(i) for i in range(32)})
        for r in records:
            if r['id']=='31':r['correct']=False
        self.assertIsNone(select(rows,records,retention))
        with self.assertRaises(ValueError):select(rows,records[:-1],retention)

    def test_authorization_and_duplicate_spend(self):
        with tempfile.TemporaryDirectory() as tmp:
            state=Path(tmp);(state/'authorizations').mkdir();name=PREFIX+'test'
            v=dict(job_name=name,authorized=True,scope=SCOPE,provider='vast',hardware_scope=SCOPE,entry=ENTRY,gpu=GPU,gpu_count=1,
                checkpoint_group=768,training_authorized=True,screening_questions=128,training_groups=128,
                stage_caps_usd={'screening':4,'training':10,'evaluation':4},disk_gb=80,offer_id=1,max_cost_usd=18,max_runtime_hours=24,max_gpu_rate=.5,source_user_instruction='approved')
            p=state/'authorizations'/(name+'.json');p.write_text(json.dumps(v));(state/'ledger.json').write_text(json.dumps({'jobs':{}}))
            self.assertIsNotNone(ticket(state,name,v))
            self.assertIsNone(ticket(state,name,dict(v,max_cost_usd=19)))
            self.assertIsNone(ticket(state,name,dict(v,gpu='NVIDIA A100-SXM4-80GB')))
            (state/'ledger.json').write_text(json.dumps({'jobs':{PREFIX+'other':{'pod_id':'vast:1'}}}))
            self.assertIsNone(ticket(state,name,v))


if __name__=='__main__':unittest.main()
