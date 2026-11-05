import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from .runtime import SCOPE,PREFIX,ENTRY,GPU,allocation_spec,verify_worker,ticket,write

class Policy(unittest.TestCase):
    def test_hardware_scoped_both_ends(self):
        args=dict(hardware_scope=SCOPE,name=PREFIX+'test',entry=ENTRY,gpu=GPU,gpu_count=1,provider='vast',max_cost_usd=15)
        allocation_spec(SimpleNamespace(**args));verify_worker([GPU],dict(args,job_name=args['name']))
        for k,v in [('gpu_count',4),('gpu','NVIDIA A100-SXM4-80GB'),('max_cost_usd',16),('entry','other.sh')]:
            with self.assertRaises(ValueError):allocation_spec(SimpleNamespace(**dict(args,**{k:v})))
        with self.assertRaises(RuntimeError):verify_worker(['NVIDIA A100-SXM4-80GB'],dict(args,job_name=args['name']))
    def test_ticket_rejects_duplicate_or_overbudget(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);name=PREFIX+'test';v=dict(job_name=name,authorized=True,scope=SCOPE,hardware_scope=SCOPE,entry=ENTRY,provider='vast',gpu=GPU,gpu_count=1,checkpoint_group=768,training_authorized=True,coding_questions=2000,optimizer_steps=84,disk_gb=80,offer_id=123,max_cost_usd=15,max_runtime_hours=18,max_gpu_rate=.5,source_user_instruction='test')
            write(root/'authorizations'/(name+'.json'),v);write(root/'ledger.json',{'jobs':{}})
            self.assertIsNotNone(ticket(root,name,v))
            self.assertIsNone(ticket(root,name,dict(v,max_cost_usd=16)))
            write(root/'ledger.json',{'jobs':{PREFIX+'other':{'pod_id':'old'}}})
            self.assertIsNone(ticket(root,name,v))

if __name__=='__main__':unittest.main()
