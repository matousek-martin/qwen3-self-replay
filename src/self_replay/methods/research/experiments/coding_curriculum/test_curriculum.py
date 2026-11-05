import copy,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from experiments.coding_curriculum import budget as b
from experiments.shorterbetter.hardware import allocation_spec,verify_worker
from experiments.opencoder_sft.runtime import write

class CurriculumTests(unittest.TestCase):
    def test_hardware_is_scoped(self):
        a=SimpleNamespace(hardware_scope=b.SCOPE,name=b.PREFIX+'test',entry=b.ENTRY,gpu=b.GPU,gpu_count=2,provider='vast',max_cost_usd=24)
        self.assertEqual(allocation_spec(a)['count'],2)
        for field,value in [('gpu','NVIDIA H100 80GB HBM3'),('gpu_count',4),('max_cost_usd',25),('entry','experiments/replay_scaled/run.sh')]:
            invalid=copy.copy(a);setattr(invalid,field,value)
            with self.assertRaises(ValueError):allocation_spec(invalid)
        lease=dict(hardware_scope=b.SCOPE,job_name=a.name,gpu=b.GPU,gpu_count=2)
        self.assertEqual(verify_worker([b.GPU]*2,lease),[b.GPU]*2)
        with self.assertRaises(RuntimeError):verify_worker([b.GPU],lease)
        with self.assertRaises(RuntimeError):verify_worker(['NVIDIA GeForce RTX 4090']*2,lease)

    def test_ticket_rejects_expansion_and_duplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            state=Path(tmp)/'cloud';state.mkdir();name=b.PREFIX+'test';cp=Path(tmp)/'coding-curriculum/campaign.json'
            c=dict(authorized=True,cap_usd=24,job_name=name,datasets=['kodcode','taco'],previous_attempts=[]);write(cp,c)
            v=dict(job_name=name,authorized=True,scope=b.SCOPE,hardware_scope=b.SCOPE,entry=b.ENTRY,provider='vast',gpu=b.GPU,gpu_count=2,
                parent_group=256,training_authorized=True,combined_cap_usd=24,disk_gb=80,offer_id=None,max_cost_usd=24,max_runtime_hours=10,
                max_gpu_rate=1.05,campaign_path=str(cp),source_user_instruction='test')
            write(state/'authorizations'/(name+'.json'),v);write(state/'ledger.json',dict(jobs={}))
            self.assertIsNotNone(b.ticket(state,name,v))
            self.assertIsNone(b.ticket(state,name,dict(v,max_cost_usd=25)))
            write(state/'ledger.json',dict(jobs={b.PREFIX+'other':dict(pod_id='vast:123')}))
            self.assertIsNone(b.ticket(state,name,v))

if __name__=='__main__':unittest.main()
