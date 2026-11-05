import copy,json,tempfile,unittest
from pathlib import Path
from .budget import ticket
from .config import training_arguments
from .hardware import allocation_spec,verify_worker,SCOPE,ENTRY
from types import SimpleNamespace

class CheapVerificationTests(unittest.TestCase):
    def test_named_budget_is_bounded_and_hardware_specific(self):
        name='shorterbetter-scaled-verify-one'
        value=dict(authorized=True,job_name=name,scope='shorterbetter_4090_verification',
                   gpu='NVIDIA GeForce RTX 4090',gpu_count=1,max_cost_usd=5,max_runtime_hours=2,max_gpu_rate=.89)
        job=dict(gpu=value['gpu'],gpu_count=1,max_cost_usd=5,hardware_scope=SCOPE,entry=ENTRY)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'authorizations';p.mkdir();(p/(name+'.json')).write_text(json.dumps(value))
            self.assertIsNotNone(ticket(d,name,job))
            for change in ({'gpu_count':2},{'gpu':'NVIDIA A100-SXM4-80GB'},{'max_cost_usd':6},
                           {'completion_budget_scope':'finish_registered_1200_updates'},
                           {'created_at':0,'hard_deadline':7202}):
                self.assertIsNone(ticket(d,name,{**job,**change}))
    def test_4090_count_checked_at_allocator_and_worker(self):
        for count in (1,2):
            a=SimpleNamespace(name='shorterbetter-scaled-verify',entry=ENTRY,hardware_scope=SCOPE,
                              gpu='NVIDIA GeForce RTX 4090',gpu_count=count,max_cost_usd=5)
            self.assertEqual(allocation_spec(a)['count'],count)
            lease=dict(job_name=a.name,hardware_scope=SCOPE,gpu=a.gpu,gpu_count=count)
            verify_worker([a.gpu]*count,lease)
            with self.assertRaises(RuntimeError):verify_worker(['NVIDIA A100-SXM4-80GB']*count,lease)
    def test_technical_retention_keeps_resumable_checkpoint_and_final_model(self):
        from unittest.mock import patch
        from . import verify_cheap
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);a=root/'output/reference';b=root/'output/resumed'
            for folder in (a/'checkpoint-28',a/'checkpoint-32',a/'checkpoint-36',a/'model',
                           b/'checkpoint-32',b/'checkpoint-36',b/'model'):
                folder.mkdir(parents=True);(folder/'artifact').write_bytes(b'verified fixture')
            with patch.object(verify_cheap,'ROOT',root):
                verify_cheap.compact_technical_artifacts(a,b,checkpoint=32,final_step=36,extra=(a/'checkpoint-28',))
            self.assertTrue((a/'checkpoint-32/artifact').is_file())
            self.assertTrue((a/'model/artifact').is_file())
            self.assertFalse((a/'checkpoint-28').exists())
            self.assertFalse((b/'checkpoint-36').exists())
            report=json.loads((root/'output/technical-retention.json').read_text())
            self.assertEqual(report['retained_checkpoint'],'reference/checkpoint-32')
            self.assertEqual(len(report['removed_artifact_sha256']),5)

    def test_short_fixture_cannot_silently_become_main_training(self):
        root=Path(__file__).resolve().parents[2]
        p=json.loads((root/'studies/shorterbetter/protocol.json').read_text())
        p['training'].update(rollout_batches=2,prompts_per_rollout_batch=8,prompts_per_optimizer_minibatch=2,max_completion_tokens=256)
        p['execution']=dict(scaled=True,verification_only=True,phase='benchmark',world_size=1,gpu='NVIDIA GeForce RTX 4090')
        a=training_arguments(p,'shorterbetter',17,'/tmp/test')
        self.assertEqual((a['max_steps'],a['generation_batch_size'],a['gradient_accumulation_steps']),(8,64,16))
        p['execution']['phase']='training'
        with self.assertRaises(ValueError):training_arguments(p,'shorterbetter',17,'/tmp/test')
