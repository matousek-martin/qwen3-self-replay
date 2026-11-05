import unittest
from unittest.mock import patch
import torch
from .deterministic_ddp import rank_order_average

class DeterministicReductionTest(unittest.TestCase):
    def test_rank_order_survives_bucket_permutation(self):
        ranks=[torch.tensor(v,dtype=torch.float32) for v in ([1e10,1,9],[-1e10,2,-3],[3,3,-3],[5,4,1])]
        def run(order):
            bucket=torch.empty(3)
            class Bucket:
                def buffer(self):return bucket
            class Work:
                def get_future(self):
                    future=torch.futures.Future();future.set_result(None);return future
            def gather(targets,value,**kwargs):
                for target,value in zip(targets,ranks):target.copy_(value[order])
                return Work()
            with patch('torch.distributed.get_world_size',return_value=4),patch('torch.distributed.all_gather',side_effect=gather):
                return rank_order_average(object(),Bucket()).wait().clone()
        torch.testing.assert_close(run([0,1,2]),torch.tensor([2.,2.5,1.]))
        self.assertTrue(torch.equal(run([0,1,2]),run([2,0,1])[[1,2,0]]))
    def test_workspace_is_chunked_and_result_keeps_original_bucket(self):
        value=torch.arange(9,dtype=torch.float32)
        expected=value.clone()+.5
        widths=[]
        class Bucket:
            def buffer(self):return value
        def gather(targets,part,**kwargs):
            self.assertNotIn('async_op',kwargs)
            widths.append(part.numel())
            targets[0].copy_(part);targets[1].copy_(part+1)
        with patch('torch.distributed.get_world_size',return_value=2), \
             patch('torch.distributed.all_gather',side_effect=gather), \
             patch('experiments.shorterbetter.deterministic_ddp.CHUNK_ELEMENTS',4):
            result=rank_order_average(object(),Bucket()).wait()
        self.assertEqual(widths,[4,4,1])
        self.assertEqual(result.data_ptr(),value.data_ptr())
        self.assertTrue(torch.equal(result,expected))

    def test_rejects_bf16_gradients(self):
        class Bucket:
            def buffer(self):return torch.ones(3,dtype=torch.bfloat16)
        with self.assertRaises(RuntimeError):rank_order_average(object(),Bucket())
