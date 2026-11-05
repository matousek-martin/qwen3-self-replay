"""Actual Gloo sum reduction must match one serial global replay update."""
import tempfile,unittest
from pathlib import Path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from experiments.replay_scaled.worker import reduce_gradients
from experiments.replay_scaled.check import equal
from experiments.replay_scaled.communication import transport_probe

def task(rank,world,path):
    dist.init_process_group('gloo',init_method='file://'+path,rank=rank,world_size=world)
    p=torch.nn.Parameter(torch.tensor([.2,-.3],dtype=torch.float64))
    opt=torch.optim.AdamW([p],lr=.001,weight_decay=0)
    for i in range(rank,4,world):((p*(i+1)).sum()**2/4).backward()
    if rank==0:(-.1*torch.log_softmax(p,0)[0]).backward()
    reduce_gradients([p],dist,torch)
    q=torch.nn.Parameter(torch.tensor([.2,-.3],dtype=torch.float64));oq=torch.optim.AdamW([q],lr=.001,weight_decay=0)
    loss=sum((q*(i+1)).sum()**2/4 for i in range(4))-.1*torch.log_softmax(q,0)[0];loss.backward()
    assert torch.allclose(p.grad,q.grad,atol=1e-12,rtol=0)
    opt.step();oq.step();assert torch.equal(p,q)
    dist.destroy_process_group()

def record_task(rank,world,path):
    dist.init_process_group('gloo',init_method='file://'+path,rank=rank,world_size=world)
    group=dist.new_group(backend='gloo')
    assert transport_probe(dist,torch,group,'cpu')['passed']
    dist.destroy_process_group(group)
    dist.destroy_process_group()

class DistributedTests(unittest.TestCase):
    def test_multi_megabyte_records_and_reductions(self):
        with tempfile.TemporaryDirectory() as tmp:
            mp.start_processes(record_task,args=(2,str(Path(tmp)/'init')),nprocs=2,start_method='fork',join=True)
    def test_two_and_four_rank_global_loss(self):
        for world in [2,4]:
            with tempfile.TemporaryDirectory() as tmp:mp.start_processes(task,args=(world,str(Path(tmp)/'init')),nprocs=world,start_method='fork',join=True)
    def test_checkpoint_state_comparison(self):
        a={'optimizer':{'state':torch.tensor([1.,2.])},'buffer':[([1],[2])]}
        self.assertTrue(equal(a,a));self.assertFalse(equal(a,{'optimizer':{'state':torch.tensor([1.,3.])},'buffer':[([1],[2])]}))
