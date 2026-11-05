"""Verify full/split GPU pilot final weights, Adam state and replay/RNG state."""
import argparse,json
from pathlib import Path
from .core import write
from .generation import PROTOCOL, settings
from .communication import PROTOCOL as COMMUNICATION_PROTOCOL

def equal(a,b):
    import torch
    import numpy as np
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b

def compare(reference,resumed,world):
    import torch
    from safetensors.torch import load_file
    a=Path(reference)/'checkpoints/group-000004';b=Path(resumed)/'checkpoints/group-000004'
    if not (a/'complete.json').exists() or not (b/'complete.json').exists():raise RuntimeError('Pilot did not finish both paths')
    if json.loads((a/'complete.json').read_text())['contract']!=json.loads((b/'complete.json').read_text())['contract']:
        raise RuntimeError('Pilot execution contracts differ')
    for rank in range(world):
        ga=json.loads((Path(reference)/f'generation-rank-{rank}.json').read_text())
        gb=json.loads((Path(resumed)/f'generation-rank-{rank}.json').read_text())
        if ga!=gb or ga.get('protocol')!=PROTOCOL:raise RuntimeError('Generation protocol differs or was not exercised')
        if ga.get('communication_protocol')!=COMMUNICATION_PROTOCOL:raise RuntimeError('Record transport was not verified')
    memory=json.loads((Path(reference)/'maximum-length-check.json').read_text())
    if not memory.get('passed') or memory.get('tokens')!=2048:raise RuntimeError('Maximum-length backward check missing')
    if not equal(load_file(a/'adapter/adapter_model.safetensors'),load_file(b/'adapter/adapter_model.safetensors')):raise RuntimeError('Split-run adapter mismatch')
    sa=torch.load(a/'state.pt',map_location='cpu',weights_only=False);sb=torch.load(b/'state.pt',map_location='cpu',weights_only=False)
    if sa['updates'] < 1 or not sa['buffer']:raise RuntimeError('Pilot exercised neither optimizer nor replay sufficiently')
    for key in ['optimizer','buffer','replay_rng','done','updates','references','baseline']:
        if not equal(sa[key],sb[key]):raise RuntimeError('Split-run state mismatch: '+key)
    for rank in range(world):
        if not equal(torch.load(a/f'rng-{rank}.pt',weights_only=False),torch.load(b/f'rng-{rank}.pt',weights_only=False)):
            raise RuntimeError('Split-run rank RNG mismatch')
    return dict(passed=True,adapter_bitwise_equal=True,optimizer_bitwise_equal=True,replay_equal=True,rank_rng_equal=True,world_size=world,groups=4,checkpoint_group=2,generation_token_cap=256,generation_protocol=PROTOCOL,generation_settings={k:ga[k] for k in settings()},communication_protocol=COMMUNICATION_PROTOCOL,technical_only=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reference',required=True);p.add_argument('--resumed',required=True);p.add_argument('--world',type=int,required=True);p.add_argument('--out',required=True)
    a=p.parse_args();write(a.out,compare(a.reference,a.resumed,a.world))
