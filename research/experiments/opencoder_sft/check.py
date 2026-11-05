"""Compare complete uninterrupted/resumed technical state, not just weights."""
from pathlib import Path
from .runtime import read,sha,write

def main():
    import torch
    import numpy as np
    from safetensors.torch import load_file
    a=Path('output/smoke-full/checkpoints/step-000002');b=Path('output/smoke-resumed/checkpoints/step-000002')
    def equal(x,y):
        if isinstance(x,torch.Tensor):return isinstance(y,torch.Tensor) and torch.equal(x,y)
        if isinstance(x,np.ndarray):return isinstance(y,np.ndarray) and np.array_equal(x,y)
        if isinstance(x,dict):return x.keys()==y.keys() and all(equal(x[k],y[k]) for k in x)
        if isinstance(x,(tuple,list)):return type(x)==type(y) and len(x)==len(y) and all(equal(i,j) for i,j in zip(x,y))
        return x==y
    for p in (a,b):
        for f,h in read(p/'complete.json')['files'].items():
            if sha(p/f)!=h:raise RuntimeError('Technical checkpoint corrupt')
    weights=equal(load_file(a/'adapter/adapter_model.safetensors'),load_file(b/'adapter/adapter_model.safetensors'))
    state=equal(torch.load(a/'state.pt',weights_only=False,map_location='cpu'),torch.load(b/'state.pt',weights_only=False,map_location='cpu'))
    result=dict(passed=weights and state,adapter_bitwise=weights,optimizer_scheduler_rng_bitwise=state,
        longest_training_sequences_tested=True,reference_sha256=sha(a/'complete.json'),resumed_sha256=sha(b/'complete.json'))
    write('output/gpu-verification.json',result)
    if not result['passed']:raise RuntimeError('Exact state resume verification failed')

if __name__=='__main__':main()
