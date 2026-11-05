"""One/two RTX4090 technical verification; never starts the scientific training."""
import json
from pathlib import Path
import subprocess
import shutil
import sys
import time
from .config import assert_launch_ready
from .distributed import atomic_json
from .launch_checks import seed_resume
from .resume_repair import compare
from .throughput import read_rows
from .prepare import sha
ROOT = Path(__file__).resolve().parents[2]


def trace_compare(reference, resumed, world):
    report=[]
    for rank in range(world):
        def rows(folder,name):
            return [r for r in read_rows(folder/f'rank-{rank:03d}'/name) if 4 <= r['optimizer_step'] < 8]
        left,right=rows(reference,'input-trace.jsonl'),rows(resumed,'input-trace.jsonl')
        differences=[]
        for index,(a,b) in enumerate(zip(left,right)):
            keys=[k for k in set(a['hashes'])|set(b['hashes']) if a['hashes'].get(k)!=b['hashes'].get(k)]
            if keys:differences.append({'index':index,'step':a['optimizer_step'],'keys':sorted(keys)})
        ga,gb=rows(reference,'gradient-trace.jsonl'),rows(resumed,'gradient-trace.jsonl')
        report.append({'rank':rank,'input_rows':[len(left),len(right)],'input_differences':differences,
                       'gradient_trace_equal':ga==gb,'gradients_reference':ga,'gradients_resumed':gb})
    return report


def optimizer_compare(reference, resumed, step=8):
    import torch
    a=torch.load(reference/f'checkpoint-{step}/optimizer.pt',map_location='cpu',mmap=True,weights_only=False)
    b=torch.load(resumed/f'checkpoint-{step}/optimizer.pt',map_location='cpu',mmap=True,weights_only=False)
    if a['param_groups']!=b['param_groups'] or a['state'].keys()!=b['state'].keys():
        raise RuntimeError('Optimizer checkpoint structure differs')
    identical=True;maximum=0.;count=0
    for index,state in a['state'].items():
        for key,value in state.items():
            other=b['state'][index][key]
            if isinstance(value,torch.Tensor):
                torch.testing.assert_close(value,other,rtol=1e-6,atol=1e-10)
                identical &= torch.equal(value,other)
                maximum=max(maximum,float((value-other).abs().max()));count+=1
            elif value!=other:raise RuntimeError('Optimizer scalar state differs')
    return {'status':'passed','bit_identical':identical,'max_abs_difference':maximum,'tensors':count}


def compact_technical_artifacts(reference, resumed, checkpoint=4, final_step=8, extra=()):
    # Retain one complete resumable checkpoint and one final model. The other
    # technical copies were consumed by comparisons, not scientific candidates.
    folders=(reference/f'checkpoint-{final_step}',resumed/f'checkpoint-{checkpoint}',resumed/f'checkpoint-{final_step}',resumed/'model',*extra)
    removed={}
    for folder in folders:
        for path in folder.rglob('*'):
            if path.is_file():removed[str(path.relative_to(ROOT/'output'))]=sha(path)
    atomic_json(ROOT/'output/technical-retention.json',{
        'retained_checkpoint':str((reference/f'checkpoint-{checkpoint}').relative_to(ROOT/'output')),
        'retained_final_model':str((reference/'model').relative_to(ROOT/'output')),
        'reason':'Redundant technical artifacts after weight/optimizer resume comparison; no scientific checkpoint deleted',
        'removed_artifact_sha256':removed})
    for folder in folders:
        shutil.rmtree(folder)


def run(protocol,path,lease,evidence):
    assert_launch_ready(protocol,evidence)
    readiness=ROOT/'output/readiness-at-start.json';atomic_json(readiness,evidence)
    world=lease['gpu_count'];reference=ROOT/'output/verification-reference';resumed=ROOT/'output/verification-resumed'
    command=[sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1',
        '--nproc_per_node='+str(world),'--max_restarts=0','-m','experiments.shorterbetter.train',
        '--protocol',str(path),'--arm','shorterbetter','--seed','17','--execute',
        '--readiness',str(readiness),'--job-name',lease['job_name']]
    atomic_json(ROOT/'output/phase.json',{'phase':'cheap_verification_reference','gpu_count':world,'time':time.time()})
    subprocess.run(command+['--out',str(reference)],check=True)
    seed_resume(reference,resumed,world,step=4)
    atomic_json(ROOT/'output/phase.json',{'phase':'cheap_verification_resume','gpu_count':world,'time':time.time()})
    subprocess.run(command+['--out',str(resumed),'--resume',str(resumed/'checkpoint-4')],check=True)
    atomic_json(ROOT/'output/resume-trace-comparison.json',trace_compare(reference,resumed,world))
    result=compare(reference,resumed,world,step=4)
    result['optimizer']=optimizer_compare(reference,resumed)
    for folder in (reference,resumed):
        if json.loads((folder/'training-finished.json').read_text())['optimizer_steps']!=8:
            raise RuntimeError('Technical verification ended before all registered updates')
    result.update(gpu_count=world,gpu='NVIDIA GeForce RTX 4090',phase='technical_only',
        completion_tokens=256,prompt_tokens=128,generation_batch_size=64,optimizer_batch_size=16,
        protocol_sha256=sha(path),main_training_started=False,
        limitation='Short sequences/small batches do not certify full-length A100 memory, throughput or efficacy.')
    atomic_json(ROOT/'output/verification-result.json',result)
    compact_technical_artifacts(reference,resumed)
    atomic_json(ROOT/'output/phase.json',{'phase':'finished','time':time.time()})
