"""Assistant-token SFT with complete deterministic checkpoint/resume on one4090."""
import argparse
import math
import os
from pathlib import Path
import random
import time
from .runtime import DATA,read,rows,sha,write,encode,check_inputs,check_lease,verify_worker


def main():
    import numpy as np
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer,AutoModelForCausalLM,get_cosine_schedule_with_warmup
    from peft import PeftModel
    from experiments.replay_scaled.core import MODEL,REVISION
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path('output/training'));p.add_argument('--until',type=int,required=True)
    p.add_argument('--resume',type=Path);p.add_argument('--technical',action='store_true');a=p.parse_args()
    cfg=check_inputs();lease=check_lease();verify_worker([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],lease)
    total=2 if a.technical else cfg['optimizer_steps']
    if not 0<a.until<=total:raise ValueError('Invalid boundary')
    a.out.mkdir(parents=True,exist_ok=True)
    contract=sha(DATA/'plan.json')
    torch.manual_seed(17);np.random.seed(17);random.seed(17)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    tok=AutoTokenizer.from_pretrained(DATA/'adapter',local_files_only=True)
    schedule=rows(DATA/'train-schedule.jsonl');encoded=[encode(tok,r) for r in schedule]
    if a.technical:
        # Two updates of32, including the longest scheduled sequence, test actual memory.
        encoded=sorted(encoded,key=lambda r:len(r['ids']),reverse=True)[:64]
    if len(encoded)!=total*32:raise ValueError('Dataset/batch contract differs')
    base=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda')
    cp=a.resume
    if cp:
        m=read(cp/'complete.json')
        if m['contract']!=contract or m['technical']!=a.technical:raise RuntimeError('Checkpoint contract mismatch')
        for f,h in m['files'].items():
            if sha(cp/f)!=h:raise RuntimeError('Checkpoint hash mismatch')
    model=PeftModel.from_pretrained(base,cp/'adapter' if cp else DATA/'adapter',is_trainable=True)
    model.enable_input_require_grads();model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.config.use_cache=False;model.train()
    parameters=[v for v in model.parameters() if v.requires_grad]
    if not parameters or any(v.dtype!=torch.float32 for v in parameters):raise RuntimeError('Expected FP32 trainable LoRA')
    optimizer=torch.optim.AdamW(parameters,lr=1e-5,weight_decay=0)
    scheduler=get_cosine_schedule_with_warmup(optimizer,num_warmup_steps=5,num_training_steps=84)
    step=0
    if cp:
        state=torch.load(cp/'state.pt',map_location='cpu',weights_only=False)
        optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler']);step=state['step']
        random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state(state['cuda_rng'])
    def save():
        dest=a.out/'checkpoints'/f'step-{step:06d}'
        if dest.exists():raise RuntimeError('Refusing checkpoint overwrite')
        dest.mkdir(parents=True)
        model.save_pretrained(dest/'adapter',safe_serialization=True);tok.save_pretrained(dest/'adapter')
        torch.save(dict(optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),step=step,
            python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),dest/'state.pt')
        write(dest/'complete.json',dict(contract=contract,step=step,technical=a.technical,world_size=1,
              files={str(f.relative_to(dest)):sha(f) for f in dest.rglob('*') if f.is_file()}))
        write(a.out/'latest-checkpoint.json',dict(group=step,step=step,path=str(dest),manifest_sha256=sha(dest/'complete.json')))
        return dest
    began=time.time();times=[];torch.cuda.reset_peak_memory_stats()
    while step<a.until:
        check_lease();batch=encoded[step*32:(step+1)*32];denom=sum(r['target_tokens'] for r in batch)
        optimizer.zero_grad(set_to_none=True);loss_sum=0.;start=time.time()
        for i in range(0,32,2):
            micro=batch[i:i+2];n=max(len(r['ids']) for r in micro)
            ids=torch.tensor([r['ids']+[tok.pad_token_id]*(n-len(r['ids'])) for r in micro],device='cuda')
            mask=torch.tensor([[1]*len(r['ids'])+[0]*(n-len(r['ids'])) for r in micro],device='cuda')
            labels=torch.tensor([r['labels']+[-100]*(n-len(r['labels'])) for r in micro],device='cuda')
            output=model(input_ids=ids,attention_mask=mask,use_cache=False)
            loss=F.cross_entropy(output.logits[:,:-1].float().reshape(-1,output.logits.shape[-1]),labels[:,1:].reshape(-1),ignore_index=-100,reduction='sum')/denom
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite supervised loss')
            loss_sum+=loss.item();loss.backward();del loss,output
        norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
        optimizer.step();scheduler.step();step+=1;torch.cuda.synchronize();seconds=time.time()-start;times.append(seconds)
        metrics=dict(step=step,loss=loss_sum,gradient_norm=float(norm),lr=scheduler.get_last_lr()[0],seconds=seconds,
                     training_rows=step*32,coding_distinct=len({r['id'] for r in schedule[:step*32] if r['source'].startswith('OpenCoder')}),
                     peak_memory_bytes=torch.cuda.max_memory_allocated(),updated_at=time.time())
        with (a.out/'training-metrics.jsonl').open('a') as f:f.write(__import__('json').dumps(metrics)+'\n')
        write(a.out/'progress.json',dict(phase='training',current=step,total=total,gpu_count=1,**metrics))
        print(f'Update {step}/{total}: loss={loss_sum:.5f}, {seconds:.2f}s',flush=True)
        if a.technical or step==a.until:save()
    write(a.out/'segment-completion.json',dict(complete=True,step=step,total=total,seconds=time.time()-began,
        seconds_per_update=sum(times)/len(times) if times else None,peak_memory_bytes=torch.cuda.max_memory_allocated(),technical=a.technical))


if __name__=='__main__':main()
