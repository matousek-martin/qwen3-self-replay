"""Same four-answer replay update split across 2/4 GPUs; no larger global batch.

Rank-local gradients are SUM-reduced because each rank handles a disjoint part
of the historical global loss. Supervised replay contributes on rank zero only.
"""
import argparse
import hashlib
import json
import os
import random
import signal
import statistics
import time
from pathlib import Path
from .core import DOMAINS, MODEL, REVISION, admission, checkpoint_boundaries, policy_rewards, sha, verify_worker, write
from .metrics import summarize
from .generation import batches, generate_batch, validate_settings


def reduce_gradients(parameters, dist, torch):
    for p in parameters:
        if p.grad is None:p.grad=torch.zeros_like(p)
        dist.all_reduce(p.grad,op=dist.ReduceOp.SUM)


def main():
    import numpy as np
    import torch
    import torch.distributed as dist
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, PeftModel
    from experiments.rl_training.train import token_logprobs
    from experiments.rl_training.objectives import reasoning_is_complete
    from experiments.shorterbetter.multidomain.rewards import verify, code_check
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--pilot',action='store_true');p.add_argument('--resume',type=Path)
    p.add_argument('--benchmark-generation',action='store_true',help='Inference-only timing on registered training prompts; no optimizer steps')
    p.add_argument('--stop-after',type=int,help='Technical split-run boundary; not a quality stop')
    args=p.parse_args();cfg=json.loads((args.data/'plan.json').read_text())
    if args.benchmark_generation and (args.pilot or args.resume):raise ValueError('Benchmark is separate from training and resume')
    generation=validate_settings(cfg.get('generation'))
    contract=sha(args.data/'plan.json');rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE'])
    lease=json.loads(Path('worker-lease.json').read_text())
    names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    verify_worker(names,lease)
    if world!=lease['gpu_count']:raise RuntimeError('torchrun world size differs from lease')
    from datetime import timedelta
    from .communication import PROTOCOL as COMMUNICATION_PROTOCOL, gather_records
    torch.cuda.set_device(rank);device=torch.device('cuda',rank)
    dist.init_process_group('nccl',device_id=device)
    records_group=dist.new_group(backend='gloo',timeout=timedelta(minutes=60))
    torch.manual_seed(cfg['seed']);random.seed(cfg['seed']);np.random.seed(cfg['seed'])
    torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    for file,expected in cfg['files_sha256'].items():
        if sha(args.data/file)!=expected:raise RuntimeError('Frozen data hash changed')
    train=[json.loads(x) for x in (args.data/'train.jsonl').open()]
    dev=[json.loads(x) for x in (args.data/'development.jsonl').open()]
    if args.pilot:
        from .pilot import fixtures
        train=fixtures();dev=fixtures() # Deliberate reuse for technical checks only; not held-out accuracy.
    total=len(train);out=args.out;out.mkdir(parents=True,exist_ok=True)
    if not args.resume and (out/'progress.json').exists():raise RuntimeError('Output exists; explicit resume required')
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION)
    base=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,torch_dtype=torch.bfloat16,attn_implementation='sdpa').to(device)
    if args.resume:
        manifest=json.loads((args.resume/'complete.json').read_text())
        if manifest['contract']!=contract or manifest['world_size']!=world or manifest['pilot']!=args.pilot:
            raise RuntimeError('Resume execution contract differs')
        for file,digest in manifest['files'].items():
            if sha(args.resume/file)!=digest:raise RuntimeError('Incomplete/corrupted checkpoint')
        model=PeftModel.from_pretrained(base,str(args.resume/'adapter'),is_trainable=True)
    else:
        model=get_peft_model(base,LoraConfig(r=16,lora_alpha=32,lora_dropout=0.,bias='none',task_type='CAUSAL_LM',
                   target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']))
    model.enable_input_require_grads();model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.config.use_cache=False
    params=[p for p in model.parameters() if p.requires_grad]
    eos=model.generation_config.eos_token_id
    eos={eos} if isinstance(eos,int) else set(eos)
    pad_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    generation_cap=256 if args.pilot else cfg['max_new_tokens']
    write(out/f'generation-rank-{rank}.json',dict(**generation,world_size=world,
          local_training_answers=len(range(rank,4,world)),generation_token_cap=generation_cap,
          communication_protocol=COMMUNICATION_PROTOCOL))
    if any(p.dtype!=torch.float32 for p in params):raise RuntimeError('LoRA trainable weights must be FP32')
    optimizer=torch.optim.AdamW(params,lr=cfg['learning_rate'],weight_decay=0)
    buffer={};replay_rng=random.Random(cfg['seed']+104729);updates=0;done=0;references={};baseline=[];curve=[];series={d:[] for d in ('overall',)+DOMAINS}
    started=time.time();elapsed_prior=0.;stop=[False]
    for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    if args.resume:
        # Checkpoint is produced and SHA-verified by this registered worker; never load untrusted pickle.
        saved=torch.load(args.resume/'state.pt',map_location='cpu',weights_only=False)
        optimizer.load_state_dict(saved['optimizer']);buffer=saved['buffer'];replay_rng.setstate(saved['replay_rng'])
        done=saved['done'];updates=saved['updates'];references=saved['references'];baseline=saved['baseline'];curve=saved['curve'];series=saved['series'];elapsed_prior=saved['elapsed']
        rng=torch.load(args.resume/f'rng-{rank}.pt',weights_only=False)
        random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.set_rng_state(rng['torch']);torch.cuda.set_rng_state(rng['cuda'])
    def progress(phase,**extra):
        if rank==0:write(out/'progress.json',dict(phase=phase,current=done,total=total,optimizer_updates=updates,
                   elapsed=elapsed_prior+time.time()-started,updated_at=time.time(),gpu_count=world,pilot=args.pilot,**extra))
    def gather(rows):
        return gather_records(rows,dist,records_group)
    prompt_cache={}
    def prompt_tokens(row):
        if row['id'] not in prompt_cache:
            ids=tokenizer.apply_chat_template(row['prompt'],tokenize=True,add_generation_prompt=True,enable_thinking=True)
            if not ids or len(ids)>2048:raise RuntimeError('Registered question exceeds prompt cap; refusing silent skip')
            prompt_cache[row['id']]=ids
        return prompt_cache[row['id']]
    def prompt(row):
        return torch.tensor([prompt_tokens(row)],device=device)
    def record(row,sample_id,seq):
        truncated=not seq or seq[-1] not in eos;text=tokenizer.decode(seq,skip_special_tokens=True)
        correct=bool(not truncated and reasoning_is_complete(text) and verify(text,row['answer']))
        return dict(id=row['id'],task=row['task'],sample=sample_id,ids=seq,correct=correct,total_tokens=len(seq),truncated=truncated,text=text)
    def sample_batch(requests):
        sequences=generate_batch(model,[prompt_tokens(row) for row,_,_ in requests],
                                 [seed for _,_,seed in requests],pad_token_id=pad_id,eos_token_ids=eos,
                                 max_new_tokens=generation_cap,temperature=cfg['temperature'],
                                 top_p=cfg['top_p'],top_k=cfg['top_k'])
        return [record(row,sample_id,seq) for (row,sample_id,_),seq in zip(requests,sequences)]
    def evaluate(rows,phase,batch_size=None):
        began=time.time();local=[];requests=[];samples=1 if args.pilot else 4
        progress(phase,validation_completed=0,validation_total=len(rows)*samples)
        for i,row in enumerate(rows):
            for sample_id in range(samples):
                if (i*samples+sample_id)%world!=rank:continue
                seed=int(hashlib.sha256(f"validation:{row['id']}:{sample_id}:9127".encode()).hexdigest()[:8],16)
                requests.append((row,sample_id,seed))
        batch_count=0
        batch_size=batch_size or generation['evaluation_batch_size']
        for indices in batches([len(prompt_tokens(row)) for row,_,_ in requests],
                               batch_size,generation['max_padded_tokens'],generation_cap):
            local.extend(sample_batch([requests[i] for i in indices]));batch_count+=1
            # Rank-zero local progress is labelled explicitly, not multiplied into a false global count.
            progress(phase,validation_rank_completed=len(local),validation_rank_total=len(requests),
                     validation_total=len(rows)*samples,generation_batches=batch_count)
        # Preserve expensive rank-local evaluation output before any communication.
        local=[{k:v for k,v in r.items() if k!='ids'} for r in local]
        write(out/'validation-shards'/f'{phase}-{done:06d}-rank-{rank}.json',local)
        result=sorted(gather(local),key=lambda r:(r['id'],r['sample']))
        if rank==0:
            progress(phase,validation_completed=len(result),validation_total=len(rows)*samples)
            with (out/'evaluation-timing.jsonl').open('a') as f:f.write(json.dumps(dict(phase=phase,group=done,questions=len(rows),seconds=time.time()-began,
                 local_batches=batch_count,batch_size=batch_size,generation_token_cap=generation_cap,generation_protocol=generation['protocol']))+'\n')
        return [{k:v for k,v in r.items() if k!='ids'} for r in result]
    def publish_validation():
        nonlocal curve
        records=evaluate(dev,'baseline' if done==0 else 'validation')
        metrics=summarize(records,baseline if baseline else None)
        value=dict(iteration=done,completed_at=time.time(),metrics=metrics,protocol='technical-fixture-not-held-out' if args.pilot else f'development-{len(dev)}x4',official_evaluation=False)
        curve=[x for x in curve if x['iteration']!=done]+[value]
        if rank==0:
            file=out/'validations'/f'{done:06d}.jsonl';file.parent.mkdir(exist_ok=True)
            file.write_text(''.join(json.dumps(r)+'\n' for r in records))
            write(file.with_suffix('.json'),value);write(out/'validation-curve.json',dict(checkpoints=curve))
        return records
    def checkpoint():
        dest=out/'checkpoints'/f'group-{done:06d}'
        if (dest/'complete.json').exists():return
        dest.mkdir(parents=True,exist_ok=True)
        torch.save(dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state()),dest/f'rng-{rank}.pt')
        dist.barrier()
        if rank==0:
            model.save_pretrained(dest/'adapter',safe_serialization=True);tokenizer.save_pretrained(dest/'adapter')
            torch.save(dict(optimizer=optimizer.state_dict(),buffer=buffer,replay_rng=replay_rng.getstate(),done=done,updates=updates,
                     references=references,baseline=baseline,curve=curve,series=series,elapsed=elapsed_prior+time.time()-started,
                     scheduler='constant_learning_rate_no_scheduler',sampler='frozen_balanced_order_cursor_is_done'),dest/'state.pt')
            write(dest/'complete.json',dict(contract=contract,world_size=world,pilot=args.pilot,group=done,
                  files={str(p.relative_to(dest)):sha(p) for p in dest.rglob('*') if p.is_file()}))
            write(out/'latest-checkpoint.json',dict(group=done,path=str(dest),manifest_sha256=sha(dest/'complete.json')))
        dist.barrier()
    progress('initializing')
    if args.benchmark_generation:
        # A tiny balanced training-only slice: no benchmark leakage and no weight updates.
        spec=cfg.get('generation_benchmark',dict(questions=8,stages=[(256,1),(256,2),(256,16),(cfg['max_new_tokens'],2),(cfg['max_new_tokens'],16)]))
        if type(spec['questions']) is not int or not 4<=spec['questions']<=32 or spec['questions']%4:raise ValueError('Invalid bounded benchmark size')
        if not 1<=len(spec['stages'])<=5:raise ValueError('Invalid bounded benchmark stages')
        if any(cap not in (256,cfg['max_new_tokens']) or not 1<=batch_size<=generation['evaluation_batch_size'] for cap,batch_size in spec['stages']):
            raise ValueError('Benchmark exceeds generation contract')
        rows=train[:spec['questions']];results=[]
        for cap,batch_size in spec['stages']:
            generation_cap=cap
            dist.barrier();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();began=time.time()
            records=evaluate(rows,f'generation_benchmark_cap{cap}_batch{batch_size}',batch_size)
            torch.cuda.synchronize();seconds=time.time()-began
            duration=torch.tensor(seconds,device=device);dist.all_reduce(duration,op=dist.ReduceOp.MAX)
            memory=torch.tensor(torch.cuda.max_memory_allocated(),device=device);dist.all_reduce(memory,op=dist.ReduceOp.MAX)
            result=dict(cap=cap,batch_size=batch_size,seconds=duration.item(),responses=len(records),
                        tokens=sum(r['total_tokens'] for r in records),training_updates=0,peak_allocated_bytes=memory.item(),
                        per_answer=[dict(id=r['id'],sample=r['sample'],total_tokens=r['total_tokens'],truncated=r['truncated'],
                                        text_sha256=hashlib.sha256(r['text'].encode()).hexdigest()) for r in records])
            results.append(result)
            if rank==0:write(out/'generation-benchmark.json',dict(complete=False,technical_only=True,
                 generation_protocol=generation['protocol'],world_size=world,questions=[r['id'] for r in rows],results=results))
        if rank==0:write(out/'generation-benchmark.json',dict(complete=True,technical_only=True,
             generation_protocol=generation['protocol'],world_size=world,questions=[r['id'] for r in rows],results=results))
        progress('completed');dist.barrier();dist.destroy_process_group();return
    if args.pilot and not args.resume:
        progress('maximum_length_memory_check')
        model.train();ids=prompt(train[0])
        tokens=torch.full((1,cfg['max_new_tokens']),tokenizer.eos_token_id,device=device,dtype=torch.long)
        lp,_=token_logprobs(model,ids,tokens,64,.6)
        (-lp.mean()).backward()
        torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
        optimizer.zero_grad(set_to_none=True)
        if rank==0:write(out/'maximum-length-check.json',dict(passed=True,tokens=cfg['max_new_tokens'],optimizer_update=False))
    if rank==0 and not code_check('print(input())',{'inputs':['ok'],'outputs':['ok']}):raise RuntimeError('Code verifier smoke failed')
    dist.barrier()
    if not baseline:
        baseline=publish_validation()
        calibration=train[:4 if args.pilot else cfg['reference_questions_per_domain']*4]
        calibration_records=baseline if args.pilot else evaluate(calibration,'reference_calibration')
        references={d:statistics.mean(r['total_tokens'] for r in calibration_records if r['task']==d) for d in DOMAINS}
        if rank==0:write(out/'references.json',dict(references=references,questions=[r['id'] for r in calibration],base_revision=REVISION))
        checkpoint()
    recent=[]
    for index in range(done,total):
        deadline=lease.get('hard_deadline') or lease.get('deadline') or float('inf')
        flag=torch.tensor(int(stop[0] or time.time()>deadline-600),device=device);dist.all_reduce(flag,op=dist.ReduceOp.MAX)
        if flag.item():break
        row=train[index];progress('training',current_question=row['task']);begin=time.time();ids=prompt(row)
        group=gather(sample_batch([(row,s,cfg['seed']+index*4+s) for s in range(rank,4,world)]))
        group.sort(key=lambda r:r['sample'])
        reward,advantage=policy_rewards([r['correct'] for r in group],[r['total_tokens'] for r in group],references[row['task']])
        admission(buffer,row['id'],ids[0].tolist(),group)
        optimizer.zero_grad(set_to_none=True);model.train()
        for sample_id in range(rank,4,world):
            if advantage[sample_id]==0:continue
            completion=torch.tensor([group[sample_id]['ids']],device=device)
            lp,_=token_logprobs(model,ids,completion,64,.6)
            loss=(-(lp-lp.detach()).exp()*advantage[sample_id]).sum()/(4*cfg['max_new_tokens'])
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite policy loss')
            loss.backward()
        # All ranks advance the replay RNG identically; only rank zero contributes this loss.
        replay_id=replay_rng.choice(list(buffer)) if buffer else None
        if replay_id and rank==0:
            pr,co=buffer[replay_id];lp,_=token_logprobs(model,torch.tensor([pr],device=device),torch.tensor([co],device=device),64,1.)
            loss=-cfg['replay_weight']*lp.mean()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite replay loss')
            loss.backward()
        if any(a!=0 for a in advantage) or replay_id:
            reduce_gradients(params,dist,torch);norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
            optimizer.step();updates+=1
            if any(not torch.isfinite(p).all() for p in params):raise FloatingPointError('Nonfinite parameter')
        done=index+1;recent.extend(group)
        if rank==0:
            with (out/'training-records.jsonl').open('a') as f:
                for r,rw in zip(group,reward):f.write(json.dumps({**{k:v for k,v in r.items() if k!='ids'},'group':done,'reward':rw})+'\n')
            with (out/'timing.jsonl').open('a') as f:f.write(json.dumps(dict(group=done,seconds=time.time()-begin,updates=updates,replay_size=len(buffer)))+'\n')
        if done%16==0 or done in cfg['milestones'] or done==total:
            for d,v in summarize(recent).items():series[d].append(dict(iteration=done,**v))
            recent=[]
            if rank==0:write(out/'training-series.json',dict(series=series))
        save_due,validation_due=checkpoint_boundaries(done,total,cfg,pilot=args.pilot,stop_after=args.stop_after)
        if validation_due:publish_validation()
        if save_due:checkpoint()
        progress('training')
        if done==args.stop_after:break
    if not (out/'checkpoints'/f'group-{done:06d}'/'complete.json').exists():checkpoint()
    progress('completed' if done==total else 'stopped')
    if rank==0:write(out/'completion.json',dict(complete=done==total,groups=done,updates=updates,pilot=args.pilot,finished_at=time.time()))
    dist.barrier();dist.destroy_process_group()

if __name__=='__main__':main()
