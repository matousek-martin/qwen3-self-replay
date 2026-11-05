"""Frozen checkpoint, paired coding decoding caps; no optimizer or training."""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import random
import time
import urllib.request

from .core import MODEL, REVISION, sha, write
from .length_budget import verify_worker
from .generation import RowSeededSampler, trim_completion, generate_batch
from experiments.rl_training.objectives import reasoning_is_complete
from experiments.shorterbetter.multidomain.rewards import code_check, verify

DATA = Path('data/coding-length-probe')
OUT = Path('output/coding-length-probe')
CAPS = [2048, 4096, 8192]


def cut(tokens, cap, eos):
    result = trim_completion(tokens[:cap], eos)
    return result, not result or result[-1] not in eos


def summarize(records, timing):
    result=[]
    for difficulty in ['all','introductory','interview']:
        subset=[r for r in records if difficulty=='all' or r['difficulty']==difficulty]
        for cap in CAPS:
            rs=[r for r in subset if r['cap']==cap]
            if not rs: continue
            groups=collections.defaultdict(list)
            for r in rs: groups[r['id']].append(r)
            result.append(dict(difficulty=difficulty, cap=cap, answers=len(rs), questions=len(groups),
                correct=sum(r['correct'] for r in rs), accuracy=sum(r['correct'] for r in rs)/len(rs),
                truncated=sum(r['truncated'] for r in rs),
                mean_tokens=sum(r['total_tokens'] for r in rs)/len(rs),
                zero_of_four=sum(not any(r['correct'] for r in g) for g in groups.values()),
                four_of_four=sum(len(g)==4 and all(r['correct'] for r in g) for g in groups.values())))
    pairs=[]
    for difficulty in ['all','introductory','interview']:
        for low,high in [(2048,4096),(4096,8192),(2048,8192)]:
            rs=[r for r in records if difficulty=='all' or r['difficulty']==difficulty]
            by_key={(r['id'],r['sample'],r['cap']):r for r in rs}
            groups=collections.defaultdict(list)
            for r in rs:
                if r['cap']!=high: continue
                prev=by_key[(r['id'],r['sample'],low)]
                groups[r['id']].append(int(r['correct'])-int(prev['correct']))
            if not groups: continue
            means=[sum(g)/len(g) for g in groups.values()];rng=random.Random(1701)
            draws=sorted(sum(rng.choices(means,k=len(means)))/len(means) for _ in range(2000))
            pairs.append(dict(difficulty=difficulty,lower_cap=low,upper_cap=high,
                accuracy_delta=sum(means)/len(means),paired_question_bootstrap_ci95=[draws[49],draws[1949]]))
    return dict(metrics=result, paired_deltas=pairs,
        estimated_separate_decode_seconds={str(cap):sum(t['cap_seconds'][str(cap)] for t in timing) for cap in CAPS},
        timing_note='Paired prefixes of one long decode; cumulative batch timing approximates separate-cap generation, excludes setup/grading and is not a training throughput measurement.')


def preload():
    m=json.loads((DATA/'model-preload.json').read_text())
    if m['model']!=MODEL or m['revision']!=REVISION: raise ValueError('Unpinned model')
    folder=Path(os.environ['HF_HOME'])/'hub/models--Qwen--Qwen3-0.6B/snapshots'/REVISION
    folder.mkdir(parents=True,exist_ok=True)
    for name,expected in m['files_sha256'].items():
        if Path(name).name!=name: raise ValueError('Invalid model filename')
        target=folder/name
        if target.exists() and sha(target)==expected: continue
        temp=target.with_suffix(target.suffix+'.download')
        with urllib.request.urlopen(f'https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}',timeout=120) as response,temp.open('wb') as out:
            while block:=response.read(8*1024*1024): out.write(block)
        if sha(temp)!=expected: raise RuntimeError('Preload SHA mismatch')
        temp.replace(target)
    write(OUT/'model-verification.json',dict(**m,verified=True))


def main():
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, LogitsProcessorList, StoppingCriteria, StoppingCriteriaList
    cfg=json.loads((DATA/'plan.json').read_text()); lease=json.loads(Path('worker-lease.json').read_text())
    verify_worker([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],lease)
    if cfg['caps']!=CAPS or cfg['training_authorized'] is not False or cfg['checkpoint_group']!=850: raise ValueError('Invalid probe contract')
    for filename,expected in cfg['files_sha256'].items():
        if sha(DATA/filename)!=expected: raise ValueError('Input SHA mismatch: '+filename)
    rows=[json.loads(l) for l in (DATA/'questions.jsonl').open()]
    if len(rows)!=47 or collections.Counter(r.get('difficulty') for r in rows)!={'introductory':22,'interview':25}: raise ValueError('Unregistered cohort')
    if any(r['task']!='coding' or r['split'] not in ('development','dev','validation') for r in rows): raise ValueError('Only separate development coding questions permitted')
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'records.jsonl').exists(): raise RuntimeError('Do not overwrite partial results')
    begin=time.time(); torch.manual_seed(cfg['seed']);torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    model_path=Path(os.environ['HF_HOME'])/'hub/models--Qwen--Qwen3-0.6B/snapshots'/REVISION
    tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
    base=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda')
    model=PeftModel.from_pretrained(base,DATA/'adapter',is_trainable=False);model.eval()
    model.requires_grad_(False)
    assert not any(p.requires_grad for p in model.parameters())
    adapter_before=sha(DATA/'adapter/adapter_model.safetensors')
    eos=model.generation_config.eos_token_id or tokenizer.eos_token_id;eos={eos} if isinstance(eos,int) else set(eos)
    pad=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    prompts={r['id']:tokenizer.apply_chat_template(r['prompt'],tokenize=True,add_generation_prompt=True,enable_thinking=True) for r in rows}
    if any(not v or len(v)>2048 for v in prompts.values()): raise RuntimeError('Prompt cap differs from training contract')
    sampling=dict(temperature=cfg['temperature'],top_k=cfg['top_k'],top_p=cfg['top_p'])
    smoke_prompts=[prompts[r['id']] for r in rows[:2]];smoke_seeds=[717,718]
    small=generate_batch(model,smoke_prompts,smoke_seeds,pad_token_id=pad,eos_token_ids=eos,max_new_tokens=64,**sampling)
    large=generate_batch(model,smoke_prompts,smoke_seeds,pad_token_id=pad,eos_token_ids=eos,max_new_tokens=128,**sampling)
    if any(a!=trim_completion(b[:64],eos) for a,b in zip(small,large)): raise RuntimeError('Paired cap prefix verification failed')
    if not code_check('print(2)',{'inputs':[''],'outputs':['2']}): raise RuntimeError('Code sandbox positive test failed')
    if code_check('print(3)',{'inputs':[''],'outputs':['2']}): raise RuntimeError('Code sandbox negative test failed')
    write(OUT/'preflight.json',dict(passed=True,prefix_check=True,sandbox_check=True,trainable_parameters=0,gpu=torch.cuda.get_device_name(),checkpoint_group=850))
    requests=[(row,sample,cfg['seed']+i*4+sample) for i,row in enumerate(rows) for sample in range(4)]
    records=[];timing=[];grades={}

    def progress(phase,done,**extra):
        write(OUT/'progress.json',dict(phase=phase,answers_completed=done,answers_total=len(requests),checkpoint_group=850,caps=CAPS,updated_at=time.time(),elapsed=time.time()-begin,**extra))

    for offset in range(0,len(requests),cfg['batch_size']):
        if time.time()>lease['hard_deadline']-600: raise RuntimeError('Probe deadline reached; preserve partial results')
        batch=requests[offset:offset+cfg['batch_size']];width=max(len(prompts[r['id']]) for r,_,_ in batch)
        ids=torch.full((len(batch),width),pad,dtype=torch.long);mask=torch.zeros_like(ids)
        for i,(r,_,_) in enumerate(batch):
            seq=prompts[r['id']];ids[i,-len(seq):]=torch.tensor(seq);mask[i,-len(seq):]=1
        ticks={};torch.cuda.synchronize();started=time.monotonic()
        progress('generating',offset,batch=offset//cfg['batch_size']+1,batch_tokens=0)
        class Monitor(StoppingCriteria):
            def __call__(self,input_ids,scores,**kwargs):
                n=input_ids.shape[1]-width
                if n in CAPS:
                    torch.cuda.synchronize();ticks[str(n)]=time.monotonic()-started
                if n%128==0:
                    progress('generating',offset,batch=offset//cfg['batch_size']+1,batch_tokens=n)
                    if time.time()>lease['hard_deadline']-600: raise RuntimeError('Probe deadline reached during batch')
                return False
        sampler=RowSeededSampler([s for _,_,s in batch],'cuda',**sampling)
        config=GenerationConfig(do_sample=False,num_beams=1,max_new_tokens=max(CAPS),pad_token_id=pad,eos_token_id=list(eos),use_cache=True,return_dict_in_generate=False)
        with torch.inference_mode():
            output=model.generate(input_ids=ids.to('cuda'),attention_mask=mask.to('cuda'),generation_config=config,use_model_defaults=False,use_cache=True,
                logits_processor=LogitsProcessorList([sampler]),stopping_criteria=StoppingCriteriaList([Monitor()]))
        torch.cuda.synchronize();seconds=time.monotonic()-started
        sequences=[trim_completion(seq,eos) for seq in output[:,width:].tolist()]
        timer=dict(batch=offset//cfg['batch_size']+1,answers=len(batch),generation_seconds=seconds,cap_seconds={str(c):ticks.get(str(c),seconds) for c in CAPS})
        timing.append(timer)
        with (OUT/'sequences.jsonl').open('a') as f:
            for (row,sample,seed),seq in zip(batch,sequences): f.write(json.dumps(dict(id=row['id'],sample=sample,seed=seed,token_ids=seq))+'\n')
        progress('grading',offset,batch_tokens=max(map(len,sequences)))
        for (row,sample,seed),seq in zip(batch,sequences):
            for cap in CAPS:
                tokens,truncated=cut(seq,cap,eos);text=tokenizer.decode(tokens,skip_special_tokens=True)
                key=(row['id'],hashlib.sha256(text.encode()).hexdigest())
                correct=False
                if not truncated and reasoning_is_complete(text):
                    if key not in grades: grades[key]=bool(verify(text,row['answer']))
                    correct=grades[key]
                record=dict(id=row['id'],sample=sample,seed=seed,difficulty=row['difficulty'],cap=cap,correct=correct,truncated=truncated,total_tokens=len(tokens),text=text)
                records.append(record)
                with (OUT/'records.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
        write(OUT/'timing.json',timing);write(OUT/'results.json',summarize(records,timing))
        progress('generating',offset+len(batch))
        print(f'Graded {offset+len(batch)}/{len(requests)} paired coding answers',flush=True)
    if sha(DATA/'adapter/adapter_model.safetensors')!=adapter_before: raise RuntimeError('Adapter changed')
    assert len(records)==47*4*3
    write(OUT/'completion.json',dict(complete=True,questions=47,answers=188,cap_records=564,training_steps=0,checkpoint_group=850,adapter_sha256=adapter_before,elapsed=time.time()-begin,protocol=cfg))
    progress('complete',len(requests))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--preload',action='store_true');args=parser.parse_args()
    preload() if args.preload else main()
