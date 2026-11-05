"""Paired development-only thinking-mode diagnostic with cached on-mode records."""
import hashlib
import json
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor
from .runtime import read,rows,write,sha
from .mode_budget import SCOPE,verify_worker
from experiments.replay_scaled.metrics import summarize
DATA=Path('data/opencoder-mode-probe');OUT=Path('output/opencoder-mode-probe')

def pair_check(records,questions):
    expected={(q['id'],s) for q in questions for s in range(2)}
    observed=[(r['id'],r['sample']) for r in records]
    if len(observed)!=len(set(observed)) or set(observed)!=expected:raise ValueError('Incomplete or mismatched paired records')
    byid={q['id']:q for q in questions}
    if any(r['task']!=byid[r['id']]['task'] or r['cohort']!=byid[r['id']]['cohort'] for r in records):raise ValueError('Domain/cohort mismatch')

def nonthinking_complete(text):
    # The prompt already contains an EMPTY closed reasoning block in this mode.
    return bool(text.strip()) and '<think>' not in text and '</think>' not in text

def metrics(records,reference=None):
    old=[r for r in records if r['cohort']=='original-development']
    ref=[r for r in reference if r['cohort']=='original-development'] if reference is not None else None
    value=summarize(old,ref)
    oc=[r for r in records if r['cohort']=='opencoder-development']
    oref=[r for r in reference if r['cohort']=='opencoder-development'] if reference is not None else None
    value['opencoder']=summarize(oc,oref)['coding']
    return value

def result_sets(sets,questions):
    for records in sets.values():pair_check(records,questions)
    comparisons={
        'sft_effect_thinking_on':metrics(sets['sft16-on'],sets['parent768-on']),
        'sft_effect_thinking_off':metrics(sets['sft16-off'],sets['parent768-off']),
        'disable_thinking_parent':metrics(sets['parent768-off'],sets['parent768-on']),
        'disable_thinking_sft':metrics(sets['sft16-off'],sets['sft16-on'])}
    return dict(conditions={k:metrics(v) for k,v in sets.items()},comparisons=comparisons,
        inference_only=True,official_evaluation=False,questions=192,samples=2,
        reused_conditions=['parent768-on','sft16-on'],new_conditions=['parent768-off','sft16-off'],
        limitations=['Previously inspected development cohort; not an untouched holdout.',
                    'Thinking-off intentionally changes only prompt mode; official evaluation protocol unchanged.',
                    'Prompt-cluster intervals describe sampling within this cohort, not training-seed uncertainty.',
                    'Token counts are generated tokens; fixed empty-think prompt prefix is not generated output.'])

def main():
    import torch
    import importlib.metadata
    from transformers import AutoTokenizer
    from vllm import LLM,SamplingParams
    from vllm.lora.request import LoRARequest
    from experiments.replay_scaled.core import MODEL,REVISION
    from experiments.shorterbetter.multidomain.rewards import verify,code_check
    OUT.mkdir(parents=True,exist_ok=True);cfg=read(DATA/'plan.json');lease=read('worker-lease.json')
    verify_worker([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],lease)
    if cfg['scope']!=SCOPE or cfg['training_authorized'] or cfg['enable_thinking'] is not False:raise RuntimeError('Probe contract changed')
    for f,h in cfg['files_sha256'].items():
        if sha(DATA/f)!=h:raise RuntimeError('Frozen diagnostic input changed: '+f)
    versions={k:importlib.metadata.version(k) for k in cfg['runtime_versions']}
    if versions!=cfg['runtime_versions']:raise RuntimeError('Runtime differs from cached thinking-on records')
    questions=rows(DATA/'questions.jsonl')
    sets={k:rows(DATA/(k+'.jsonl')) for k in ['parent768-on','sft16-on']}
    for v in sets.values():pair_check(v,questions)
    if len(questions)!=192:raise RuntimeError('Unexpected cohort')
    if not code_check('print(2)',{'inputs':[''],'outputs':['2']}) or code_check('print(3)',{'inputs':[''],'outputs':['2']}):raise RuntimeError('Sandbox unavailable')
    tokenizer=AutoTokenizer.from_pretrained(DATA/'adapters/parent768',local_files_only=True)
    engine=LLM(model=MODEL,revision=REVISION,tokenizer_revision=REVISION,dtype='bfloat16',max_model_len=4096,
        max_num_seqs=32,gpu_memory_utilization=.75,enable_lora=True,max_lora_rank=16,max_loras=1,max_cpu_loras=2,
        enable_prefix_caching=False,enforce_eager=True,generation_config='vllm',seed=9127)
    started=time.time();completed=0
    write(OUT/'preflight.json',dict(gpu=torch.cuda.get_device_name(),versions=versions,inference_only=True,source_hashes_verified=True,
        cached_pairs_verified=True,enable_thinking=False,model_revision=REVISION))
    try:
        for route_id,model in enumerate(['parent768','sft16'],1):
            condition=model+'-off';path=OUT/(condition+'.jsonl')
            if path.exists():raise RuntimeError('Refusing to overwrite generated answers')
            adapter=DATA/'adapters'/model;route=LoRARequest(model,route_id,str(adapter.resolve()));engine.llm_engine.add_lora(route)
            requests=[(q,s) for q in questions for s in range(2)];records=[]
            for offset in range(0,384,32):
                live_lease=read('worker-lease.json')
                if live_lease.get('stop_requested') or time.time()>live_lease['hard_deadline']-300:raise RuntimeError('Budget/user stop; partial evidence preserved')
                batch=requests[offset:offset+32]
                ids=[tokenizer.apply_chat_template(q['prompt'],tokenize=True,add_generation_prompt=True,enable_thinking=False) for q,s in batch]
                if any(len(t)+2048>4096 for t in ids):raise RuntimeError('Prompt would be truncated')
                if any(not tokenizer.decode(t).endswith('<think>\n\n</think>\n\n') for t in ids):raise RuntimeError('Thinking-off prefix not applied')
                seeds=[int(hashlib.sha256(f"opencoder-dev:{q['id']}:{s}:9127".encode()).hexdigest()[:8],16) for q,s in batch]
                params=[SamplingParams(n=1,max_tokens=2048,temperature=.6,top_p=.95,top_k=20,seed=s) for s in seeds]
                answers=engine.generate([{'prompt_token_ids':t} for t in ids],params,lora_request=route,use_tqdm=False)
                if any(a.lora_request is None or a.lora_request.lora_int_id!=route_id for a in answers):raise RuntimeError('Wrong adapter route')
                def grade(item):
                    q,s,a,seed,prompt=item;o=a.outputs[0];text=o.text;truncated=o.finish_reason!='stop'
                    return dict(id=q['id'],task=q['task'],cohort=q['cohort'],sample=s,text=text,total_tokens=len(o.token_ids),
                        truncated=truncated,correct=bool(not truncated and nonthinking_complete(text) and verify(text,q['answer'])),
                        condition=condition,enable_thinking=False,seed=seed,prompt_sha256=hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
                        generated_token_ids=list(o.token_ids))
                with ThreadPoolExecutor(max_workers=4) as pool:new=list(pool.map(grade,[(q,s,a,seed,prompt) for (q,s),a,seed,prompt in zip(batch,answers,seeds,ids)]))
                records.extend(new)
                with path.open('a') as f:
                    for v in new:f.write(json.dumps(v)+'\n')
                completed+=len(new);write(OUT/'progress.json',dict(phase='inference',condition=condition,current=completed,total=768,updated_at=time.time(),elapsed=time.time()-started))
                print(f'{condition}: {len(records)}/384; total {completed}/768',flush=True)
            pair_check(records,questions);sets[condition]=records
            write(OUT/(condition+'-metrics.json'),metrics(records))
        result=result_sets(sets,questions);result['provenance']=cfg['provenance'];result['elapsed']=time.time()-started
        write(OUT/'results.json',result)
        write(OUT/'completion.json',dict(complete=True,generated_responses=768,reused_responses=768,training_updates=0,
            results_sha256=sha(OUT/'results.json'),generated_files_sha256={k+'.jsonl':sha(OUT/(k+'.jsonl')) for k in ['parent768-off','sft16-off']}))
        write(OUT/'progress.json',dict(phase='complete',current=768,total=768,updated_at=time.time(),elapsed=time.time()-started))
    except Exception as e:
        write(OUT/'failure.json',dict(error=str(e),detected_at=time.time()));raise

if __name__=='__main__':main()
