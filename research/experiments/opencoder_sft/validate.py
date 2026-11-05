"""Independent development validation; never feeds generated answers to training."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor
from .runtime import DATA,read,rows,write,check_inputs,check_lease,verify_worker,sha
from experiments.replay_scaled.metrics import summarize


def main():
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM,SamplingParams
    from vllm.lora.request import LoRARequest
    from experiments.replay_scaled.core import MODEL,REVISION
    from experiments.shorterbetter.multidomain.rewards import verify,code_check
    from experiments.rl_training.objectives import reasoning_is_complete
    p=argparse.ArgumentParser();p.add_argument('--step',type=int,required=True);p.add_argument('--adapter',type=Path,required=True);a=p.parse_args()
    cfg=check_inputs();lease=check_lease();verify_worker([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],lease)
    out=Path('output/validation')/f'{a.step:06d}';out.mkdir(parents=True)
    if (out/'records.jsonl').exists():raise RuntimeError('Validation already exists')
    full=a.step in (0,84);questions=rows(DATA/'development.jsonl') if full else rows(DATA/'monitor.jsonl')
    tok=AutoTokenizer.from_pretrained(DATA/'adapter',local_files_only=True)
    if not code_check('print(2)',{'inputs':[''],'outputs':['2']}) or code_check('print(3)',{'inputs':[''],'outputs':['2']}):raise RuntimeError('Sandbox unavailable')
    engine=LLM(model=MODEL,revision=REVISION,tokenizer_revision=REVISION,dtype='bfloat16',max_model_len=4096,
        max_num_seqs=32,gpu_memory_utilization=.75,enable_lora=True,max_lora_rank=16,
        enable_prefix_caching=False,enforce_eager=True,generation_config='vllm',seed=9127)
    route=LoRARequest('opencoder-checkpoint',1,str(a.adapter.resolve()));engine.llm_engine.add_lora(route)
    if 1 not in engine.llm_engine.list_loras():raise RuntimeError('Adapter not registered')
    requests=[(q,s) for q in questions for s in range(2)]
    records=[];started=time.time()
    def grade(item):
        q,s,o=item;text=o.text;truncated=o.finish_reason!='stop'
        return dict(id=q['id'],task=q['task'],cohort=q['cohort'],sample=s,text=text,total_tokens=len(o.token_ids),
            truncated=truncated,correct=bool(not truncated and reasoning_is_complete(text) and verify(text,q['answer'])))
    for offset in range(0,len(requests),32):
        check_lease();batch=requests[offset:offset+32]
        ids=[tok.apply_chat_template(q['prompt'],tokenize=True,add_generation_prompt=True,enable_thinking=True) for q,s in batch]
        if any(len(t)+2048>4096 for t in ids):raise RuntimeError('Development prompt would be truncated')
        params=[SamplingParams(n=1,max_tokens=2048,temperature=.6,top_p=.95,top_k=20,
             seed=int(hashlib.sha256(f"opencoder-dev:{q['id']}:{s}:9127".encode()).hexdigest()[:8],16)) for q,s in batch]
        results=engine.generate([{'prompt_token_ids':t} for t in ids],params,lora_request=route,use_tqdm=False)
        if any(r.lora_request is None or r.lora_request.lora_int_id!=1 for r in results):raise RuntimeError('Adapter routing lost')
        with ThreadPoolExecutor(max_workers=4) as pool:new=list(pool.map(grade,[(q,s,r.outputs[0]) for (q,s),r in zip(batch,results)]))
        records.extend(new)
        with (out/'records.jsonl').open('a') as f:
            for r in new:f.write(json.dumps(r)+'\n')
        write(Path('output/training/progress.json'),dict(phase='baseline' if a.step==0 else 'validation',current=a.step,total=84,
            validation_completed=len(records),validation_total=len(requests),updated_at=time.time(),gpu_count=1))
        print(f'Validation {a.step}: {len(records)}/{len(requests)}',flush=True)
    baseline=rows('output/validation/000000/records.jsonl') if a.step else records
    monitor_ids={q['id'] for q in rows(DATA/'monitor.jsonl')}
    def metrics(selected):
        traditional=[r for r in selected if r['cohort']=='original-development']
        original_ids={r['id'] for r in traditional}
        result=summarize(traditional,[r for r in baseline if r['id'] in original_ids])
        coding=[r for r in selected if r['cohort']=='opencoder-development'];coding_ids={r['id'] for r in coding}
        result['opencoder']=summarize(coding,[r for r in baseline if r['id'] in coding_ids])['coding']
        return result
    point=dict(iteration=a.step,completed_at=time.time(),metrics=metrics([r for r in records if r['id'] in monitor_ids]),
        protocol='fixed-development-192x2',official_evaluation=False,seconds=time.time()-started,
        adapter_sha256=sha(a.adapter/'adapter_model.safetensors'),records_sha256=sha(out/'records.jsonl'))
    if full:point['full_development_metrics']=metrics(records)
    write(out/'metrics.json',point)
    curvepath=Path('output/training/validation-curve.json');curve=read(curvepath).get('checkpoints',[]) if curvepath.exists() else []
    if any(r['iteration']==a.step for r in curve):raise RuntimeError('Duplicate validation point')
    curve.append(point);write(curvepath,dict(checkpoints=curve))
    write(out/'complete.json',dict(complete=True,step=a.step,questions=len(questions),responses=len(records),records_sha256=point['records_sha256']))


if __name__=='__main__':main()
