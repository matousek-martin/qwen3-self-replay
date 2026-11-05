"""Frozen768 training-only screening; verified own answers seed the new branch."""
import collections
import json
import os
from pathlib import Path
import time
from .core import MODEL, REVISION, sha, write
from .coding_budget import verify_worker, select
from .generation import generate_batch

DATA=Path('data/coding-replay')
OUT=Path('output/coding-screen')


def main():
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from peft import PeftModel
    from experiments.rl_training.objectives import reasoning_is_complete
    from experiments.shorterbetter.multidomain.rewards import verify,code_check
    cfg=json.loads((DATA/'plan.json').read_text());lease=json.loads(Path('worker-lease.json').read_text())
    verify_worker([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],lease)
    for name,digest in cfg['files_sha256'].items():
        if sha(DATA/name)!=digest:raise RuntimeError('Changed coding campaign input: '+name)
    rows=[json.loads(l) for l in (DATA/'screening-candidates.jsonl').open()]
    assert len(rows)==128 and len({r['id'] for r in rows})==128
    assert collections.Counter(r['difficulty'] for r in rows)=={'introductory':64,'interview':64}
    assert all(r['task']=='coding' and r['split']=='train' for r in rows)
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'records.jsonl').exists():raise RuntimeError('Explicit recovery required; do not overwrite screening')
    torch.manual_seed(17);torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True)
    path=Path(os.environ['HF_HOME'])/'hub/models--Qwen--Qwen3-0.6B/snapshots'/REVISION
    tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
    base=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda')
    model=PeftModel.from_pretrained(base,DATA/'adapter',is_trainable=False).eval();model.requires_grad_(False)
    assert not any(p.requires_grad for p in model.parameters())
    if not code_check('print(2)',{'inputs':[''],'outputs':['2']}) or code_check('print(3)',{'inputs':[''],'outputs':['2']}):
        raise RuntimeError('Sandbox preflight failed')
    eos=model.generation_config.eos_token_id;eos={eos} if isinstance(eos,int) else set(eos)
    pad=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    prompts={r['id']:tokenizer.apply_chat_template(r['prompt'],tokenize=True,add_generation_prompt=True,enable_thinking=True) for r in rows}
    if any(not p or len(p)>2048 for p in prompts.values()):raise RuntimeError('Invalid prompt cap')
    write(OUT/'preflight.json',dict(passed=True,gpu=torch.cuda.get_device_name(),trainable_parameters=0,checkpoint=768,sandbox=True))
    requests=[(r,s,17001+i*4+s) for i,r in enumerate(rows) for s in range(4)]
    records=[];started=time.time();deadline=float(os.environ['CODING_SCREENING_DEADLINE'])
    for offset in range(0,512,16):
        if time.time()>deadline-300:raise RuntimeError('Screening stage budget reached')
        write(OUT/'progress.json',dict(phase='screening',current=offset,total=512,updated_at=time.time(),elapsed=time.time()-started))
        batch=requests[offset:offset+16];begin=time.time()
        seqs=generate_batch(model,[prompts[r['id']] for r,_,_ in batch],[seed for _,_,seed in batch],
            pad_token_id=pad,eos_token_ids=eos,max_new_tokens=2048,temperature=.6,top_k=20,top_p=.95)
        with (OUT/'records.jsonl').open('a') as f:
            for (r,s,seed),seq in zip(batch,seqs):
                text=tokenizer.decode(seq,skip_special_tokens=True);truncated=not seq or seq[-1] not in eos
                correct=bool(not truncated and reasoning_is_complete(text) and verify(text,r['answer']))
                record=dict(id=r['id'],sample=s,seed=seed,task='coding',difficulty=r['difficulty'],correct=correct,
                    truncated=truncated,ids=seq,prompt_ids=prompts[r['id']],text=text,total_tokens=len(seq))
                records.append(record);f.write(json.dumps(record)+'\n')
        write(OUT/'progress.json',dict(phase='screening',current=len(records),total=512,updated_at=time.time(),elapsed=time.time()-started,
            correct=sum(r['correct'] for r in records),last_batch_seconds=time.time()-begin))
        print(f'Screened {len(records)}/512 own-model answers',flush=True)
    retention=[json.loads(l) for l in (DATA/'retention.jsonl').open()]
    schedule=select(rows,records,retention)
    counts=collections.Counter(sum(r['correct'] for r in records if r['id']==q['id']) for q in rows)
    write(OUT/'results.json',dict(correct_of_four=dict(counts),eligible_questions=128-counts[0],answers=512,correct=sum(r['correct'] for r in records)))
    if schedule is None:
        write(OUT/'completion.json',dict(complete=True,eligible=False,reason='Fewer than32 eligible coding questions; no training',elapsed=time.time()-started))
        return
    selected={r['id'] for r in schedule if r['task']=='coding'}
    seeds=[]
    by_id={r['id']:r for r in rows}
    for identity in sorted(selected):
        best=min((r for r in records if r['id']==identity and r['correct']),key=lambda r:r['total_tokens'])
        if not verify(best['text'],by_id[identity]['answer']):raise RuntimeError('Replay seed failed regrading')
        seeds.append({k:best[k] for k in ('id','ids','prompt_ids','correct','truncated')})
    target=Path('output/coding-data');target.mkdir()
    (target/'train.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in schedule))
    for name in ('development.jsonl','references.json'):(target/name).write_bytes((DATA/name).read_bytes())
    write(target/'seed-replay.json',seeds)
    plan=cfg['training_plan']
    plan['files_sha256']={p.name:sha(p) for p in target.iterdir() if p.is_file()}
    write(target/'plan.json',plan)
    write(OUT/'completion.json',dict(complete=True,eligible=True,selected_coding_questions=len(selected),training_slots=128,
        schedule_sha256=sha(target/'train.jsonl'),plan_sha256=sha(target/'plan.json'),elapsed=time.time()-started))


if __name__=='__main__':main()
