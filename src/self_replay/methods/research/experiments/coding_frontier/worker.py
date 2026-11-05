"""Frozen768, eight samples/task, ascending source difficulty, full native context.

Responses and reference-check outcomes are append-only; no optimizer or quality stop.
"""
import collections,gzip,hashlib,json,os,statistics,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from experiments.opencoder_sft.runtime import read,write,sha
from experiments.coding_frontier.grading import check,grade
from experiments.coding_frontier.budget import SCOPE,verify_worker
DATA=Path('data/coding-frontier');OUT=Path('output/coding-frontier')

def seed(identity,sample):return int(hashlib.sha256(f'frontier-v1:{identity}:{sample}:17001'.encode()).hexdigest()[:8],16)

def reference(row):
    started=time.time();outcomes=[]
    for i,code in enumerate(row['references']):
        if time.time()-started>30:break
        result=check(code,row['spec']);outcomes.append(result)
        if result['status']=='passed':return dict(id=row['id'],valid=True,reference_index=i,result=result,seconds=time.time()-started)
    return dict(id=row['id'],valid=False,reason='reference_not_verified',outcomes=outcomes,seconds=time.time()-started)

def summary(records,questions):
    levels={}
    for level in sorted({r['difficulty'] for r in records},key=lambda d:min(r['difficulty_rank'] for r in records if r['difficulty']==d)):
        rs=[r for r in records if r['difficulty']==level];qs=[q for q in questions if q['difficulty']==level and q.get('samples')==8]
        tokens=sorted(r['total_tokens'] for r in rs);correct=[r['total_tokens'] for r in rs if r['correct']]
        levels[level]=dict(answers=len(rs),questions=len(qs),accuracy=sum(r['correct'] for r in rs)/len(rs),
            solved_questions=sum(q['correct_samples']>0 for q in qs),correct_of_eight=dict(collections.Counter(q['correct_samples'] for q in qs)),
            mean_tokens=statistics.mean(tokens),median_tokens=statistics.median(tokens),p90_tokens=tokens[min(len(tokens)-1,int(.9*len(tokens)))],
            mean_correct_tokens=statistics.mean(correct) if correct else None,failures=dict(collections.Counter(r['status'] for r in rs if not r['correct'])),
            completed_correct_within_output_cap={str(cap):sum(r['correct'] and r['total_tokens']<=cap for r in rs) for cap in (2048,4096,8192,16384,32768)},
            known_parent_exposure_answers=sum(r['parent768_seen_exact'] for r in rs))
    return levels

def append(path,values):
    with path.open('a') as f:
        for v in values:f.write(json.dumps(v)+'\n')
        f.flush();os.fsync(f.fileno())

def main():
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM,SamplingParams
    from vllm.lora.request import LoRARequest
    from experiments.replay_scaled.core import MODEL,REVISION
    cfg=read(DATA/'plan.json');OUT.mkdir(parents=True,exist_ok=True)
    verify_worker([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],read('worker-lease.json'))
    if cfg['scope']!=SCOPE or cfg['training_authorized'] or cfg['samples']!=8 or cfg['context_tokens']!=40960:raise RuntimeError('Frozen survey contract changed')
    for f,h in cfg['files_sha256'].items():
        if sha(DATA/f)!=h:raise RuntimeError('Survey input hash mismatch: '+f)
    if cfg.get('resume'):
        from experiments.coding_frontier.resume import restore
        restore(DATA/'resume',OUT)
    for spec,good,bad in [(dict(kind='io',inputs=[''],outputs=['2']),'print(2)','print(3)'),
            (dict(kind='python_tests',test='from solution import f\ndef test_a(): assert f(1)==2'),'def f(x): return x+1','def f(x): return x')]:
        if check(good,spec)['status']!='passed' or check(bad,spec)['status']=='passed':raise RuntimeError('Sandbox preflight failed')
    # Immutable response chunks are backed up independently of progress/status.
    chunks=OUT/'responses';chunks.mkdir(exist_ok=True)
    records=[];questions=[];done=set()
    if (OUT/'questions.jsonl').exists():
        questions=[json.loads(l) for l in (OUT/'questions.jsonl').open()];done={q['id'] for q in questions}
        for p in sorted(chunks.glob('*.jsonl.gz')):
            with gzip.open(p,'rt') as f:
                for l in f:
                    r=json.loads(l)
                    if r['id'] in done:records.append({k:v for k,v in r.items() if k not in ['text','generated_token_ids']})
        if any(sum(r['id']==q['id'] for r in records)!=q.get('samples',0) for q in questions):raise RuntimeError('Resume response manifest incomplete')
    tokenizer=AutoTokenizer.from_pretrained(DATA/'adapter',local_files_only=True)
    engine=LLM(model=MODEL,revision=REVISION,tokenizer_revision=REVISION,dtype='bfloat16',max_model_len=40960,
        max_num_seqs=16,max_num_batched_tokens=8192,gpu_memory_utilization=.85,enable_lora=True,max_lora_rank=16,max_loras=1,
        enable_prefix_caching=False,enforce_eager=True,generation_config='vllm',seed=17001)
    route=LoRARequest('parent768',1,str((DATA/'adapter').resolve()));engine.llm_engine.add_lora(route)
    prior={}
    if questions:
        from experiments.coding_frontier.resume import validate
        prior=validate(OUT)
    started=time.time()-cfg.get('resume',{}).get('prior_elapsed',0)
    generation_seconds=prior.get('generation_seconds',0);all_tokens=prior.get('generated_tokens',0);survey_complete=True;phase='preflight'
    write(OUT/'preflight.json',dict(passed=True,gpu=torch.cuda.get_device_name(),context_tokens=40960,parent_group=768,
        parent_adapter_sha256=sha(DATA/'adapter/adapter_model.safetensors'),model_revision=REVISION,inference_only=True,training_updates=0))
    def progress(phase,current_id=None):
        value=dict(phase=phase,dataset=cfg['dataset'],visited=len(done),total_questions=cfg['questions'],answers=len(records),
            expected_answers=cfg['questions']*8,graded_questions=sum(q.get('samples')==8 for q in questions),
            skipped_questions=sum(q.get('samples',0)==0 for q in questions),current_id=current_id,updated_at=time.time(),
            elapsed=time.time()-started,generation_seconds=generation_seconds,generated_tokens=all_tokens,
            levels=summary(records,questions),current_difficulty=questions[-1]['difficulty'] if questions else None)
        write(OUT/'progress.json',value);return value
    try:
        with gzip.open(DATA/'questions.jsonl.gz','rt') as stream,ThreadPoolExecutor(max_workers=4) as pool:
            batch=[]
            def process(batch):
                nonlocal generation_seconds,all_tokens
                lease=read('worker-lease.json')
                if lease.get('stop_requested') or time.time()>lease['hard_deadline']-1200 or Path('output/stop-request.json').exists():return False
                progress('reference_check',batch[0]['id'])
                refs=list(pool.map(reference,batch));append(OUT/'reference-checks.jsonl',refs)
                valid=[]
                for q,ref in zip(batch,refs):
                    ids=tokenizer.apply_chat_template(q['prompt'],tokenize=True,add_generation_prompt=True,enable_thinking=True)
                    if not ref['valid'] or len(ids)>=40960:
                        rec=dict(id=q['id'],difficulty=q['difficulty'],difficulty_rank=q['difficulty_rank'],samples=0,
                            reason='prompt_exceeds_context' if len(ids)>=40960 else 'reference_not_verified',parent768_seen_exact=q['parent768_seen_exact'])
                        append(OUT/'questions.jsonl',[rec]);questions.append(rec);done.add(q['id'])
                    else:valid.append((q,ids))
                if not valid:progress('survey');return True
                requests=[(q,s,ids) for q,ids in valid for s in range(8)]
                # Each batch spans up to two questions; vLLM keeps their16samples active.
                params=[SamplingParams(n=1,max_tokens=40960-len(ids),temperature=.6,top_p=.95,top_k=20,seed=seed(q['id'],s)) for q,s,ids in requests]
                progress('generation',valid[0][0]['id']);begin=time.time()
                outputs=engine.generate([{'prompt_token_ids':ids} for q,s,ids in requests],params,lora_request=route,use_tqdm=False)
                elapsed=time.time()-begin;generation_seconds+=elapsed
                def finish(item):
                    (q,s,ids),output=item
                    if output.lora_request is None or output.lora_request.lora_int_id!=1:raise RuntimeError('Wrong adapter route')
                    o=output.outputs[0];result=grade(o.text,o.finish_reason,q['spec']);final=o.text.rsplit('</think>',1)[-1] if '</think>' in o.text else ''
                    final_tokens=len(tokenizer.encode(final,add_special_tokens=False)) if final else 0
                    return dict(result,id=q['id'],dataset=cfg['dataset'],sample=s,seed=seed(q['id'],s),difficulty=q['difficulty'],difficulty_rank=q['difficulty_rank'],
                        text=o.text,generated_token_ids=list(o.token_ids),total_tokens=len(o.token_ids),prompt_tokens=len(ids),
                        final_tokens_retokenized=final_tokens,reasoning_tokens_approx=len(o.token_ids)-final_tokens,
                        finish_reason=o.finish_reason,stop_reason=o.stop_reason,context_tokens=40960,output_allowance=40960-len(ids),
                        batch_generation_seconds=elapsed,parent768_seen_exact=q['parent768_seen_exact'],cross_dataset_duplicate=q['cross_dataset_duplicate'])
                progress('grading',valid[0][0]['id']);new=list(pool.map(finish,zip(requests,outputs)))
                key=hashlib.sha256('|'.join(q['id'] for q,ids in valid).encode()).hexdigest()[:20];dest=chunks/(key+'.jsonl.gz')
                if dest.exists():raise RuntimeError('Existing uncommitted response chunk requires explicit reconciliation')
                tmp=dest.with_suffix('.tmp')
                with gzip.open(tmp,'wt') as f:
                    for r in new:f.write(json.dumps(r)+'\n')
                os.replace(tmp,dest)
                newq=[]
                for q,ids in valid:
                    rs=[r for r in new if r['id']==q['id']];correct=[r['total_tokens'] for r in rs if r['correct']]
                    newq.append(dict(id=q['id'],difficulty=q['difficulty'],difficulty_rank=q['difficulty_rank'],samples=8,
                        correct_samples=sum(r['correct'] for r in rs),truncated_samples=sum(r['finish_reason']!='stop' for r in rs),
                        shortest_correct_tokens=min(correct) if correct else None,mean_tokens=statistics.mean(r['total_tokens'] for r in rs),
                        parent768_seen_exact=q['parent768_seen_exact'],chunk=dest.name,chunk_sha256=sha(dest)))
                append(OUT/'questions.jsonl',newq);questions.extend(newq);done.update(q['id'] for q,ids in valid)
                all_tokens+=sum(r['total_tokens'] for r in new);records.extend({k:v for k,v in r.items() if k not in ['text','generated_token_ids']} for r in new)
                progress('survey');print(f"{cfg['dataset']}: {len(done)}/{cfg['questions']} questions, {len(records)} answers",flush=True);return True
            for line in stream:
                q=json.loads(line)
                if q['id'] in done:continue
                batch.append(q)
                if len(batch)==2:
                    if not process(batch):survey_complete=False;break
                    batch=[]
            else:
                if batch and not process(batch):survey_complete=False
        result=progress('complete' if survey_complete else 'budget_checkpoint')
        write(OUT/'results.json',dict(result,complete=survey_complete,official_evaluation=False,training_updates=0,
            limitations=['Observed sampling success, not proof of an absolute model limit.','Difficulty labels differ across sources.',
                'Reference/test/harness failures excluded from model accuracy and reported separately.','Cross-dataset duplicates and prior exact training exposure are flagged.']))
        write(OUT/'completion.json',dict(complete=survey_complete,partial_preserved=True,results_sha256=sha(OUT/'results.json'),
            reason='finished' if survey_complete else 'funded_stage_deadline',training_updates=0))
    except Exception as e:
        write(OUT/'failure.json',dict(error=str(e),detected_at=time.time()));raise

if __name__=='__main__':main()
