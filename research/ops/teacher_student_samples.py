"""Registered additional student samples; retain every success and failure."""
import hashlib
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]


def normalize(text):
    text=text.strip()
    if text.startswith('<think>'):
        text=text[len('<think>'):].lstrip()
    return text


def main():
    from transformers import AutoTokenizer
    from vllm import LLM,SamplingParams
    from experiments.rl_training.train import verifier,reasoning_is_complete,REVISION,MODEL
    from experiments.teacher_generation.prompt_format import VARIANTS,FORMAT
    plan=json.loads((ROOT/'data/student-sampling-plan.json').read_text())
    rows=[json.loads(x) for x in (ROOT/'data/teacher-pilot-20261024/teacher-comparison-128.jsonl').read_text().splitlines()]
    tok=AutoTokenizer.from_pretrained(MODEL,revision=REVISION)
    grade=verifier();destination=ROOT/'output/student.jsonl'
    if destination.exists():raise ValueError('Never silently repeat a sampled batch')
    engine=LLM(model=MODEL,revision=REVISION,tokenizer_revision=REVISION,dtype='bfloat16',
               max_model_len=8192,gpu_memory_utilization=.85,seed=20261026,trust_remote_code=False)
    jobs=[(row,attempt) for row in rows for attempt in range(plan['attempt_start'],plan['attempt_end'])]
    start=time.time()
    with destination.open('w',buffering=1) as stream:
        for offset in range(0,len(jobs),16):
            batch=jobs[offset:offset+16]
            prefixes=[tok.apply_chat_template([{'role':'system','content':VARIANTS['explicit']+' '+FORMAT}]+r['prompt'],
                        tokenize=True,add_generation_prompt=True,enable_thinking=True) for r,a in batch]
            seeds=[int(hashlib.sha256(f"teacher-v2:{r['id']}:{a}".encode()).hexdigest()[:8],16) for r,a in batch]
            results=engine.generate([{'prompt_token_ids':ids} for ids in prefixes],
                [SamplingParams(n=1,temperature=.6,top_p=.95,top_k=20,max_tokens=2048,seed=s) for s in seeds],use_tqdm=False)
            for (row,attempt),prefix,seed,result in zip(batch,prefixes,seeds,results):
                answer=result.outputs[0];text=normalize(answer.text)
                accepted=bool(answer.finish_reason=='stop' and reasoning_is_complete(text)
                              and grade(text,row['answer'],row['task'])
                              and len(tok.encode(text,add_special_tokens=False))+1<=2048)
                evidence={'raw_text':answer.text,'generated_token_ids':list(answer.token_ids),
                          'generation_prompt_ids':prefix,'seed':seed,'attempt':attempt}
                sha=hashlib.sha256(json.dumps(evidence,sort_keys=True).encode()).hexdigest()
                stream.write(json.dumps({'prompt_id':row['id'],'source':row['source'],'completion':text,
                    'accepted_answer':accepted,'finish_reason':answer.finish_reason,'attempt':attempt,
                    'model':MODEL,'revision':REVISION,'response_sha256':sha,'request_key':'student-v2-'+sha,
                    'normalization':'Remove only redundant leading <think>; training prefix already opens it.',
                    'generation':{'temperature':.6,'top_p':.95,'top_k':20,'max_new_tokens':2048,'thinking':True},
                    'raw_evidence':evidence})+'\n')
            print('student additional',offset+len(batch),'/',len(jobs),flush=True)
    (ROOT/'output/role-result.json').write_text(json.dumps({'development_complete':True,'responses':len(jobs),
                  'plan':plan,'elapsed_seconds':time.time()-start},indent=2)+'\n')


if __name__=='__main__':main()
