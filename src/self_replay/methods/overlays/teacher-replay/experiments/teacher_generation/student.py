"""One frozen student sample per prompt for the preselected teacher style."""
import argparse
import json
from pathlib import Path
from .prompt_format import VARIANTS, FORMAT
from .common import REVISION, durable_json, digest


def main():
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from experiments.rl_training.train import verifier, reasoning_is_complete
    p=argparse.ArgumentParser();p.add_argument('--variant',required=True,choices=VARIANTS);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[2]
    rows=[json.loads(x) for x in (root/'data/teacher-pilot-20261024/teacher-comparison-128.jsonl').read_text().splitlines()]
    if a.out.exists():raise ValueError('Student output exists; never implicit regeneration')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    tok=AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B',revision=REVISION)
    grade=verifier()
    engine=LLM(model='Qwen/Qwen3-0.6B',revision=REVISION,tokenizer_revision=REVISION,dtype='bfloat16',
               max_model_len=8192,gpu_memory_utilization=.85,seed=20261025,trust_remote_code=False)
    with a.out.open('w',buffering=1) as f:
        for start in range(0,len(rows),16):
            batch=rows[start:start+16]
            prompts=[tok.apply_chat_template([{'role':'system','content':VARIANTS[a.variant]+' '+FORMAT}]+r['prompt'],tokenize=True,add_generation_prompt=True,enable_thinking=True) for r in batch]
            results=engine.generate([{'prompt_token_ids':ids} for ids in prompts],
                [SamplingParams(n=1,temperature=.6,top_p=.95,top_k=20,max_tokens=2048,seed=20261025) for _ in batch],use_tqdm=False)
            for row,prefix,result in zip(batch,prompts,results):
                answer=result.outputs[0];text=answer.text
                good=bool(answer.finish_reason=='stop' and reasoning_is_complete(text) and grade(text,row['answer'],row['task']))
                r={'prompt_id':row['id'],'source':row['source'],'completion':text,'generated_token_ids':list(answer.token_ids),
                   'generation_prompt_ids':prefix,'accepted_answer':good,'finish_reason':answer.finish_reason,
                   'model':'Qwen/Qwen3-0.6B','revision':REVISION,'variant':a.variant,'seed':20261025,
                   'generation':{'temperature':.6,'top_p':.95,'top_k':20,'max_new_tokens':2048},
                   'response_sha256':digest({'text':text,'ids':list(answer.token_ids),'prefix':prefix})}
                f.write(json.dumps(r)+'\n')
            print('student responses',start+len(batch),'/',len(rows),flush=True)
    durable_json(a.out.with_suffix('.summary.json'),{'responses':len(rows),'complete':True,'variant':a.variant,'revision':REVISION})


if __name__=='__main__':main()
