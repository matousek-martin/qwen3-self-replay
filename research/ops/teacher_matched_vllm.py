"""Frozen 800-prompt matched greedy validation; fresh model process for each arm."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def truncate_final(ids,close_id,cap=1024):
    if close_id in ids:
        boundary=ids.index(close_id)+1+cap
        return ids[:boundary],len(ids)>boundary
    return ids,False


def main():
    from transformers import AutoTokenizer
    from vllm import LLM,SamplingParams
    from vllm.lora.request import LoRARequest
    from experiments.rl_training.train import MODEL,REVISION,verifier
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--adapter',type=Path);a=p.parse_args()
    rows=[json.loads(x) for x in a.data.read_text().splitlines()]
    data_sha=hashlib.sha256(a.data.read_bytes()).hexdigest()
    if len(rows)!=800 or data_sha!='dc42294cc3ce31436278e72673e285f9aadf0833928ac1e7b23590c23448b43a':
        raise ValueError('Validation data changed')
    if a.out.exists():raise ValueError('Existing evaluation preserved; no implicit overwrite')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=REVISION);grade=verifier()
    close=tok.convert_tokens_to_ids('</think>');eos=tok.eos_token_id
    model=LLM(model=MODEL,revision=REVISION,tokenizer_revision=REVISION,dtype='bfloat16',
              max_model_len=8192,gpu_memory_utilization=.85,enable_lora=bool(a.adapter),max_lora_rank=64,
              generation_config='vllm',seed=17,trust_remote_code=False,enable_prefix_caching=False)
    route=LoRARequest('candidate',1,str(a.adapter.resolve())) if a.adapter else None
    if route:
        model.llm_engine.add_lora(route)
        if 1 not in model.llm_engine.list_loras():raise RuntimeError('Adapter not registered')
    start=time.time();count=0
    with a.out.open('w',buffering=1) as stream:
        for offset in range(0,len(rows),16):
            batch=rows[offset:offset+16]
            inputs=[{'prompt_token_ids':tok.apply_chat_template(r['prompt'],tokenize=True,add_generation_prompt=True,enable_thinking=True)} for r in batch]
            outputs=model.generate(inputs,SamplingParams(temperature=0,top_p=1,top_k=-1,max_tokens=4096,
                         seed=17,stop_token_ids=[eos],include_stop_str_in_output=True),lora_request=route,use_tqdm=False)
            for row,output in zip(batch,outputs):
                actual=output.lora_request
                if (actual is None)!=(route is None) or (route and actual.lora_int_id!=1):
                    raise RuntimeError('Incorrect evaluation LoRA routing')
                sample=output.outputs[0];raw=list(sample.token_ids)
                ids,final_limited=truncate_final(raw,close)
                text=tok.decode(ids,skip_special_tokens=True)
                r={'prompt_id':row['id'],'source':row['source'],'correct':bool(grade(text,row['answer'],row['task'])),
                   'completion':text,'generated_token_ids':ids,'total_tokens':len(ids),
                   'thinking_tokens':ids.index(close) if close in ids else len(ids),
                   'truncated':final_limited or sample.finish_reason!='stop','seed':17,'greedy':True,
                   'raw_generated_tokens':len(raw),'final_cap_applied':final_limited}
                stream.write(json.dumps(r)+'\n');count+=1
            print('matched validation',count,'/800',flush=True)
    a.out.with_suffix('.summary.json').write_text(json.dumps({'complete':True,'responses':count,'model':MODEL,
         'revision':REVISION,'data_sha256':data_sha,'backend':'vllm','greedy':True,'seed':17,
         'max_new_tokens':4096,'final_answer_cap':1024,'elapsed_seconds':time.time()-start,
         'adapter':str(a.adapter) if a.adapter else None},indent=2)+'\n')


if __name__=='__main__':main()
