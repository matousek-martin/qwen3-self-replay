"""Frozen, verified teacher trajectories with original token IDs; inference only."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from .integration import TEACHER, TEACHER_REVISION, STUDENT_REVISION, hashed


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from experiments.rl_training.train import load_rows, verifier, reasoning_is_complete
    from .loss import assert_tokenizer_alignment, freeze_teacher
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); config=json.loads(a.config.read_text());root=Path(__file__).resolve().parents[2]
    data=root/'data'/config['training']['data'];rows=load_rows(data)
    if a.out.exists():
        raise ValueError('Preparation output already exists; no implicit re-generation')
    a.out.mkdir(parents=True)
    start=time.monotonic(); torch.manual_seed(41017)
    tok=AutoTokenizer.from_pretrained(TEACHER,revision=TEACHER_REVISION);tok.pad_token=tok.eos_token
    student_tok=AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B',revision=STUDENT_REVISION);student_tok.pad_token=student_tok.eos_token
    teacher=freeze_teacher(AutoModelForCausalLM.from_pretrained(TEACHER,revision=TEACHER_REVISION,
                           torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda'))
    assert_tokenizer_alignment(student_tok,tok,151936,teacher.config.vocab_size)
    eos=teacher.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos]);eos.add(tok.eos_token_id)
    grade=verifier();entries={};generated_tokens=0
    with (a.out/'teacher-generation.jsonl').open('w',buffering=1) as log:
        for row in rows:
            prompt=tok.apply_chat_template(row['prompt'],tokenize=True,add_generation_prompt=True,enable_thinking=True)
            ids=torch.tensor([prompt],device='cuda')
            entry={'prompt_ids':prompt,'completion_ids':[],'accepted':False}
            for attempt in range(4):
                with torch.inference_mode():
                    tokens=teacher.generate(ids,attention_mask=torch.ones_like(ids),max_new_tokens=2048,
                        do_sample=True,temperature=.6,top_p=.95,top_k=20,pad_token_id=tok.pad_token_id,
                        use_cache=True)[0,len(prompt):].tolist()
                generated_tokens+=len(tokens);text=tok.decode(tokens,skip_special_tokens=True)
                ok=bool(tokens and tokens[-1] in eos and reasoning_is_complete(text) and grade(text,row['answer'],row['task']))
                log.write(json.dumps({'prompt_id':row['id'],'attempt':attempt,'prompt_ids':prompt,'completion_ids':tokens,
                     'completion':text,'accepted':ok,'seed':41017,'teacher_revision':TEACHER_REVISION})+'\n')
                if ok:
                    entry.update(completion_ids=tokens,accepted=True,attempt=attempt);break
            entries[row['id']]=entry
            print('teacher cache',len(entries),'/',len(rows),'accepted',sum(e['accepted'] for e in entries.values()),flush=True)
    cache={'entries':entries,'teacher_revision':TEACHER_REVISION,'student_revision':STUDENT_REVISION,
           'tokenizer_sha256':hashed(tok.backend_tokenizer.to_str()),'data_sha256':hashlib.sha256(data.read_bytes()).hexdigest(),
           'generated_tokens':generated_tokens,'elapsed_seconds':time.monotonic()-start,
           'gpu':torch.cuda.get_device_name(),'peak_gpu_bytes':torch.cuda.max_memory_allocated()}
    (a.out/'teacher-cache.json').write_text(json.dumps(cache,indent=2)+'\n')
    if not any(e['accepted'] for e in entries.values()):
        raise RuntimeError('Zero teacher coverage; no student training')


if __name__=='__main__':main()
