"""Pinned single-4090 SFT contract and assistant-only labels; no answer generation."""
import json
import math
import time
from pathlib import Path
from experiments.replay_scaled.core import sha, write

SCOPE='opencoder-sft-768-v1'
PREFIX='opencoder-sft-768-'
ENTRY='experiments/opencoder_sft/run.sh'
GPU='NVIDIA GeForce RTX 4090'
DATA=Path('data/opencoder-sft')

def read(path):return json.loads(Path(path).read_text())
def rows(path):return [json.loads(l) for l in Path(path).open()]

def allocation_spec(args):
    if (args.hardware_scope!=SCOPE or not args.name.startswith(PREFIX) or args.entry!=ENTRY
        or args.gpu!=GPU or args.gpu_count!=1 or args.provider!='vast' or not 1<(args.max_cost_usd or 0)<=15):
        raise ValueError('Unregistered OpenCoder allocation')
    return dict(scope=SCOPE,gpu=GPU,count=1,max_gpu_rate=.6)

def verify_worker(names,lease):
    if (names!=[GPU] or lease.get('gpu')!=GPU or lease.get('gpu_count')!=1
        or lease.get('hardware_scope')!=SCOPE or not lease.get('job_name','').startswith(PREFIX)):
        raise RuntimeError('OpenCoder requires exactly one RTX4090 under named lease')
    return names

def ticket(state,name,job):
    path=Path(state)/'authorizations'/(name+'.json')
    if not name.startswith(PREFIX) or not name.replace('-','').isalnum() or not path.is_file():return None
    v=read(path)
    required=dict(job_name=name,authorized=True,scope=SCOPE,hardware_scope=SCOPE,entry=ENTRY,
                  provider='vast',gpu=GPU,gpu_count=1,checkpoint_group=768,training_authorized=True,
                  coding_questions=2000,optimizer_steps=84,disk_gb=80)
    if any(v.get(k)!=x for k,x in required.items()):return None
    if any(v.get(k)!=job.get(k) for k in ('provider','hardware_scope','entry','gpu_count','offer_id','disk_gb')):return None
    if (job.get('gpu') or job.get('requested_gpu'))!=GPU or job.get('completion_budget_scope'):return None
    for k,ceiling in [('max_cost_usd',15),('max_runtime_hours',18),('max_gpu_rate',.6)]:
        x=v.get(k)
        if not isinstance(x,(float,int)) or not math.isfinite(x) or not 0<x<=ceiling:return None
    if not 1<(job.get('max_cost_usd') or 0)<=v['max_cost_usd']:return None
    if not 0<(job.get('max_runtime_hours') or 0)<=v['max_runtime_hours']:return None
    if type(v.get('offer_id')) is not int or v['offer_id']<=0 or not v.get('source_user_instruction'):return None
    jobs=read(Path(state)/'ledger.json')['jobs']
    if any(n!=name and n.startswith(PREFIX) and (j.get('pod_id') or j.get('allocation_attempts')) for n,j in jobs.items()):return None
    return v

def check_inputs():
    cfg=read(DATA/'plan.json')
    for name,digest in cfg['files_sha256'].items():
        if sha(DATA/name)!=digest:raise RuntimeError('Frozen input changed: '+name)
    if cfg['scope']!=SCOPE or cfg['optimizer_steps']!=84 or cfg['effective_batch_size']!=32:raise RuntimeError('Training contract differs')
    return cfg

def check_lease():
    lease=read('worker-lease.json')
    if time.time()>lease['hard_deadline']-300 or lease.get('stop_requested'):raise RuntimeError('Lease requests safe stop; checkpoint retained')
    return lease

def encode(tokenizer,row):
    messages=row['messages']
    if len(messages)<2 or messages[-1]['role']!='assistant':raise ValueError('Missing supplied assistant answer')
    text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=False)
    marker='<|im_start|>assistant\n'
    start=text.rfind(marker)
    if start<0:raise ValueError('Assistant boundary absent')
    prefix=text[:start+len(marker)]
    completion=text[len(prefix):]
    # Empty reasoning carries no teacher evidence; never reward an immediate think-close.
    empty='<think>\n\n</think>\n\n'
    if completion.startswith(empty):prefix+=empty
    ids=tokenizer.encode(text,add_special_tokens=False)
    prompt=tokenizer.encode(prefix,add_special_tokens=False)
    if ids[:len(prompt)]!=prompt or not 0<len(prompt)<len(ids)<=4096:raise ValueError('Invalid token boundary or length')
    labels=[-100]*len(prompt)+ids[len(prompt):]
    return dict(ids=ids,labels=labels,target_tokens=len(ids)-len(prompt),id=row['id'])
