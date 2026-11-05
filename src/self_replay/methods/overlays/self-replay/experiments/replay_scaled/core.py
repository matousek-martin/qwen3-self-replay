import hashlib
import json
import random
from pathlib import Path
from experiments.rl_training.objectives import rewards, advantages

DOMAINS = ('math', 'coding', 'tools', 'instructions')
SCOPE = 'replay-scaled-a100-v1'
ENTRY = 'experiments/replay_scaled/run.sh'
GPU = 'NVIDIA A100-SXM4-80GB'
REVISION = 'c1899de289a04d12100db370d81485cdf75e47ca'
MODEL = 'Qwen/Qwen3-0.6B'

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n'); tmp.replace(path)

def balanced(rows, per_domain, seed):
    rng = random.Random(seed); pools = {}
    for domain in DOMAINS:
        pool = sorted((r for r in rows if r['task']==domain), key=lambda r:r['id'])
        if len(pool)<per_domain: raise ValueError('Insufficient unique '+domain+' questions')
        rng.shuffle(pool); pools[domain] = pool[:per_domain]
    out = []
    for i in range(per_domain):
        chunk = [pools[d][i] for d in DOMAINS]; rng.shuffle(chunk); out.extend(chunk)
    if len({r['id'] for r in out}) != len(out): raise ValueError('Duplicate questions')
    return out

def policy_rewards(correct, lengths, reference):
    # Exact historical fixed correctness-gated reward and unscaled Dr.GRPO advantage.
    result = rewards(correct, lengths, 'replay', .1, reference, replay_reward='fixed')
    return result, advantages(result, scale=False)

def checkpoint_boundaries(done,total,config,*,pilot=False,stop_after=None):
    """Cheap complete-state saves are independent of expensive validation."""
    terminal=done==total or done==stop_after or (pilot and done==2)
    validate=terminal or done%config['validate_every']==0 or done in config['milestones']
    save=validate or done%config['save_every']==0
    return save,validate

def admission(buffer, prompt_id, prompt_ids, samples, capacity=128):
    if all(s['correct'] for s in samples):
        best = min(samples, key=lambda s:len(s['ids']))
        if prompt_id not in buffer or len(best['ids'])<len(buffer[prompt_id][1]):
            buffer[prompt_id] = (prompt_ids, best['ids'])
        if len(buffer)>capacity: del buffer[next(iter(buffer))]

def allocation_spec(args):
    if (args.hardware_scope != SCOPE or not args.name.startswith('replay-scaled-')
        or args.entry != ENTRY or args.gpu != GPU or type(args.gpu_count) is not int
        or args.gpu_count not in (2,4) or not 1 < (args.max_cost_usd or 0) <= 150):
        raise ValueError('Invalid scoped replay hardware request')
    return dict(scope=SCOPE,gpu=GPU,count=args.gpu_count,max_gpu_rate=2.20)

def verify_worker(names, lease):
    if (lease.get('hardware_scope')!=SCOPE or not lease.get('job_name','').startswith('replay-scaled-')
        or lease.get('gpu')!=GPU or lease.get('gpu_count') not in (2,4)
        or len(names)!=lease['gpu_count'] or any(n!=GPU for n in names)):
        raise RuntimeError('Replay worker differs from named 2/4 A100 reservation')
    return names

def ticket(state, name, job):
    if not name.startswith('replay-scaled-') or not name.replace('-','').isalnum(): return None
    path = Path(state)/'authorizations'/(name+'.json')
    if not path.is_file(): return None
    v=json.loads(path.read_text())
    fields={'job_name':name,'scope':'replay_vast_bounded_v1','authorized':True,'provider':'vast',
            'hardware_scope':SCOPE,'entry':ENTRY,'gpu':GPU}
    if any(v.get(k)!=value for k,value in fields.items()): return None
    for k in ('provider','hardware_scope','entry','gpu','gpu_count','offer_id','disk_gb'):
        if v.get(k)!=job.get(k): return None
    if (v['gpu_count'] not in (2,4) or type(v.get('offer_id')) is not int or v['offer_id']<=0
        or not 50<=v.get('disk_gb',0)<=150 or not 1<(job.get('max_cost_usd') or 0)<=v.get('max_cost_usd',0)<=150
        or not 0<(job.get('max_runtime_hours') or 0)<=v.get('max_runtime_hours',0)<=72
        or not 0<v.get('max_gpu_rate',0)<=2.2 or not v.get('source_user_instruction')
        or job.get('completion_budget_scope')): return None
    if v.get('phase')=='pilot' and (v['max_cost_usd']>10 or v['max_runtime_hours']>2): return None
    if v.get('phase') not in ('pilot','training'): return None
    return v
