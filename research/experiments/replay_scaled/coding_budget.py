"""One user-approved coding replay campaign, eighteen dollars, one RTX4090."""
import json
import math
from pathlib import Path

SCOPE = 'coding-replay-768-v1'
PREFIX = 'coding-replay-768-'
ENTRY = 'experiments/replay_scaled/coding_campaign.sh'
GPU = 'NVIDIA GeForce RTX 4090'


def allocation_spec(args):
    if (args.hardware_scope != SCOPE or not args.name.startswith(PREFIX)
            or args.entry != ENTRY or args.gpu != GPU or args.gpu_count != 1
            or args.provider != 'vast' or not 1 < (args.max_cost_usd or 0) <= 18):
        raise ValueError('Unregistered coding replay allocation')
    return dict(scope=SCOPE, gpu=GPU, count=1, max_gpu_rate=.5)


def verify_worker(names, lease):
    if (lease.get('hardware_scope') != SCOPE or not lease.get('job_name','').startswith(PREFIX)
            or lease.get('gpu') != GPU or lease.get('gpu_count') != 1 or names != [GPU]):
        raise RuntimeError('Coding replay requires exactly one RTX4090')
    return names


def ticket(state, name, job):
    p = Path(state)/'authorizations'/(name+'.json')
    if not name.startswith(PREFIX) or not name.replace('-','').isalnum() or not p.is_file():
        return None
    v = json.loads(p.read_text())
    required = dict(job_name=name, authorized=True, scope=SCOPE, provider='vast',
        hardware_scope=SCOPE, entry=ENTRY, gpu=GPU, gpu_count=1, checkpoint_group=768,
        training_authorized=True, screening_questions=128, training_groups=128,
        stage_caps_usd={'screening':4,'training':10,'evaluation':4}, disk_gb=80)
    if any(v.get(k)!=x for k,x in required.items()):return None
    if any(v.get(k)!=job.get(k) for k in ('provider','hardware_scope','entry','gpu_count','offer_id','disk_gb')):return None
    if (job.get('gpu') or job.get('requested_gpu'))!=GPU or job.get('completion_budget_scope'):return None
    for k,ceiling in [('max_cost_usd',18),('max_runtime_hours',24),('max_gpu_rate',.5)]:
        x=v.get(k)
        if not isinstance(x,(int,float)) or not math.isfinite(x) or not 0<x<=ceiling:return None
    if not 1 < (job.get('max_cost_usd') or 0) <= v['max_cost_usd']:return None
    if not 0 < (job.get('max_runtime_hours') or 0) <= v['max_runtime_hours']:return None
    if type(v.get('offer_id')) is not int or v['offer_id']<=0 or not v.get('source_user_instruction'):return None
    # A single registration; a replacement cannot reuse the entire envelope.
    jobs=json.loads((Path(state)/'ledger.json').read_text())['jobs']
    if any(n!=name and n.startswith(PREFIX) and (j.get('pod_id') or j.get('allocation_attempts')) for n,j in jobs.items()):return None
    return v


def admit(buffer, prompt_id, prompt_ids, samples, capacity=128):
    passing=[r for r in samples if r['correct'] and not r['truncated'] and r['ids']]
    if not passing:return
    best=min(passing,key=lambda r:len(r['ids']))
    if prompt_id not in buffer or len(best['ids']) < len(buffer[prompt_id][1]):
        buffer[prompt_id]=(prompt_ids,best['ids'])
    if len(buffer)>capacity:del buffer[next(iter(buffer))]


def select(rows, records, retention):
    """Freeze a bounded 96coding/32retention schedule from the screening only."""
    import collections, random
    groups=collections.defaultdict(list)
    for r in records:groups[r['id']].append(r)
    if len(rows)!=128 or set(groups)!={r['id'] for r in rows}:raise ValueError('Incomplete screening')
    if any(len(g)!=4 or sorted(r['sample'] for r in g)!=[0,1,2,3] for g in groups.values()):raise ValueError('Incomplete samples')
    eligible=[r for r in rows if any(x['correct'] for x in groups[r['id']])]
    eligible.sort(key=lambda r:(sum(x['correct'] for x in groups[r['id']])==4,r['id']))
    if len(eligible)<32:return None
    chosen=eligible[:96]
    coding=[chosen[i%len(chosen)] for i in range(96)]
    assert max(collections.Counter(r['id'] for r in coding).values())<=3
    if len(retention)!=32 or any(r['task']=='coding' or r['split']!='train' for r in retention):raise ValueError('Invalid retention set')
    random.Random(17).shuffle(coding)
    schedule=[]
    for i,r in enumerate(retention):schedule+=coding[i*3:i*3+3]+[r]
    return schedule
