"""Bounded inference-only coding length probe, separate from training."""
import json
import math
from pathlib import Path

SCOPE = 'coding-length-probe-v1'
PREFIX = 'coding-length-probe-'
ENTRY = 'experiments/replay_scaled/length_probe.sh'
GPU = 'NVIDIA GeForce RTX 4090'


def remaining_budget(previous_jobs):
    spent=0.
    for job in previous_jobs:
        empty=(job.get('state')=='allocation_unavailable' and job.get('safely_no_pod') and not job.get('pod_id') and not job.get('allocation_attempts'))
        startup=(job.get('state')=='failed' and job.get('startup_failed_before_launch') is True and not job.get('worker_launch_attempted_at'))
        charge=job.get('charged_estimate_usd')
        if (not job.get('released_at') or not (empty or startup) or not isinstance(charge,(int,float)) or not math.isfinite(charge) or charge<0):
            raise ValueError('Prior probe is not a reconciled pre-inference failure')
        spent+=charge
    remaining=round(4-spent,4)
    if remaining<=1:raise ValueError('Combined probe budget exhausted')
    return remaining


def allocation_spec(args):
    if (args.hardware_scope != SCOPE or not args.name.startswith(PREFIX)
            or args.entry != ENTRY or args.gpu != GPU or args.gpu_count != 1
            or not 1 < (args.max_cost_usd or 0) <= 4):
        raise ValueError('Invalid inference-only coding length reservation')
    return dict(scope=SCOPE, gpu=GPU, count=1, max_gpu_rate=.5)


def verify_worker(names, lease):
    if (lease.get('hardware_scope') != SCOPE or not lease.get('job_name', '').startswith(PREFIX)
            or lease.get('gpu') != GPU or lease.get('gpu_count') != 1 or names != [GPU]):
        raise RuntimeError('Coding probe hardware differs from lease')
    return names


def ticket(state, name, job):
    path = Path(state)/'authorizations'/(name+'.json')
    if not name.startswith(PREFIX) or not name.replace('-', '').isalnum() or not path.is_file(): return None
    v = json.loads(path.read_text())
    required = dict(job_name=name, authorized=True, scope=SCOPE, provider='vast',
        hardware_scope=SCOPE, entry=ENTRY, gpu=GPU, gpu_count=1, phase='evaluation',
        evaluation_only=True, training_authorized=False, checkpoint_group=850,
        caps=[2048,4096,8192], samples=4, questions=47, disk_gb=60)
    if any(v.get(k) != x for k,x in required.items()): return None
    if any(v.get(k) != job.get(k) for k in ['provider','hardware_scope','entry','gpu_count','offer_id','disk_gb']): return None
    if (job.get('gpu') or job.get('requested_gpu')) != GPU: return None
    for key, maximum in [('max_cost_usd',4),('max_runtime_hours',6),('max_gpu_rate',.5)]:
        value=v.get(key)
        if not isinstance(value,(int,float)) or not math.isfinite(value) or not 0 < value <= maximum: return None
    if not 1 < (job.get('max_cost_usd') or 0) <= v['max_cost_usd']: return None
    if not 0 < (job.get('max_runtime_hours') or 0) <= v['max_runtime_hours']: return None
    if type(v.get('offer_id')) is not int or v['offer_id'] <= 0 or not v.get('source_user_instruction'): return None
    if job.get('completion_budget_scope'): return None
    previous=v.get('previous_attempts',[])
    if len(previous)!=len(set(previous)) or name in previous or any(not n.startswith(PREFIX) for n in previous):return None
    if previous:
        try:
            jobs=json.loads((Path(state)/'ledger.json').read_text())['jobs']
            remaining=remaining_budget([jobs[n] for n in previous])
        except (KeyError,ValueError,OSError):return None
        if v['max_cost_usd']>remaining:return None
    return v
