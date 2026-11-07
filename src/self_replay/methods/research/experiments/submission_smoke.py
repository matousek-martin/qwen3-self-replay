"""Finite submission smoke authorization, independent of historical campaigns."""
import json, math, time
from pathlib import Path
SCOPE = 'submission-method-smoke-v1'
GPU = 'NVIDIA GeForce RTX 4090'
PREFIX = 'submission-smoke-'
ENTRY = 'experiments/run_method_smokes.sh'

def read(path):
    return json.loads(Path(path).read_text())

def verify_worker(names, lease):
    if (names != [GPU] or lease.get('gpu') != GPU or lease.get('gpu_count') != 1
            or lease.get('hardware_scope') != SCOPE or not lease.get('job_name', '').startswith(PREFIX)):
        raise RuntimeError('Submission smoke requires exactly one registered RTX4090')
    if time.time() >= min(lease['deadline'], lease['hard_deadline']):
        raise RuntimeError('Smoke lease expired')
    return names

def allocation_spec(args):
    if (args.hardware_scope != SCOPE or not args.name.startswith(PREFIX) or args.entry != ENTRY
            or args.gpu != GPU or args.gpu_count != 1 or args.provider != 'vast'
            or not 1 < (args.max_cost_usd or 0) <= 2.5):
        raise ValueError('Invalid submission smoke allocation')
    return dict(scope=SCOPE, gpu=GPU, count=1, max_gpu_rate=.65)

def ticket(state, name, job):
    path=Path(state)/'authorizations'/(name+'.json')
    if not name.startswith(PREFIX) or not path.is_file():return None
    a=read(path)
    required=dict(job_name=name, authorized=True, scope=SCOPE, hardware_scope=SCOPE,
                  gpu=GPU, gpu_count=1, provider='vast', entry=ENTRY, combined_cap_usd=10)
    if any(a.get(k)!=v for k,v in required.items()) or not a.get('source_user_instruction'):return None
    if any(a.get(k)!=job.get(k) for k in ('provider','entry','hardware_scope','gpu_count','offer_id','disk_gb')):return None
    if (job.get('gpu') or job.get('requested_gpu'))!=GPU:return None
    for key,limit in [('max_cost_usd',2.5),('max_runtime_hours',2),('max_gpu_rate',.65)]:
        value=a.get(key)
        if type(value) not in (int,float) or not math.isfinite(value) or not 0<value<=limit:return None
    if not 1<(job.get('max_cost_usd') or 0)<=a['max_cost_usd']:return None
    if not 0<(job.get('max_runtime_hours') or 0)<=a['max_runtime_hours']:return None
    cp=Path(a.get('campaign_path','')).resolve()
    if not cp.is_relative_to(Path(state).resolve().parent/'submission-smoke') or not cp.is_file():return None
    c=read(cp)
    if c.get('authorized') is not True or c.get('cap_usd')!=10 or c.get('max_concurrent')!=5 or name not in c['jobs']:return None
    jobs=read(Path(state)/'ledger.json').get('jobs',{})
    total=0.;active=0
    for key,registration in c['jobs'].items():
        row=jobs.get(key,{})
        if row.get('released_at'):
            charge=row.get('charged_estimate_usd')
            if type(charge) not in (int,float) or not math.isfinite(charge) or charge<0:return None
            total+=charge
        else:
            total+=registration['max_cost_usd'];active+=1
    if total>10.000001 or active>5:return None
    return a
