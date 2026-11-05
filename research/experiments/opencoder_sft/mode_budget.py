"""One bounded inference-only 2x2 diagnostic; never resume SFT."""
import math
from pathlib import Path
from .runtime import read
SCOPE='opencoder-mode-probe-v1'
PREFIX='opencoder-mode-probe-'
ENTRY='experiments/opencoder_sft/mode_probe.sh'
GPU='NVIDIA GeForce RTX 4090'

def allocation_spec(args):
    if (args.hardware_scope!=SCOPE or not args.name.startswith(PREFIX) or args.entry!=ENTRY or args.gpu!=GPU
        or args.gpu_count!=1 or args.provider!='vast' or not 1<(args.max_cost_usd or 0)<=4):
        raise ValueError('Unregistered mode diagnostic allocation')
    return dict(scope=SCOPE,gpu=GPU,count=1,max_gpu_rate=.6)

def verify_worker(names,lease):
    if (names!=[GPU] or lease.get('gpu')!=GPU or lease.get('gpu_count')!=1 or lease.get('hardware_scope')!=SCOPE
        or not lease.get('job_name','').startswith(PREFIX)):raise RuntimeError('Diagnostic GPU/lease mismatch')
    return names

def ticket(state,name,job):
    path=Path(state)/'authorizations'/(name+'.json')
    if not name.startswith(PREFIX) or not name.replace('-','').isalnum() or not path.is_file():return None
    v=read(path)
    required=dict(job_name=name,authorized=True,scope=SCOPE,hardware_scope=SCOPE,entry=ENTRY,provider='vast',gpu=GPU,
        gpu_count=1,evaluation_only=True,training_authorized=False,questions=192,samples=2,conditions=['parent768-off','sft16-off'],disk_gb=60)
    if any(v.get(k)!=x for k,x in required.items()):return None
    if any(v.get(k)!=job.get(k) for k in ['provider','hardware_scope','entry','gpu_count','offer_id','disk_gb']):return None
    if (job.get('gpu') or job.get('requested_gpu'))!=GPU or job.get('completion_budget_scope'):return None
    for key,limit in [('max_cost_usd',4),('max_runtime_hours',4),('max_gpu_rate',.6)]:
        value=v.get(key)
        if not isinstance(value,(float,int)) or not math.isfinite(value) or not 0<value<=limit:return None
    if not 1<(job.get('max_cost_usd') or 0)<=v['max_cost_usd'] or not 0<(job.get('max_runtime_hours') or 0)<=v['max_runtime_hours']:return None
    if type(v.get('offer_id')) is not int or v['offer_id']<=0 or not v.get('source_user_instruction'):return None
    jobs=read(Path(state)/'ledger.json')['jobs']
    previous=v.get('previous_attempts',[])
    if len(previous)!=len(set(previous)) or name in previous:return None
    spent=0.
    for prior in previous:
        j=jobs.get(prior,{})
        charge=j.get('charged_estimate_usd')
        if (not prior.startswith(PREFIX) or not j.get('released_at') or not j.get('startup_failed_before_launch')
            or j.get('worker_launch_attempted_at') or not isinstance(charge,(float,int))
            or not math.isfinite(charge) or charge<0):return None
        spent+=charge
    if v['max_cost_usd']>round(4-spent,4):return None
    if any(n!=name and n not in previous and n.startswith(PREFIX) and (j.get('pod_id') or j.get('allocation_attempts')) for n,j in jobs.items()):return None
    return v
