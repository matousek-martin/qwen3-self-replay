"""Exactly three independent inference jobs, one per dataset, joint60USD envelope."""
import math
from pathlib import Path
from experiments.opencoder_sft.runtime import read
SCOPE='coding-frontier-768-v1';PREFIX='coding-frontier-';ENTRY='experiments/coding_frontier/run.sh';GPU='NVIDIA GeForce RTX 4090'
DATASETS=('apps','taco','kodcode')

def allocation_spec(args):
    if (args.hardware_scope!=SCOPE or not args.name.startswith(PREFIX) or args.entry!=ENTRY or args.gpu!=GPU
        or args.gpu_count!=1 or args.provider!='vast' or not 1<(args.max_cost_usd or 0)<=20):raise ValueError('Unregistered frontier allocation')
    return dict(scope=SCOPE,gpu=GPU,count=1,max_gpu_rate=.6)

def verify_worker(names,lease):
    if names!=[GPU] or lease.get('gpu')!=GPU or lease.get('gpu_count')!=1 or lease.get('hardware_scope')!=SCOPE or not lease.get('job_name','').startswith(PREFIX):
        raise RuntimeError('Capability survey GPU/lease mismatch')
    return names

def ticket(state,name,job):
    p=Path(state)/'authorizations'/(name+'.json')
    if not name.startswith(PREFIX) or not name.replace('-','').isalnum() or not p.is_file():return None
    v=read(p)
    required=dict(job_name=name,authorized=True,scope=SCOPE,hardware_scope=SCOPE,entry=ENTRY,provider='vast',gpu=GPU,gpu_count=1,
        evaluation_only=True,training_authorized=False,parent_group=768,samples=8,disk_gb=80,combined_cap_usd=60)
    if any(v.get(k)!=x for k,x in required.items()) or v.get('dataset') not in DATASETS:return None
    if any(v.get(k)!=job.get(k) for k in ['provider','hardware_scope','entry','gpu_count','offer_id','disk_gb']):return None
    if (job.get('gpu') or job.get('requested_gpu'))!=GPU or job.get('completion_budget_scope'):return None
    for key,limit in [('max_cost_usd',20),('max_runtime_hours',48),('max_gpu_rate',.6)]:
        value=v.get(key)
        if not isinstance(value,(float,int)) or not math.isfinite(value) or not 0<value<=limit:return None
    if not 1<(job.get('max_cost_usd') or 0)<=v['max_cost_usd'] or not 0<(job.get('max_runtime_hours') or 0)<=v['max_runtime_hours']:return None
    path=Path(v.get('campaign_path','')).resolve()
    if not path.is_relative_to(Path(state).resolve().parent/'coding-frontier') or not path.is_file():return None
    c=read(path);arms=c.get('arms',{})
    if set(arms)!=set(DATASETS) or c.get('cap_usd')!=60 or c.get('authorized') is not True:return None
    arm=arms[v['dataset']];previous=arm.get('previous_attempts',[])
    if arm.get('job_name')!=name and name not in previous:return None
    if sum(a['max_cost_usd'] for a in arms.values())>60:return None
    jobs=read(Path(state)/'ledger.json')['jobs']
    def archived_or_never_started(j):
        absent=j.get('absence_reconciliation',{})
        return j.get('released_at') and (j.get('backup_verified') or (j.get('startup_failed_before_launch') and not j.get('worker_launch_attempted_at'))
            or (j.get('safely_no_pod') and not j.get('pod_id') and j.get('charged_estimate_usd')==0 and
                (not j.get('allocation_attempts') or (j.get('absence_reconciled_at') and absent.get('successful_full_reads',0)>=12 and absent.get('elapsed_seconds',0)>=300))))
    if name in previous:
        return v if archived_or_never_started(jobs.get(name,{})) else None
    spent=0.
    for old in previous:
        j=jobs.get(old,{})
        if not old.startswith(PREFIX+v['dataset']+'-') or not archived_or_never_started(j):return None
        cost=j.get('charged_estimate_usd')
        if not isinstance(cost,(float,int)) or not math.isfinite(cost) or cost<0:return None
        spent+=cost
    if v['max_cost_usd']>round(20-spent,4):return None
    for n,j in jobs.items():
        if n!=name and n not in previous and n.startswith(PREFIX+v['dataset']+'-') and (j.get('pod_id') or j.get('allocation_attempts')):return None
    if not v.get('source_user_instruction'):return None
    return v
