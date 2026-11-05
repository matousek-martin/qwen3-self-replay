import math
from pathlib import Path
from experiments.opencoder_sft.runtime import read
SCOPE='coding-abc-768-v1'; PREFIX='coding-abc-'; ENTRY='experiments/coding_abc/run.sh'
GPU='NVIDIA GeForce RTX 4090'; ARMS=('a','b','c')

def allocation_spec(args):
    if (args.hardware_scope!=SCOPE or not args.name.startswith(PREFIX) or args.entry!=ENTRY or args.gpu!=GPU
        or args.gpu_count!=1 or args.provider!='vast' or not 1<(args.max_cost_usd or 0)<=12):
        raise ValueError('Unregistered ABC allocation')
    return dict(scope=SCOPE,gpu=GPU,count=1,max_gpu_rate=.6)

def verify_worker(names,lease):
    if names!=[GPU] or lease.get('gpu')!=GPU or lease.get('gpu_count')!=1 or lease.get('hardware_scope')!=SCOPE or not lease.get('job_name','').startswith(PREFIX):
        raise RuntimeError('ABC requires one registered RTX4090')
    return names

def ticket(state,name,job):
    p=Path(state)/'authorizations'/(name+'.json')
    if not name.startswith(PREFIX) or not name.replace('-','').isalnum() or not p.is_file():return None
    v=read(p)
    required=dict(job_name=name,authorized=True,scope=SCOPE,hardware_scope=SCOPE,entry=ENTRY,provider='vast',gpu=GPU,
        gpu_count=1,training_authorized=True,parent_group=768,combined_cap_usd=36,disk_gb=80)
    if any(v.get(k)!=x for k,x in required.items()) or v.get('arm') not in ARMS:return None
    if any(v.get(k)!=job.get(k) for k in ['provider','hardware_scope','entry','gpu_count','offer_id','disk_gb']):return None
    if (job.get('gpu') or job.get('requested_gpu'))!=GPU or job.get('completion_budget_scope'):return None
    for k,lim in [('max_cost_usd',12),('max_runtime_hours',18),('max_gpu_rate',.6)]:
        x=v.get(k)
        if not isinstance(x,(float,int)) or not math.isfinite(x) or not 0<x<=lim:return None
    if not 1<(job.get('max_cost_usd') or 0)<=v['max_cost_usd'] or not 0<(job.get('max_runtime_hours') or 0)<=v['max_runtime_hours']:return None
    path=Path(v.get('campaign_path','')).resolve()
    if not path.is_relative_to(Path(state).resolve().parent/'coding-abc') or not path.is_file():return None
    c=read(path)
    if c.get('authorized') is not True or c.get('cap_usd')!=36 or set(c.get('arms',{}))!=set(ARMS):return None
    arm=c['arms'][v['arm']];previous=arm.get('previous_attempts',[])
    if sum(a['max_cost_usd'] for a in c['arms'].values())>36:return None
    jobs=read(Path(state)/'ledger.json')['jobs']
    def archived(j):
        absent=j.get('absence_reconciliation',{})
        return j.get('released_at') and (j.get('backup_verified') or
            (j.get('startup_failed_before_launch') is True and not j.get('worker_launch_attempted_at')) or
            (j.get('safely_no_pod') and not j.get('pod_id') and j.get('charged_estimate_usd')==0 and
             j.get('absence_reconciled_at') and absent.get('successful_full_reads',0)>=12 and absent.get('elapsed_seconds',0)>=300))
    if name in previous:return v if archived(jobs.get(name,{})) else None
    if arm['job_name']!=name:return None
    spent=0.
    for old in previous:
        j=jobs.get(old,{})
        if not old.startswith(PREFIX+v['arm']+'-') or not archived(j):return None
        cost=j.get('charged_estimate_usd')
        if not isinstance(cost,(float,int)) or not math.isfinite(cost) or cost<0:return None
        spent+=cost
    if v['max_cost_usd']>round(12-spent,4):return None
    if any(n!=name and n not in previous and n.startswith(PREFIX+v['arm']+'-') and (j.get('pod_id') or j.get('allocation_attempts')) for n,j in jobs.items()):return None
    return v if v.get('source_user_instruction') else None
