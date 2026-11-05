"""One bounded two-A100 rental for the two checkpoint256 curriculum branches."""
import math
from pathlib import Path
from experiments.opencoder_sft.runtime import read

SCOPE='coding-curriculum-256-v1'
PREFIX='coding-curriculum-256-'
ENTRY='experiments/coding_curriculum/run.sh'
GPU='NVIDIA A100-SXM4-80GB'
CAP=24

def allocation_spec(args):
    if (args.hardware_scope!=SCOPE or not args.name.startswith(PREFIX) or args.entry!=ENTRY
        or args.gpu!=GPU or args.gpu_count!=2 or args.provider!='vast'
        or not 1<(args.max_cost_usd or 0)<=CAP):
        raise ValueError('Unregistered checkpoint256 curriculum allocation')
    return dict(scope=SCOPE,gpu=GPU,count=2,max_gpu_rate=1.05)

def verify_worker(names,lease):
    if (names!=[GPU,GPU] or lease.get('gpu')!=GPU or lease.get('gpu_count')!=2
        or lease.get('hardware_scope')!=SCOPE or not lease.get('job_name','').startswith(PREFIX)):
        raise RuntimeError('Curriculum requires exactly two registered A100 SXM4 80GB')
    return names

def ticket(state,name,job):
    p=Path(state)/'authorizations'/(name+'.json')
    if not name.startswith(PREFIX) or not name.replace('-','').isalnum() or not p.is_file():return None
    v=read(p)
    required=dict(job_name=name,authorized=True,scope=SCOPE,hardware_scope=SCOPE,entry=ENTRY,
        provider='vast',gpu=GPU,gpu_count=2,parent_group=256,training_authorized=True,
        combined_cap_usd=CAP,disk_gb=80)
    if any(v.get(k)!=x for k,x in required.items()):return None
    if any(v.get(k)!=job.get(k) for k in ('provider','hardware_scope','entry','gpu_count','offer_id','disk_gb')):return None
    if (job.get('gpu') or job.get('requested_gpu'))!=GPU or job.get('completion_budget_scope'):return None
    for k,limit in [('max_cost_usd',CAP),('max_runtime_hours',10),('max_gpu_rate',1.05)]:
        x=v.get(k)
        if not isinstance(x,(float,int)) or not math.isfinite(x) or not 0<x<=limit:return None
    if not 1<(job.get('max_cost_usd') or 0)<=v['max_cost_usd']:return None
    if not 0<(job.get('max_runtime_hours') or 0)<=v['max_runtime_hours']:return None
    cp=Path(v.get('campaign_path','')).resolve()
    if not cp.is_relative_to(Path(state).resolve().parent/'coding-curriculum') or not cp.is_file():return None
    c=read(cp)
    if c.get('authorized') is not True or c.get('cap_usd')!=CAP or c.get('datasets')!=['kodcode','taco']:return None
    previous=c.get('previous_attempts',[]);jobs=read(Path(state)/'ledger.json')['jobs']
    if name!=c.get('job_name') and name not in previous:return None
    spent=0
    for old in previous:
        j=jobs.get(old,{})
        if not (j.get('released_at') and (j.get('backup_verified') or
            (j.get('startup_failed_before_launch') and not j.get('worker_launch_attempted_at')))):return None
        charge=j.get('charged_estimate_usd')
        if not isinstance(charge,(int,float)) or not math.isfinite(charge) or charge<0:return None
        spent+=charge
    if name not in previous and v['max_cost_usd']>round(CAP-spent,4):return None
    if any(n!=name and n not in previous and n.startswith(PREFIX) and
        (j.get('pod_id') or j.get('allocation_attempts')) for n,j in jobs.items()):return None
    return v if v.get('source_user_instruction') else None
