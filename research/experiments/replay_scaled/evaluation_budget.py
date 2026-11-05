"""One evaluation-only replay checkpoint job, separate from training budgets."""
import json
import math
from pathlib import Path

SCOPE = 'replay-evaluation-v1'
ENTRY = 'experiments/run-replay-checkpoint-eval.sh'
GPU = 'NVIDIA GeForce RTX 4090'
PARENT1280 = 'replay-scaled-vast-resume1024-20261105t075310z'


def startup_retry_budget(previous_jobs, ceiling=4):
    """Reuse only the unspent part of the original evaluation cap after released startup failures."""
    spent=0.
    for job in previous_jobs:
        no_rental=(job.get('safely_no_pod') and not job.get('pod_id') and not job.get('allocation_attempts')
                   and job.get('state')=='allocation_unavailable')
        absence=job.get('absence_reconciliation',{})
        no_rental=no_rental or (job.get('safely_no_pod') and not job.get('pod_id')
            and job.get('state')=='create_absent_reconciled' and job.get('charged_estimate_usd')==0
            and absence.get('successful_full_reads',0)>=12 and absence.get('elapsed_seconds',0)>=300
            and bool(job.get('late_arrival_watch_until')))
        failed_start=(job.get('state')=='failed' and job.get('startup_failed_before_launch') is True
                      and not job.get('worker_launch_attempted_at'))
        charge=job.get('charged_estimate_usd')
        if (not job.get('released_at') or not (no_rental or failed_start)
                or not isinstance(charge,(int,float)) or not math.isfinite(charge) or charge<0):
            raise ValueError('Previous evaluation is not a reconciled pre-launch failure')
        spent+=charge
    remaining=round(ceiling-spent,4)
    if remaining<=1:raise ValueError('Evaluation repair budget exhausted')
    return remaining


def allocation_spec(args):
    if args.name.startswith('replay-evaluation-1024-') or (args.name.startswith('replay-evaluation-1280-') and getattr(args,'provider',None)=='runpod'):
        if (args.hardware_scope!=SCOPE or args.entry!=ENTRY or args.gpu!=GPU or args.gpu_count!=1
                or getattr(args,'provider',None)!='runpod' or not 1<(args.max_cost_usd or 0)<=6):
            raise ValueError('Invalid RunPod1024 evaluation reservation')
        return dict(scope=SCOPE,gpu=GPU,count=1,max_gpu_rate=.99)
    if (args.hardware_scope != SCOPE or not args.name.startswith('replay-evaluation-')
            or args.entry != ENTRY or args.gpu != GPU or args.gpu_count != 1
            or not 1 < (args.max_cost_usd or 0) <= 4):
        raise ValueError('Invalid evaluation-only replay reservation')
    return dict(scope=SCOPE, gpu=GPU, count=1, max_gpu_rate=.5)


def verify_worker(names, lease):
    if (lease.get('hardware_scope') != SCOPE
            or not lease.get('job_name', '').startswith('replay-evaluation-')
            or lease.get('gpu') != GPU or lease.get('gpu_count') != 1 or names != [GPU]):
        raise RuntimeError('Evaluation hardware/entry does not match its lease')
    return names


def ticket(state, name, job):
    p = Path(state)/'authorizations'/(name+'.json')
    if not name.startswith('replay-evaluation-') or not p.is_file():return None
    v = json.loads(p.read_text())
    if v.get('scope')=='replay_runpod_evaluation_v1':return runpod_ticket(state,name,job,v)
    required = dict(job_name=name, authorized=True, scope='replay_vast_bounded_v1',
                    provider='vast', hardware_scope=SCOPE, entry=ENTRY, gpu=GPU,
                    gpu_count=1, phase='evaluation', evaluation_only=True,
                    training_authorized=False)
    if any(v.get(k)!=x for k,x in required.items()):return None
    group=v.get('checkpoint_group')
    if type(group) is not int or group not in (128,256,576,768,850,1280) or not name.startswith(f'replay-evaluation-{group}-'):return None
    if any(v.get(k)!=job.get(k) for k in ['provider','hardware_scope','entry','gpu','gpu_count','offer_id','disk_gb']):return None
    if (not 1<(job.get('max_cost_usd') or 0)<=v.get('max_cost_usd',0)<=4
            or not 0<(job.get('max_runtime_hours') or 0)<=v.get('max_runtime_hours',0)<=6
            or not 0<v.get('max_gpu_rate',0)<=.5 or v.get('disk_gb')!=60
            or type(v.get('offer_id')) is not int or v['offer_id']<=0
            or not v.get('source_user_instruction') or job.get('completion_budget_scope')):return None
    parent=v.get('concurrent_training_job','')
    jobs=json.loads((Path(state)/'ledger.json').read_text())['jobs']
    training=jobs.get(parent,{})
    limit=v.get('account_spending_limit_usd',0)
    concurrent=v.get('concurrent_evaluation_jobs',[])
    if concurrent and (group!=768 or len(concurrent)!=1 or not concurrent[0].startswith('replay-evaluation-576-')
                       or jobs.get(concurrent[0],{}).get('hardware_scope')!=SCOPE):return None
    concurrent_cap=sum(jobs[n].get('max_cost_usd',0) for n in concurrent)
    if group==1280 and not checkpoint1280_budget(state,name,v,jobs):return None
    expected_parent=PARENT1280 if group==1280 else 'replay-scaled-training-20261101t122440z'
    if (parent!=expected_parent
            or training.get('hardware_scope')!='replay-scaled-a100-v1'
            or not isinstance(limit,(int,float)) or not math.isfinite(limit)
            or not v['max_cost_usd']<=limit<=v['max_cost_usd']+training.get('max_cost_usd',0)+concurrent_cap):
        return None
    return v


def runpod_ticket(state,name,job,v):
    group=v.get('checkpoint_group')
    if type(group) is not int or group not in (1024,1280):return None
    required=dict(job_name=name,authorized=True,scope='replay_runpod_evaluation_v1',provider='runpod',hardware_scope=SCOPE,
        entry=ENTRY,gpu=GPU,gpu_count=1,phase='evaluation',evaluation_only=True,training_authorized=False,checkpoint_group=group)
    if not name.startswith(f'replay-evaluation-{group}-') or any(v.get(k)!=x for k,x in required.items()):return None
    if any(v.get(k)!=job.get(k) for k in ['provider','hardware_scope','entry','gpu_count']):return None
    if (job.get('gpu') or job.get('requested_gpu'))!=GPU or job.get('completion_budget_scope'):return None
    if not (1<(job.get('max_cost_usd') or 0)<=v.get('max_cost_usd',0)<=6
            and 0<(job.get('max_runtime_hours') or 0)<=v.get('max_runtime_hours',0)<=6
            and 0<v.get('max_gpu_rate',0)<=.99 and v.get('source_user_instruction')):return None
    parent=v.get('concurrent_training_job')
    if parent!=(PARENT1280 if group==1280 else 'replay-scaled-runpod-resume-20261102t162212z'):return None
    jobs=json.loads((Path(state)/'ledger.json').read_text())['jobs']
    training=jobs.get(parent,{})
    if not training.get('released_at') or not training.get('backup_verified'):return None
    if group==1280 and not checkpoint1280_budget(state,name,v,jobs):return None
    return v


def checkpoint1280_budget(state,name,v,jobs):
    parent=jobs.get(PARENT1280,{})
    if not parent.get('released_at') or not parent.get('backup_verified'):return False
    previous=v.get('previous_attempts',[])
    if len(previous)!=len(set(previous)) or name in previous:return False
    if any(not n.startswith('replay-evaluation-1280-') or n not in jobs for n in previous):return False
    try:remaining=startup_retry_budget([jobs[n] for n in previous],ceiling=6)
    except ValueError:return False
    return v.get('combined_cap_usd')==6 and 1<v.get('max_cost_usd',0)<=remaining
