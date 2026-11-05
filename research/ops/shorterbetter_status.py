"""Read and record the current ShorterBetter worker; never allocate or restart."""
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import cloud

REMOTE = r'''
import json
from pathlib import Path
base=Path('/workspace/project/output')
def read(p):
    try: return json.loads(p.read_text())
    except (FileNotFoundError,ValueError): return None
def tail(p,size=5000):
    if not p.exists(): return ''
    with p.open('rb') as f:
        f.seek(max(0,p.stat().st_size-size));return f.read().decode(errors='replace')
out={'phase':read(base/'phase.json'),'exit':read(base/'exit.json'),'launch_failure':read(base/'launch-failure.json')}
for stage in ('startup-check','engineering-check','deterministic-reference','resume-check','training'):
    folder=base/stage
    if not folder.exists(): continue
    state={n:read(folder/(n+'.json')) for n in ('progress','decision','training-finished')}
    state['ranks']={}
    for rank in sorted(folder.glob('rank-*')):
        values={n:read(rank/(n+'.json')) for n in ('gpu-memory-check','gpu-sampling-check','precision-check','parameter-update-check')}
        update=tail(rank/'parameter-updates.jsonl').strip().splitlines()
        if update:
            try: values['last_update']=json.loads(update[-1])
            except ValueError: pass
        # Fingerprints are large; return only status/counts for the startup check.
        if values['parameter-update-check']:
            values['parameter-update-check']={k:v for k,v in values['parameter-update-check'].items() if k in ('status','optimizer_steps','changed_parameter_tensors','total_parameter_tensors')}
        state['ranks'][rank.name]=values
    state['failures']=[read(p) for p in folder.glob('failure-rank-*.json')]
    curve=read(folder/'learning-curve.json')
    if curve: state['last_validation']=curve['checkpoints'][-1]
    out[stage]=state
out['log_tail']=tail(base/'runner.log')
print(json.dumps(out))
'''


def main():
    current=json.loads((ROOT/'lab/experiments/shorterbetter-scaled/current.json').read_text())
    name=current['job_name']
    job=json.loads((cloud.STATE/'ledger.json').read_text())['jobs'][name]
    observation={'job_name':name,'checked_at':time.time(),'ledger':{k:job.get(k) for k in
        ('state','pod_id','created_at','released_at','release_reason','backup_verified','rate','error','max_cost_usd','hard_deadline')}}
    if not job.get('released_at') and job.get('pod_id'):
        pod=cloud.api('pods/'+job['pod_id'])
        observation['probe']=cloud.worker_probe(pod)
        observation['worker']=json.loads(cloud.remote(pod,"python3 - <<'PY'\n"+REMOTE+'\nPY',timeout=35))
    directory=ROOT/current['directory']/'observations'
    cloud.write(directory/(time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'.json'),observation)
    worker=observation.get('worker',{})
    summary={**observation,'probe':{k:observation.get('probe',{}).get(k) for k in ('worker_status','gpu','health')}}
    summary['worker']={k:v for k,v in worker.items() if k not in ('log_tail','startup-check','engineering-check','deterministic-reference','resume-check','training')}
    for stage in ('startup-check','engineering-check','deterministic-reference','resume-check','training'):
        if stage in worker:
            s=worker[stage]
            summary['worker'][stage]={k:v for k,v in s.items() if k!='ranks'}
            summary['worker'][stage]['ranks']={r:{k:v for k,v in info.items() if k in ('last_update','parameter-update-check','precision-check')} for r,info in s['ranks'].items()}
    summary['log_tail']=worker.get('log_tail','')[-2400:]
    print(json.dumps(summary,indent=2,ensure_ascii=False))


if __name__=='__main__':main()
