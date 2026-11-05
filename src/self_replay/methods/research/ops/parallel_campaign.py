"""Parallel registered seeds, with one shared allocator and unchanged job identity."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import fcntl
import json
import threading
import time
from completion_jobs import run_job
from experiments import cloud


def run(release,directory,config,payload,workers):
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=directory/'campaign.json'
        state=json.loads(path.read_text()) if path.exists() else {
            'release':str(release),'payload_sha256':payload,'created_at':time.time(),'state':'prepared',
            'trials':{str(s):{'job':f"{config['id']}-seed{s}-{payload[:10]}",'state':'queued'} for s in config['seeds']}}
        if state['payload_sha256']!=payload:raise ValueError('Campaign release changed')
        mutex=threading.Lock()
        def update(seed,**values):
            with mutex:
                state['trials'][str(seed)].update(values);state['updated_at']=time.time();cloud.write(path,state)
        def trial(seed):
            row=state['trials'][str(seed)]
            try:
                while True:
                    name=row['job'];update(seed,state='running')
                    try:
                        output=run_job(release,name,f'experiments/run-isolated-seed{seed}.sh',config['hours'])
                        break
                    except Exception:
                        ledger=json.loads((cloud.STATE/'ledger.json').read_text())
                        failed=ledger['jobs'].get(name,{})
                        # A bounded, clean capacity refusal is a queue condition.
                        # Preserve it under its own identity; never retry an
                        # ambiguous allocation, a running pod, or failed work.
                        if not (failed.get('state')=='allocation_unavailable' and failed.get('safely_no_pod')
                                and failed.get('released_at') and not failed.get('pod_id')):
                            raise
                        attempts=list(row.get('capacity_refusals',[]))+[name]
                        replacement=f"{config['id']}-seed{seed}-{payload[:10]}-capacity{len(attempts)+1}"
                        update(seed,state='queued_capacity',job=replacement,capacity_refusals=attempts)
                        time.sleep(30)
                role=json.loads((output/'role-result.json').read_text())
                if role.get('seed')!=seed or not role.get('development_complete'):raise ValueError('Incomplete matched seed')
                update(seed,state='complete',backup_verified=True,gpu_released=True)
            except Exception as exc:
                update(seed,state='needs_inspection',error=str(exc));raise
        state['state']='running';state['parallel_seeds']=workers;cloud.write(path,state)
        errors=[]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures=[]
            for seed in config['seeds']:
                if state['trials'][str(seed)]['state']=='complete':continue
                future=pool.submit(trial,seed);futures.append(future)
                # Let each dispatch acquire a real/pending provider identity
                # before another seed races for the same visible free slot.
                while not future.done():
                    ledger=json.loads((cloud.STATE/'ledger.json').read_text())
                    name=state['trials'][str(seed)]['job']
                    job=ledger['jobs'].get(name,{})
                    if job.get('pod_id') or job.get('state')=='creating':break
                    time.sleep(2)
            for future in as_completed(futures):
                try:future.result()
                except Exception as exc:errors.append(str(exc))
        state.update(state='needs_inspection' if errors else 'development_complete',errors=errors,updated_at=time.time())
        cloud.write(path,state)
        if errors:raise RuntimeError('One or more parallel seeds need inspection')
