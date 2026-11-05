"""Build a versioned, reviewable SFT plan from CPU-verified data. No GPU launch."""
import collections
import hashlib
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

from .prepare import ROOT, PARENT, sha, write, rows_write, balanced
from experiments.shorterbetter.multidomain.rewards import verify


def finalize(reg):
    from transformers import AutoTokenizer
    reg=Path(reg);out=reg/'prepared'
    if out.exists():raise ValueError('Prepared experiment already exists')
    out.mkdir()
    manifest=json.loads((reg/'data/manifest.json').read_text())
    for f,h in manifest['files_sha256'].items():
        if sha(reg/'data'/f)!=h:raise ValueError('Data changed: '+f)
    parent=json.loads((PARENT/'complete.json').read_text())
    for f,h in parent['files'].items():
        if sha(PARENT/f)!=h:raise ValueError('Parent checkpoint changed: '+f)
    tok=AutoTokenizer.from_pretrained(PARENT/'adapter',local_files_only=True)
    limits=collections.Counter();lengths=[]
    def fits(row):
        ids=tok.apply_chat_template(row['messages'],tokenize=True,add_generation_prompt=False)
        n=len(ids);lengths.append(n)
        if n>4096:limits['over4096']+=1;return False
        return True
    reservoir=[r for r in map(json.loads,(reg/'data/train-reservoir.jsonl').open()) if fits(r)]
    coding=balanced(reservoir,2000)
    if len(coding)!=2000:raise ValueError('Insufficient token-compatible training data')
    rows_write(out/'coding-reservoir.jsonl',reservoir)
    rows_write(out/'coding-2000.jsonl',coding)
    # Existing independent development data and all source tests are preserved.
    for s in ['development','confirmation']:shutil.copyfile(reg/f'data/{s}-verified.jsonl',out/f'{s}-verified.jsonl')
    retention_source=ROOT/'lab/experiments/replay-scaled/20261101-checkpoints-v5/train.jsonl'
    old=json.loads((retention_source.parent/'plan.json').read_text())
    if sha(retention_source)!=old['files_sha256']['train.jsonl']:raise ValueError('Retention source changed')
    records=ROOT/'lab/experiments/cloud/replay-scaled-runpod-resume-20261102t162212z/output/training/training-records.jsonl'
    own={}
    for response in map(json.loads,records.open()):
        if response['group']<=768 and response['correct'] and not response['truncated']:
            own.setdefault(response['id'],response['text'])
    pools=collections.defaultdict(list)
    for r in map(json.loads,retention_source.open()):
        if r['task']=='coding' or r['split']!='train':continue
        tag='no_call' if r['task']=='tools' and json.loads(r['answer'])['calls']==[] else r['task']
        # Reuse existing verified own traces, not newly generated answers.
        # No-call gold is the existing deterministic empty-call label.
        answer='[]' if tag=='no_call' else own.get(r['id'])
        if answer is None or not verify(answer,r['answer']):continue
        if '</think>' not in answer:answer='<think>\n\n</think>\n\n'+answer
        row={'id':r['id'],'family':r.get('split_family',r['id']),'category':tag,'split':'train',
             'messages':r['prompt']+[{'role':'assistant','content':answer}],
             'source':r['source'],'source_revision':r['source_revision']}
        if fits(row):pools[tag].append(row)
    sizes={'math':128,'instructions':128,'tools':84,'no_call':64}
    retention=[]
    for k,n in sizes.items():
        pool=sorted(pools[k],key=lambda r:hashlib.sha256(('17:'+r['id']).encode()).hexdigest())
        if len(pool)<n:raise ValueError('Insufficient retention '+k)
        retention.extend(pool[:n])
    rows_write(out/'retention.jsonl',retention)
    # Every effective32-example optimizer batch has24coding+8retention.
    # Pad coding exposure with16documented repeats, preserving2000distinct tasks.
    batches=84; schedule=[]
    queues={k:[r for r in retention if r['category']==k] for k in sizes}
    for i in range(batches):
        retention_block=[queues[k][(i*3+j)%len(queues[k])] for k in ('math','instructions') for j in range(3)]
        retention_block+=[queues['tools'][i],queues['no_call'][i%len(queues['no_call'])]]
        block=[coding[(i*24+j)%len(coding)] for j in range(24)]+retention_block
        schedule.extend(block)
    rows_write(out/'train-schedule.jsonl',schedule)
    lineage={'parent_checkpoint':768,'path':str(PARENT.relative_to(ROOT)),
             'complete_sha256':sha(PARENT/'complete.json'),
             'adapter_sha256':sha(PARENT/'adapter/adapter_model.safetensors'),
             'initialization':'Load existing adapter trainable; fresh optimizer. Not exact RL continuation.'}
    swift=json.loads((reg/'swift-source-lock.json').read_text())
    recipe={'model':'Qwen/Qwen3-0.6B','model_revision':'c1899de289a04d12100db370d81485cdf75e47ca',
        'adapters':str(PARENT/'adapter'),'tuner_type':'lora','torch_dtype':'bfloat16',
        'dataset':str(out/'train-schedule.jsonl'),'dataset_shuffle':False,'train_dataloader_shuffle':False,
        'split_dataset_ratio':0,'num_train_epochs':1,'learning_rate':1e-5,'warmup_ratio':.05,
        'per_device_train_batch_size':2,'gradient_accumulation_steps':4,'max_length':4096,
        'loss_scale':'ignore_empty_think','save_steps':16,'save_only_model':False,
        'logging_steps':1,'seed':17,'data_seed':17,'report_to':'none',
        'output_dir':'output/opencoder-sft-768','packing':False}
    plan={'schema_version':1,'id':'opencoder-sft-768-v1','state':'data_prepared_runtime_pending',
        'parent':lineage,'training_method':'supplied-solution supervised fine-tuning',
        'new_model_generation':False,'external_teacher_api':False,
        'source_answers':'Published OpenCoder reference code; supplied synthetic solutions, not self-replay.',
        'coding_distinct':2000,'coding_exposures':2016,'retention_distinct':len(retention),
        'training_rows':len(schedule),'optimizer_steps':84,'epochs':1,
        'reservoir_distinct':len(reservoir),'automatic_expansion':False,
        'hardware':{'provider_preference':'vast','gpu_model':'NVIDIA A100-SXM4-80GB','gpu_count':4,
            'user_allows_more_a100':True,'eight_gpu_configuration_measured':False},
        'budget':{'proposed_total_ceiling_usd':50,'allocated_usd':0,'launch_requested':False,
            'initial_quote_required':True,'runtime_estimate_not_measured_hours':[3,6]},
        'swift':{'repository':swift['repo'],'revision':swift['revision'],'config':recipe},
        'monitoring':{'checkpoint_every_optimizer_steps':16,'preserve_all_checkpoints':True,
            'development_at_steps':[0,16,32,48,64,80,84],
            'full_development_at_steps':[0,84],'routine_telegram':'completed validation only',
            'timezone':'Europe/Prague','quality_stopping':False,'numerical_budget_stopping':True,
            'ui_units':'distinct coding questions /2000 AND optimizer updates /84; no mixed counters'},
        'evaluation':{'official':'2759questions x8answers; frozen protocol; candidate on4090',
            'baseline':'mandatory compatible cached baseline','select':'development only before final official evaluation',
            'report':['coding accuracy','all other domains','total tokens','truncation','paired prompt intervals']},
        'token_audit':{'limit':4096,'rejected':dict(limits),'max_observed':max(lengths),
            'method':'Parent768tokenizer; no silent output truncation'},
        'retention_counts':sizes,'retention_exposures':{'math':252,'instructions':252,'tools':84,'no_call':84},
        'data_manifest_sha256':sha(reg/'data/manifest.json'),
        'retention_source_sha256':sha(retention_source),
        'retention_existing_own_traces_sha256':sha(records),
        'changes_from_previous_pilot':['RL+replay→SFT','54codingproblems→2000distinct',
            'own generated answers→published verified code','empty think prefix loss masked',
            'single4090→4A100scoped exception','balanced retention per effective batch'],
        'launch_gates_pending':['Pin and test full training runtime dependency set',
            'Allocator and worker scoped hardware/budget lease enforcement',
            'Actual768adapter equivalence, masked labels, four-rank batching and checkpoint resume GPU smoke',
            'Training validation/Telegram hooks and official evaluator adapter registration',
            'Reconcile live capacity/account reservation and finite authorization before renting'],
        'limitations':manifest['limitations']+['2000is a pilot size, not an established optimum.',
            'Synthetic reference solutions can still regress accuracy; historical teacher failures remain relevant.',
            'Source tests plus null-function control are not independent correctness proofs.']}
    plan['files_sha256']={f.name:sha(f) for f in out.glob('*.jsonl')}
    plan['source_sha256']={str(f.relative_to(ROOT)):sha(f) for f in (ROOT/'experiments/opencoder_sft').glob('*.py')}
    write(reg/'plan.json',plan)
    write(reg/'campaign.json',{'state':plan['state'],'job_name':None,'parent_checkpoint':768,'gpu_allocated':False,
        'created_at':datetime.now(timezone.utc).isoformat(),'plan_sha256':sha(reg/'plan.json')})
    rows_write(reg/'events.jsonl',[{'at':datetime.now(timezone.utc).isoformat(),'event':'preparation_completed',
        'plan_sha256':sha(reg/'plan.json'),'parent':768,'generation':False,'allocated_usd':0}])
    write(reg.parent/'current.json',{'directory':str(reg.resolve())})
    print(json.dumps({'coding':2000,'reservoir':len(reservoir),'retention':len(retention),'steps':84,'state':plan['state']}))


if __name__=='__main__':
    import sys
    finalize(Path(sys.argv[1]).resolve())
