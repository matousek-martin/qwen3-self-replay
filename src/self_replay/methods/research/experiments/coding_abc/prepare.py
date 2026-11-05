"""Reuse committed self answers and screened training tasks; freeze common inputs."""
import collections,gzip,hashlib,json,random,shutil,time
from pathlib import Path
from experiments.opencoder_sft.runtime import read,write,sha,rows
from experiments.coding_frontier.resume import validate
from experiments.coding_abc.logic import check_contract
ROOT=Path(__file__).resolve().parents[2]

def prepare(dest):
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=False)
    frontier=Path(read(ROOT/'lab/experiments/coding-frontier/current.json')['directory']);fc=read(frontier/'campaign.json')
    ledger=read(ROOT/'lab/experiments/cloud/ledger.json')['jobs'];pool={};evidence={};seen=set();seen_ids=set();families=set();train=[]
    for dataset in ('apps','taco','kodcode'):
        name=fc['arms'][dataset]['job_name'];j=ledger[name]
        if not j.get('released_at') or not j.get('backup_verified'):raise RuntimeError('Wait for verified survey release: '+dataset)
        out=ROOT/'lab/experiments/cloud'/name/'output/coding-frontier';counts=validate(out)
        launch=read(frontier/dataset/'launch.json');source=Path(launch['release'])/'lab/experiments/data/coding-frontier'
        with gzip.open(source/'questions.jsonl.gz','rt') as f:questions={q['id']:q for q in map(json.loads,f)}
        answers=collections.defaultdict(list)
        for p in sorted((out/'responses').glob('*.jsonl.gz')):
            with gzip.open(p,'rt') as f:
                for r in map(json.loads,f):
                    if r['correct']:answers[r['id']].append(dict(text=r['text'],generated_token_ids=r['generated_token_ids'],correct=True,
                        truncated=False,source_job=name,source_chunk=p.name,source_sha256=sha(p),sample=r['sample']))
        candidates=sorted(answers,key=lambda x:hashlib.sha256(('abc17:'+x).encode()).hexdigest());selected=[]
        for identity in candidates:
            q=questions[identity];key=hashlib.sha256(json.dumps(q['prompt'],sort_keys=True).encode()).hexdigest()
            if key in seen:continue
            # Flagged cross-dataset duplicates are excluded rather than counted twice.
            if q.get('cross_dataset_duplicate') in seen_ids or q.get('family') in families:continue
            seen.add(key);seen_ids.add(identity);families.add(q.get('family'));selected.append(dict(id=identity,task='coding',split='train',dataset=dataset,prompt=q['prompt'],spec=q['spec'],family=q.get('family'),
                own_answers=answers[identity],difficulty=q['difficulty'],parent768_seen_exact=q.get('parent768_seen_exact',False)))
            if len(selected)==12:break
        if len(selected)!=12:raise RuntimeError('Need12 unique eligible tasks in '+dataset+'; found'+str(len(selected)))
        pool[dataset]=selected;evidence[dataset]=dict(job_name=name,counts=counts,completion_sha256=sha(out/'completion.json') if (out/'completion.json').exists() else None,archive_exit_sha256=sha(out.parent/'exit.json'),
            source_plan_sha256=sha(source/'plan.json'),selected_ids=[q['id'] for q in selected])
    old=ROOT/'lab/experiments/replay-scaled/20261101-checkpoints-v5';original={q['id']:q for q in rows(old/'train.jsonl')}
    historical=ROOT/'lab/experiments/cloud/replay-scaled-training-20261101t122440z/output/training/training-records.jsonl'
    own=collections.defaultdict(list)
    for r in rows(historical):
        if r['group']<=768 and r['task']!='coding' and r['correct'] and not r['truncated']:
            own[r['id']].append(dict(text=r['text'],correct=True,truncated=False,source_job='replay-scaled-training-20261101t122440z',source_group=r['group']))
    retention=[]
    for task in ('math','tools','instructions'):
        ids=sorted((i for i in own if original[i]['task']==task),key=lambda x:hashlib.sha256(('retain17:'+x).encode()).hexdigest())[:4]
        if len(ids)!=4:raise RuntimeError('Insufficient own retention answers')
        retention.extend(dict(original[i],own_answers=own[i]) for i in ids)
    # Every four updates include three coding sources plus one retention example.
    for i in range(12):
        block=[pool[d][i] for d in ('apps','taco','kodcode')]+[retention[i]]
        random.Random(17+i).shuffle(block);train+=block
    devall=rows(old/'development.jsonl');dev=[]
    for task in ('math','coding','tools','instructions'):
        dev+=sorted((q for q in devall if q['task']==task),key=lambda q:hashlib.sha256(('dev17:'+q['id']).encode()).hexdigest())[:16]
    cached=ROOT/'lab/experiments/cloud/coding-replay-768-20261103t103403z/output/training/validations/000000.jsonl'
    baseline=[r for r in rows(cached) if r['id'] in {q['id'] for q in dev} and r['sample']<2]
    if len(baseline)!=128:raise RuntimeError('Cached768 development incomplete')
    for name,data in [('train.jsonl',train),('development.jsonl',dev),('baseline.jsonl',baseline)]:
        (dest/name).write_text(''.join(json.dumps(r)+'\n' for r in data))
    previous=read(ROOT/'lab/experiments/opencoder-mode-probe/20261103t231900z/launch.json')
    source=Path(previous['release'])/'lab/experiments/data/opencoder-mode-probe/adapters/parent768'
    parent=ROOT/'lab/experiments/cloud/replay-scaled-training-20261101t122440z/output/training/checkpoints/group-000768'
    for f,h in read(parent/'complete.json')['files'].items():
        if sha(parent/f)!=h:raise RuntimeError('Corrupted parent768')
    if sha(source/'adapter_model.safetensors')!=sha(parent/'adapter/adapter_model.safetensors'):raise RuntimeError('Wrong parent')
    shutil.copytree(source,dest/'adapter')
    write(dest/'selection.json',dict(frontier=evidence,retention_source_sha256=sha(historical),
        train_ids=[q['id'] for q in train],dev_ids=[q['id'] for q in dev],own_answers_only=True,
        retention_provenance='Own verified historical student answers at group<=768; coding answers frozen768',
        split_policy='Original screened train/development separation retained; frontier candidates screened against historical holdouts; cross-dataset aliases and families deduplicated within selected training tasks',
        limitations=['Small inspected development monitor, not official evaluation or untouched holdout.',
            'Eligibility conditions on at least one correct answer; does not measure learning entirely unsolved tasks.',
            'One seed and48 matched task exposures; compute and assistant-token exposure differ across objectives.',
            'Cached HF baseline reused with same model/runtime/sampling; smaller validation batching may alter numerical trajectories.']))
    return dict(parent_group=768,steps=48,save_every=8,validate_every=16,seed=17,length_coefficient=0,
        learning_rate=1e-5,replay_weight=.1,coding_context=40960,retention_output_cap=2048,development_output_cap=2048,
        group_size=4,policy_loss_normalizer=2048,
        baseline_provenance=dict(path=str(cached),sha256=sha(cached),checkpoint=768,runtime='pinnedHF',
            original_gpu='NVIDIA GeForce RTX 4090',same_gpu_type=True,official_baseline=False,
            caveat='Cached parent768 development from prior4090 coding pilot; development grading versioned separately from official evaluation'),
        files_sha256={str(p.relative_to(dest)):sha(p) for p in dest.rglob('*') if p.is_file()})
