"""All436verified frozen768 answers, canonical exact prompt grouping, no new generation."""
import collections,gzip,hashlib,json,random,shutil
from pathlib import Path
from experiments.opencoder_sft.runtime import read,write,sha,rows
from experiments.coding_frontier.resume import validate
ROOT=Path(__file__).resolve().parents[2]

def prepare(dest):
 dest=Path(dest);dest.mkdir(parents=True,exist_ok=False)
 oldreg=Path(read(ROOT/'lab/experiments/coding-abc/current.json')['directory']);common=read(oldreg/'common-plan.json')
 frontier=Path(read(ROOT/'lab/experiments/coding-frontier/current.json')['directory']);fc=read(frontier/'campaign.json');jobs=read(ROOT/'lab/experiments/cloud/ledger.json')['jobs']
 canonical={};provenance={};raw=0
 for dataset in ('apps','taco','kodcode'):
  name=fc['arms'][dataset]['job_name'];j=jobs[name]
  if not(j.get('released_at') and j.get('backup_verified')):raise RuntimeError('Survey archive unverified')
  out=ROOT/'lab/experiments/cloud'/name/'output/coding-frontier';counts=validate(out);answers=collections.defaultdict(list)
  for p in sorted((out/'responses').glob('*.jsonl.gz')):
   h=sha(p)
   with gzip.open(p,'rt') as f:
    for r in map(json.loads,f):
     if r['correct']:
      raw+=1;answers[r['id']].append(dict(text=r['text'],generated_token_ids=r['generated_token_ids'],correct=True,truncated=False,
       source_job=name,source_question=r['id'],source_chunk=p.name,source_sha256=h,sample=r['sample'],total_tokens=r['total_tokens']))
  launch=read(frontier/dataset/'launch.json');source=Path(launch['release'])/'lab/experiments/data/coding-frontier';need=set(answers);selected={}
  with gzip.open(source/'questions.jsonl.gz','rt') as f:
   for q in map(json.loads,f):
    if q['id'] in need:selected[q['id']]=q;need.remove(q['id'])
    if not need:break
  if need:raise RuntimeError('Missing source task')
  for identity,q in selected.items():
   # Only merge identical prompt AND grading contract; never move an answer to a different interface.
   key=hashlib.sha256(json.dumps([q['prompt'],q['spec']],sort_keys=True).encode()).hexdigest()
   if key not in canonical:canonical[key]=dict(id=identity,task='coding',split='train',dataset=dataset,prompt=q['prompt'],spec=q['spec'],
      family=q.get('family'),own_answers=[],source_question_ids=[],difficulty=q['difficulty'],parent768_seen_exact=q.get('parent768_seen_exact',False),cross_dataset_duplicate=q.get('cross_dataset_duplicate'))
   canonical[key]['own_answers']+=answers[identity];canonical[key]['source_question_ids'].append(identity)
  provenance[dataset]=dict(job=name,counts=counts,correct_answers=sum(map(len,answers.values())),eligible_questions=len(answers),source_plan_sha256=sha(source/'plan.json'))
 if raw!=436:raise RuntimeError('Expected explicitly authorized436 own correct answers')
 coding=sorted(canonical.values(),key=lambda q:hashlib.sha256(('expanded17:'+q['id']).encode()).hexdigest())
 original={q['id']:q for q in rows(ROOT/'lab/experiments/replay-scaled/20261101-checkpoints-v5/train.jsonl')};own=collections.defaultdict(list)
 history=ROOT/'lab/experiments/cloud/replay-scaled-training-20261101t122440z/output/training/training-records.jsonl'
 for r in rows(history):
  if r['group']<=768 and r['task']!='coding' and r['correct'] and not r['truncated']:
   own[r['id']].append(dict(text=r['text'],correct=True,truncated=False,source_job='replay-scaled-training-20261101t122440z',source_group=r['group']))
 retention=[]
 for task in ('math','tools','instructions'):
  ids=sorted((i for i in own if original[i]['task']==task),key=lambda i:hashlib.sha256(('expanded-retain17:'+i).encode()).hexdigest())[:9]
  if len(ids)!=9:raise RuntimeError('Insufficient retention')
  retention += [dict(original[i],own_answers=own[i]) for i in ids]
 train=[]
 for i,q in enumerate(coding):
  train.append(q)
  if i%3==2 and i//3<len(retention):train.append(retention[i//3])
 added={q['id'] for q in train};train += [q for q in retention if q['id'] not in added]
 shutil.copytree(oldreg/'data/adapter',dest/'adapter')
 for filename in ('development.jsonl','baseline.jsonl'):shutil.copy2(oldreg/'data'/filename,dest/filename)
 dev=rows(dest/'development.jsonl')
 if {q['id'] for q in train}&{q['id'] for q in dev}:raise RuntimeError('Training/development overlap')
 (dest/'train.jsonl').write_text(''.join(json.dumps(q)+'\n' for q in train))
 write(dest/'selection.json',dict(own_answers_only=True,raw_correct_coding_answers=raw,coding_tasks=len(coding),retention_tasks=len(retention),
  source=provenance,question_deduplication='Identical prompt plus grading contract; aliases retained. Related nonidentical source tasks flagged.',
  answer_policy='Use every one of436 verified coding answers inB; keep provenance and per-question normalized loss.',
  sft_exposure='One complete pass; one optimizer step per question averaging all its verified answers; no shortest-answer choice.',
  limitations=['One seed; different compute/answer exposure across objectives.','Small repeatedly inspected development monitor; final frozen official benchmark separately.']))
 cfg=dict(common,variant='expanded-v2',steps=len(train),coding_questions=len(coding),retention_questions=len(retention),coding_answers=436,
  generation_cap=16384,all_sft_answers=True,training_deadline=1793856600,overall_deadline=1793862000,final_official_evaluation=True)
 # UTC05:30training /07:00overall on2026-11-05: reserve90min for final evaluation and archive.
 import datetime
 cfg['training_deadline']=datetime.datetime(2026,11,5,5,30,tzinfo=datetime.timezone.utc).timestamp()
 cfg['overall_deadline']=datetime.datetime(2026,11,5,7,tzinfo=datetime.timezone.utc).timestamp()
 cfg['files_sha256']={str(p.relative_to(dest)):sha(p) for p in dest.rglob('*') if p.is_file()}
 return cfg
