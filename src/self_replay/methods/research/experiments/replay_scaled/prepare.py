"""Prepare deterministic data and a reviewable plan; never allocates GPUs."""
import argparse
import json
import random
from pathlib import Path
from .core import DOMAINS, MODEL, REVISION, balanced, sha, write
from .generation import settings
ROOT=Path(__file__).resolve().parents[2]

def select_rows(rows, per_domain, seed):
    """Retain all existing split-local no-call rows, then fill positive tools."""
    negatives=[r for r in rows if r['task']=='tools' and json.loads(r['answer'])['calls']==[]]
    positives=sorted((r for r in rows if r['task']=='tools' and json.loads(r['answer'])['calls']),key=lambda r:r['id'])
    if len(negatives)>per_domain:raise ValueError('Explicit no-call quota decision required')
    random.Random(seed+31).shuffle(positives)
    tools=negatives+positives[:per_domain-len(negatives)]
    return balanced([r for r in rows if r['task']!='tools']+tools,per_domain,seed)

def prepare(out,activate=True):
    out=Path(out).resolve()
    if (out/'plan.json').exists(): raise ValueError('Registration exists; do not overwrite')
    source=ROOT/'lab/experiments/shorterbetter-multidomain/data-v5-full'
    manifest=json.loads((source/'manifest.json').read_text())
    selected={}
    for split,number in [('train',960),('development',64)]:
        p=source/(split+'.jsonl')
        if sha(p)!=manifest['files'][p.name]: raise ValueError('Frozen source changed')
        rows=[json.loads(x) for x in p.open()]
        if any(r['split']!=split or r['task'] not in DOMAINS for r in rows): raise ValueError('Invalid split')
        selected[split]=select_rows(rows,number,17 if split=='train' else 9127)
        out.mkdir(parents=True,exist_ok=True)
        (out/(split+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in selected[split]))
    if {r['id'] for r in selected['train']} & {r['id'] for r in selected['development']}: raise ValueError('Split overlap')
    plan=dict(schema_version=5,id='replay-scaled-v5-frequent-checkpoints',state='prepared',model=MODEL,revision=REVISION,
              groups=3840,group_size=4,seed=17,gpu_count=2,allowed_gpu_counts=[2,4],
              learning_rate=1e-5,lora_rank=16,replay_weight=.1,replay_capacity=128,
              length_coefficient=.1,optimizer='dr_grpo',scale_rewards=False,max_new_tokens=2048,
              temperature=.6,top_p=.95,top_k=20,gradient_clip=1.,
              generation=settings(),
              generation_benchmark=dict(questions=32,stages=[[256,16],[256,32],[256,64],[2048,32],[2048,64]],
                                        optimizer_updates=0,scientific_result=False),
              milestones=[128,512,1024,1920,3840],save_every=32,validate_every=256,
              monitor_questions=256,monitor_samples=4,reference_questions_per_domain=8,
              reference_policy='Pinned base calibration on training prompts; frozen per-domain means',
              quality_stopping=False,teacher=False,full_evaluation_automatic=False,
              pilot=dict(groups=4,checkpoint=2,generation_token_cap=256,evaluation_samples=1,
                         fixtures='experiments.replay_scaled.pilot',scientific_result=False,maximum_length_backward_check=2048),
              no_call_counts={s:sum(r['task']=='tools' and json.loads(r['answer'])['calls']==[] for r in rows) for s,rows in selected.items()},
              no_call_policy='Include all existing split-local negatives; keep confirmation untouched; no new generated labels.',
              source_manifest_sha256=sha(source/'manifest.json'),
              files_sha256={s+'.jsonl':sha(out/(s+'.jsonl')) for s in selected},
              limitations=['Four-domain data and verifiers differ from historical replay.',
                          'Validation is a development monitor, not official benchmark accuracy.',
                          'Instruction checks cover format, not broad semantic correctness.',
                          'Development no-call coverage is too small to certify that ability.',
                          'Original 2048 training token cap may limit coding success.',
                          'A100 timing and multi-GPU/resume checks are pending.'])
    write(out/'plan.json',plan)
    write(out/'campaign.json',dict(state='prepared',job_name=None,plan='plan.json',telegram_authorized=True,
                                 gpu_count=2,total=3840,full_training_authorized=False))
    if activate:write(out.parent/'current.json',dict(path=str((out/'campaign.json').relative_to(ROOT))))
    return plan

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();print(json.dumps(prepare(a.out),indent=2))
