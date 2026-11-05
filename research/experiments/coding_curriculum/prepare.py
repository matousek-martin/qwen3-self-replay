"""Screen training-only curricula without exposing reference solutions to training."""
import collections,gzip,hashlib,json,shutil
from pathlib import Path
from experiments.opencoder_sft.runtime import read,write,sha,rows
from experiments.coding_frontier.grading import check
from experiments.coding_curriculum.budget import SCOPE

ROOT=Path(__file__).resolve().parents[2]
PARENT=ROOT/'lab/experiments/cloud/replay-scaled-training-20261101t122440z/output/training/checkpoints/group-000256'
SOURCE=ROOT/'lab/experiments/replay-scaled/20261101-checkpoints-v5'
QUOTAS={'kodcode':{'easy':16,'medium':10,'hard':6},
        'taco':{'EASY':16,'MEDIUM':8,'MEDIUM_HARD':4,'HARD':2,'VERY_HARD':2}}

def verify_parent(path=PARENT):
    m=read(path/'complete.json')
    if m['group']!=256 or m['world_size']!=2 or m['pilot']:raise ValueError('Wrong parent checkpoint')
    for f,h in m['files'].items():
        if sha(path/f)!=h:raise ValueError('Corrupted parent: '+f)
    return m

def prepare(dest):
    from transformers import AutoTokenizer
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=False);verify_parent()
    tok=AutoTokenizer.from_pretrained(PARENT/'adapter',local_files_only=True)
    src=ROOT/'lab/experiments/coding-frontier/data-v2';inventory=read(src/'inventory.json')
    # Preserve the original next128 questions outside coding, including their positions.
    historical=rows(SOURCE/'train.jsonl')[256:384]
    assert collections.Counter(r['task'] for r in historical)==dict(math=32,coding=32,tools=32,instructions=32)
    dev=rows(SOURCE/'development.jsonl')[:64]
    assert collections.Counter(r['task'] for r in dev)==dict(math=16,coding=16,tools=16,instructions=16)
    for dataset,quotas in QUOTAS.items():
        path=src/(dataset+'.jsonl.gz')
        if sha(path)!=inventory['datasets'][dataset]['sha256']:raise ValueError('Source hash mismatch')
        selected=[];counts=collections.Counter();rejected=collections.Counter();families=set();proof=[]
        with gzip.open(path,'rt') as f:
            for q in map(json.loads,f):
                level=q['difficulty']
                if level not in quotas or counts[level]>=quotas[level]:continue
                if q['parent768_seen_exact'] or q['family'] in families:rejected['prior_training_or_duplicate_family']+=1;continue
                ids=tok.apply_chat_template(q['prompt'],tokenize=True,add_generation_prompt=True,enable_thinking=True)
                if len(ids)>2048:rejected['prompt_over_2048']+=1;continue
                checked=[]
                for code in q['references'][:3]:
                    result=check(code,q['spec']);checked.append(result)
                    if result['status']=='passed':break
                if not checked or checked[-1]['status']!='passed':rejected['reference_not_verified']+=1;continue
                selected.append({k:q[k] for k in ('id','dataset','source_index','source_id','difficulty','difficulty_rank','family','problem','prompt','spec','split')})
                selected[-1]['task']='coding';families.add(q['family']);counts[level]+=1
                proof.append(dict(id=q['id'],reference_checks=checked,prompt_tokens=len(ids)))
                if dict(counts)==quotas:break
        if dict(counts)!=quotas:raise RuntimeError('Insufficient verified curriculum: '+dataset+str(counts))
        assert [q['difficulty_rank'] for q in selected]==sorted(q['difficulty_rank'] for q in selected)
        it=iter(selected);train=[next(it) if r['task']=='coding' else r for r in historical]
        if {q['id'] for q in train}&{q['id'] for q in dev}:raise ValueError('Split overlap')
        out=dest/dataset;out.mkdir();shutil.copytree(PARENT,out/'parent')
        for name,rs in [('train',train),('development',dev)]:
            (out/(name+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rs))
        write(out/'selection.json',dict(dataset=dataset,counts=dict(counts),rejected=dict(rejected),reference_checks=proof,
            source_sha256=sha(path),source_metadata=read(ROOT/'lab/experiments/coding-frontier/sources'/dataset/'metadata.json'),
            source_license='cc-by-nc-4.0' if dataset=='kodcode' else 'apache-2.0',
            overlap_screen=inventory['source_hashes'],difficulty='Published source labels, not measured parent256 difficulty',
            reference_policy='Reference solutions used only to validate sandbox tests; omitted from model inputs and training targets',
            caveats=['Near-duplicate screening is heuristic, not proof of no semantic overlap.',
                'Difficulty labels across datasets are not calibrated.', 'Historical576 comparison is not a matched128-step control.']))
        cfg=read(SOURCE/'plan.json')
        cfg.update(id=SCOPE+'-'+dataset,branch=SCOPE,dataset=dataset,parent_group=256,groups=128,
            milestones=[64,128],save_every=16,validate_every=64,monitor_questions=64,
            full_evaluation_automatic=False,parent_state_policy='Inherit adapter, Adam, replay buffer, replay RNG and rank RNG; reset branch cursor and development reference.',
            files_sha256={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()})
        write(out/'plan.json',cfg)
    write(dest/'summary.json',dict(parent_group=256,parent_manifest_sha256=sha(PARENT/'complete.json'),
        branches={d:dict(iterations=128,coding_questions=32,difficulty_quotas=q) for d,q in QUOTAS.items()}))

if __name__=='__main__':
    import sys
    prepare(sys.argv[1])
