"""Validate committed survey responses before continuing on a replacement GPU."""
import gzip,json,shutil
from pathlib import Path
from experiments.opencoder_sft.runtime import sha

def validate(directory):
    root=Path(directory)
    with (root/'questions.jsonl').open() as f:questions=[json.loads(line) for line in f]
    if len({q['id'] for q in questions})!=len(questions):raise RuntimeError('Duplicate resumed question')
    expected={q['chunk']:q['chunk_sha256'] for q in questions if q.get('chunk')}
    if set(expected)!={p.name for p in (root/'responses').glob('*.jsonl.gz')}:raise RuntimeError('Uncommitted response chunk')
    records=[];seconds=0
    for name,digest in expected.items():
        p=root/'responses'/name
        if sha(p)!=digest:raise RuntimeError('Resume response hash mismatch')
        with gzip.open(p,'rt') as f:rs=[json.loads(line) for line in f]
        if not rs:raise RuntimeError('Empty response chunk')
        seconds+=rs[0]['batch_generation_seconds'];records.extend(rs)
    ids={q['id'] for q in questions}
    if any(r['id'] not in ids for r in records):raise RuntimeError('Uncommitted response')
    for q in questions:
        rs=[r for r in records if r['id']==q['id']]
        if len(rs)!=q['samples'] or (rs and {r['sample'] for r in rs}!=set(range(8))):raise RuntimeError('Incomplete resumed samples')
    return dict(questions=len(questions),answers=len(records),generated_tokens=sum(r['total_tokens'] for r in records),generation_seconds=seconds)

def restore(source,target):
    counts=validate(source);target=Path(target)
    if (target/'questions.jsonl').exists():
        validate(target)
        return counts
    if target.exists() and any(target.iterdir()):raise RuntimeError('Refusing to overwrite partial resume output')
    shutil.copytree(source,target,dirs_exist_ok=True)
    validate(target)
    return counts
