"""Small live curves and permanent inference snapshots for the paired short runs."""
from collections import Counter,defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import time
from .distributed import atomic_json

DOMAINS=('math','coding','tools','instructions')


def publish_series(output,completed_step):
    output=Path(output);groups=defaultdict(list)
    paths=sorted(output.glob('rank-*/rollouts.jsonl'))
    for path in paths:
        with path.open() as stream:
            for line in stream:
                try:r=json.loads(line)
                except json.JSONDecodeError:continue
                step=r['optimizer_step']
                if r['split']=='train' and step%4==0 and step+4<=completed_step:
                    groups[step].append((r['prompt_id'],r['task'],r['total_tokens'],r['correct']))
    series={d:[] for d in ('overall',*DOMAINS)}
    for step,rows in sorted(groups.items()):
        if (len(paths)!=4 or len(rows)!=512 or len({r[0] for r in rows})!=64
                or set(Counter(r[0] for r in rows).values())!={8}
                or Counter(r[1] for r in rows)!={d:128 for d in DOMAINS}):continue
        for domain in series:
            selected=rows if domain=='overall' else [r for r in rows if r[1]==domain]
            series[domain].append({'iteration':step//4+1,'median_tokens':statistics.median(r[2] for r in selected),
                'mean_tokens':statistics.mean(r[2] for r in selected),'accuracy':statistics.mean(int(r[3]) for r in selected)})
    result={'updated_at':time.time(),'completed_optimizer_step':completed_step,'series':series,
            'scope':'Changing training batches; not held-out evaluation.'}
    atomic_json(output/'training-series.json',result)
    return result


def preserve_snapshot(output,step):
    output=Path(output);source=output/f'checkpoint-{step}'
    target=output/'snapshots'/f'iteration-{step//4:03d}'
    if target.exists():return
    if step%20 or not (source/'model.safetensors').is_file():
        raise ValueError('Snapshot requires a complete five-rollout checkpoint')
    if json.loads((source/'trainer_state.json').read_text())['global_step']!=step:
        raise ValueError('Snapshot step differs from saved model')
    target.parent.mkdir(exist_ok=True);temporary=target.with_name(target.name+'.pending')
    if temporary.exists():shutil.rmtree(temporary)
    temporary.mkdir()
    files={}
    for path in sorted(source.iterdir()):
        if not path.is_file() or path.suffix not in ('.json','.safetensors','.txt','.jinja'):continue
        if path.name=='trainer_state.json':continue
        with path.open('rb') as stream:files[path.name]=hashlib.file_digest(stream,'sha256').hexdigest()
        os.link(path,temporary/path.name)
    if not {'model.safetensors','config.json','tokenizer_config.json'}<=files.keys():
        raise ValueError('Inference snapshot missing required files')
    atomic_json(temporary/'snapshot-manifest.json',{'optimizer_step':step,'rollout':step//4,'files_sha256':files})
    temporary.rename(target)
