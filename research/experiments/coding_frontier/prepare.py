"""Frozen train-split survey; benchmark/heldout text is used only for rejection."""
import ast,collections,gzip,hashlib,json,time
from pathlib import Path
import pyarrow.parquet as pq
from experiments.data_pipeline.pipeline import DuplicateIndex,normalize
from experiments.opencoder_sft.runtime import read,write,sha
from experiments.coding_frontier.interface import interface_stub

ROOT=Path(__file__).resolve().parents[2];REG=ROOT/'lab/experiments/coding-frontier'
ORDER={'apps':['introductory','interview','competition'],'taco':['EASY','MEDIUM','MEDIUM_HARD','HARD','VERY_HARD','UNKNOWN_DIFFICULTY'],'kodcode':['easy','medium','hard']}

def stream(label):
    if label=='apps':
        for i,line in enumerate((ROOT/'lab/experiments/shorterbetter-multidomain/sources/apps.jsonl').open()):yield i,json.loads(line)
    else:
        index=0
        for p in sorted((REG/'sources'/label).rglob('train-*.parquet')):
            for batch in pq.ParquetFile(p).iter_batches(batch_size=128):
                for r in batch.to_pylist():yield index,r;index+=1

def build():
    out=REG/'data-v2'
    if out.exists():raise RuntimeError('Data already prepared; never overwrite')
    out.mkdir()
    exclude=DuplicateIndex(.7);excluded_families=set();sources={}
    base=ROOT/'lab/experiments/shorterbetter-multidomain/data-v5-full'
    for file in ['benchmark-prompts.jsonl','development.jsonl','confirmation.jsonl']:
        p=base/file;sources[str(p.relative_to(ROOT))]=sha(p)
        for r in map(json.loads,p.open()):
            exclude.add(file+':'+r['id'],r.get('text',r.get('problem','')))
            if r.get('family'):excluded_families.add(r['family'])
    # Preserve all later OpenCoder dev/confirmation tasks too, if found.
    for p in sorted((ROOT/'lab/experiments/opencoder-sft').glob('*/data/*.jsonl')):
        if any(s in p.name for s in ['development','confirmation']):
            sources[str(p.relative_to(ROOT))]=sha(p)
            for r in map(json.loads,p.open()):
                problem=r.get('problem') or r.get('instruction')
                if problem:exclude.add(str(p)+':'+r['id'],problem)
    prior=[json.loads(l) for l in (ROOT/'lab/experiments/replay-scaled/20261101-checkpoints-v5/train.jsonl').open()][:768]
    seen_training={normalize(r['problem']) for r in prior if r['task']=='coding'}
    seen_urls={r.get('family') for r in prior if r['task']=='coding'}
    global_seen={};report={};all_rejected=[]
    for label in ORDER:
        kept=[];counts=collections.Counter();difficulty=collections.Counter();seen=set();overlaps=0
        sourcefiles=[REG/'sources'/label/'metadata.json'] if label!='apps' else [ROOT/'lab/experiments/shorterbetter-multidomain/sources/apps.jsonl']
        for p in sourcefiles:sources[str(p.relative_to(ROOT))]=sha(p)
        for index,r in stream(label):
            counts['raw']+=1
            try:
                if label=='kodcode':
                    problem=r['question'];level=r['gpt_difficulty'];family=r['question_id'];spec=dict(kind='python_tests',test=r['test']);refs=[r['solution']]
                    tags=[r['subset']];source_id=r['question_id']
                    if not r['test'].strip():raise ValueError('missing_tests')
                    ast.parse(r['test'])
                else:
                    problem=r['question'];level=r['difficulty'];family=r.get('url') or f'{label}:{index}';source_id=r.get('problem_id',index)
                    spec=json.loads(r['input_output']);spec['kind']='io';refs=json.loads(r['solutions']);tags=r.get('tags',[])
                    if not spec.get('inputs') or len(spec['inputs'])!=len(spec['outputs']):raise ValueError('missing_or_misaligned_tests')
                    if r.get('picture_num') not in (None,'',0,'0'):raise ValueError('image_dependent')
                if not isinstance(problem,str) or not problem.strip():raise ValueError('empty_problem')
                if level not in ORDER[label]:raise ValueError('unknown_difficulty')
                norm=normalize(problem)
                if norm in seen:raise ValueError('duplicate_within_dataset')
                match=exclude.match(problem)
                if family in excluded_families or match:raise ValueError('benchmark_or_heldout_overlap')
                seen.add(norm);identity=label+'-'+str(source_id);other=global_seen.get(norm)
                if other:overlaps+=1
                global_seen.setdefault(norm,identity)
                suffix='\n\nReturn your final solution as one Python 3 code block.'
                if label=='kodcode':
                    try:suffix+='\n\nRequired public interface (replace pass with your implementation):\n```python\n'+interface_stub(refs[0],spec['test'])+'\n```'
                    except (ValueError,SyntaxError,KeyError,IndexError):refs=[]
                if label!='kodcode' and not spec.get('fn_name'):suffix+=' Read from standard input and write to standard output.'
                if label!='kodcode' and spec.get('fn_name'):suffix+=' Implement '+spec['fn_name']+'.'
                starter=r.get('starter_code','')
                if starter:suffix+='\nStarter code:\n'+starter
                row=dict(id=identity,dataset=label,source_index=index,source_id=source_id,difficulty=level,difficulty_rank=ORDER[label].index(level),family=family,
                    problem=problem,prompt=[{'role':'user','content':problem+suffix}],spec=spec,references=refs,tags=tags,
                    parent768_seen_exact=norm in seen_training or family in seen_urls,cross_dataset_duplicate=other,split='train',
                    source_model_difficulty=label=='kodcode')
                kept.append(row);difficulty[level]+=1
            except (ValueError,TypeError,KeyError,SyntaxError) as e:
                reason=str(e)[:120];counts['rejected:'+reason]+=1;all_rejected.append(dict(dataset=label,index=index,reason=reason))
        kept.sort(key=lambda r:(r['difficulty_rank'],hashlib.sha256(r['id'].encode()).hexdigest()))
        p=out/(label+'.jsonl.gz')
        with gzip.open(p,'wt') as f:
            for r in kept:f.write(json.dumps(r)+'\n')
        report[label]=dict(raw=counts.pop('raw'),eligible_before_reference_check=len(kept),difficulty=dict(difficulty),rejections=dict(counts),
            cross_dataset_exact_duplicates=overlaps,parent768_seen_exact=sum(r['parent768_seen_exact'] for r in kept),sha256=sha(p))
        write(out/'inventory.json',dict(datasets=report,source_hashes=sources,updated_at=time.time()))
        print(label,report[label],flush=True)
    with (out/'rejections.jsonl').open('w') as f:
        for r in all_rejected:f.write(json.dumps(r)+'\n')
    write(out/'complete.json',dict(complete=True,datasets=report,source_hashes=sources,grading='reference_checked_on_worker_before_generation',
        overlap_policy='Reject benchmark and historical holdout near matches; flag cross-dataset exact duplicates; training exposure flagged, semantic leakage not ruled out.',
            ordering='Source difficulty ascending; deterministic ID hash within ties; unknown difficulty separate last. No stopping on failure or difficulty.',created_at=time.time()))

if __name__=='__main__':build()
