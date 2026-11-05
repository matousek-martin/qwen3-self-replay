"""Reusable official baselines keyed by GPU model and evaluation contract, never UUID."""
from __future__ import annotations
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

POLICY = 'gpu-model-and-protocol-cache-v1'
PACKAGES = ('torch','vllm','transformers','tokenizers','numpy','triton','flashinfer-python',
            'evalplus','sympy','nltk','langdetect','rouge-score','absl-py')
FILES = ('records.jsonl', 'results.json')


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def key(contract):
    return hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def versions(freeze=None):
    if freeze is not None:
        values = dict(line.split('==', 1) for line in freeze.splitlines() if '==' in line)
        values = {k.lower().replace('_','-'):v for k,v in values.items()}
        return {name:values.get(name) for name in PACKAGES}
    result = {}
    for name in PACKAGES:
        try:result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:result[name] = None
    return result


def contract(repo, questions, gpu_model, runtime, *, loading='hub', sandbox='rlimit', compat=None):
    from experiments import study_eval as study
    repo, questions = Path(repo), Path(questions)
    paths = [repo/'evaluate.py',repo/'compare.py'] + sorted((repo/'utils/eval').rglob('*.py'))
    if any(not runtime.get(name) for name in ('torch','vllm','transformers','tokenizers','numpy')):
        raise ValueError('Baseline runtime provenance is incomplete')
    return dict(policy=POLICY,gpu_model=gpu_model.strip(),model=study.BASE_MODEL,revision=study.BASE_REVISION,
                loading=loading,cohort_sha256=digest(questions/'questions.jsonl'),
                questions_manifest_sha256=digest(questions/'manifest.json'),samples=8,seed=9001,
                sampling=study.SAMPLING,max_tokens={d:8192 if d=='mbpp' else 4096 for d in study.DATASETS},
                max_model_len=12288,sandbox=sandbox,workers=4,runtime=runtime,vllm_compat=compat,
                evaluator_files_sha256={str(p.relative_to(repo)):digest(p) for p in paths})


def verify_entry(directory, expected=None):
    directory = Path(directory)
    entry = read(directory/'baseline-cache-entry.json')
    if (entry.get('schema_version')!=1 or entry.get('key')!=key(entry['contract'])
            or entry['contract'].get('policy')!=POLICY or set(entry.get('files_sha256',{}))!=set(FILES)
            or (expected is not None and entry['contract']!=expected)):
        raise ValueError('Baseline cache contract mismatch')
    for name, value in entry['files_sha256'].items():
        if (directory/name).is_symlink() or digest(directory/name)!=value:
            raise ValueError('Baseline cache checksum mismatch: '+name)
    result = read(directory/'results.json')
    c = entry['contract']
    if (result.get('adapter') or result.get('failed_benchmarks') or result.get('selection')!='full'
            or result.get('cohort_total')!=2759 or result.get('cohort_sha256')!=c['cohort_sha256']
            or result.get('samples')!=8 or result.get('seed')!=9001
            or result.get('sampling')!=c['sampling'] or result.get('max_tokens')!=c['max_tokens']
            or result.get('max_model_len')!=c['max_model_len']
            or result.get('sandbox',{}).get('mode')!=c['sandbox']):
        raise ValueError('Cache is not a complete official baseline')
    if c['loading']=='hub' and (result.get('model')!=c['model'] or result.get('revision')!=c['revision']):
        raise ValueError('Cache model identity differs from pinned baseline')
    from collections import Counter
    counts=Counter();seen=set()
    with (directory/'records.jsonl').open() as stream:
        for line in stream:
            row=json.loads(line);identifier=(row['dataset'],row['id'],row['sample_index'])
            if (identifier in seen or type(row.get('correct')) is not bool or row.get('grading_error')
                    or row['sample_index'] not in range(8) or type(row.get('output_tokens')) is not int
                    or row['output_tokens']<0):raise ValueError('Invalid cached baseline record')
            seen.add(identifier);counts[(row['dataset'],row['id'])]+=1
    if len(seen)!=22072 or len(counts)!=2759 or any(n!=8 for n in counts.values()):
        raise ValueError('Incomplete cached baseline responses')
    return entry


def publish(source, catalog):
    """Immutable, atomic, first-writer-wins publication; never replace a baseline."""
    source, catalog = Path(source), Path(catalog)
    entry = verify_entry(source)
    catalog.mkdir(parents=True,exist_ok=True)
    destination = catalog/entry['key']
    with (catalog/'.publish.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if destination.exists():
            verify_entry(destination,entry['contract'])
            return destination
        temporary = Path(tempfile.mkdtemp(prefix='.publishing-',dir=catalog))
        try:
            for name in (*FILES,'baseline-cache-entry.json'):
                shutil.copy2(source/name,temporary/name)
            verify_entry(temporary,entry['contract'])
            temporary.rename(destination)
        finally:
            if temporary.exists():shutil.rmtree(temporary)
    return destination


def import_archive(run, output, release, catalog, source_job):
    """Backfill a complete historical baseline without spending GPU time."""
    from experiments import study_eval as study
    run, output, release = Path(run), Path(output), Path(release)
    status = read(run.parent.parent/'status/base.json')
    if status.get('state')!='complete' or status.get('records_sha256')!=digest(run/'records.jsonl'):
        raise ValueError('Historical baseline has no verified completion record')
    result = read(run/'results.json')
    rows = study.read_jsonl(release/'ThinkingCap-Test/questions/questions.jsonl')
    if len(rows)!=2759 or result.get('adapter'):raise ValueError('Historical source is not the full baseline')
    study.validate_records(run,rows,8,digest(release/'ThinkingCap-Test/questions/questions.jsonl'),'full',
                           model=result['model'],revision=result['revision'])
    parent = run.parent.parent
    provenance = read(parent/'provenance.json')
    worker = read(parent/'worker.json') if (parent/'worker.json').exists() else provenance
    identity = status.get('gpu_identity') or worker.get('gpu_identity')
    if not identity or len(identity.split(','))<2:raise ValueError('Historical GPU identity missing')
    loading = 'hub' if result['model']==study.BASE_MODEL else 'local'
    if loading=='hub' and result['revision']!=study.BASE_REVISION:raise ValueError('Unpinned historical model')
    if loading=='local':
        local = read(parent/'baseline-loading.json')
        if local['model']!=study.BASE_MODEL or local['revision']!=study.BASE_REVISION or local['path']!=result['model']:
            raise ValueError('Local historical baseline was not pinned')
    compat_path=output/'vllm-compat.json'
    compat=read(compat_path)['patched_sha256'] if compat_path.exists() else None
    c=contract(release/'ThinkingCap-Test',release/'ThinkingCap-Test/questions',identity.split(',')[1],
               versions((output/'packages.txt').read_text()),loading=loading,sandbox=result['sandbox']['mode'],compat=compat)
    observed=provenance.get('evaluator_files_sha256',provenance.get('config',{}).get('evaluator_files_sha256'))
    if observed is not None and observed!=c['evaluator_files_sha256']:raise ValueError('Historical evaluator source differs')
    # Historical archives are read-only: construct the manifest in a new temporary directory.
    with tempfile.TemporaryDirectory() as tmp:
        staging=Path(tmp)
        for name in FILES:shutil.copy2(run/name,staging/name)
        study.write_json(staging/'baseline-cache-entry.json',dict(schema_version=1,key=key(c),contract=c,
            files_sha256={name:digest(staging/name) for name in FILES},
            source=dict(job=source_job,gpu_identity=identity,finished_at=status.get('finished_at'),
                        original_run=str(run),imported=True)))
        return publish(staging,catalog)


def stage(release, catalog):
    """Include verified shared baselines in new payloads before they are sealed."""
    release, catalog = Path(release), Path(catalog)
    if not (release/'experiments/study_eval.py').exists() or not catalog.exists():return 0
    if 'baseline_cache' not in (release/'experiments/study_eval.py').read_text():return 0
    if not (release/'experiments/baseline_cache.py').exists():
        raise ValueError('Shared baseline cache implementation missing from new payload')
    count=0
    for source in sorted(catalog.iterdir()):
        if not source.is_dir() or source.name.startswith('.'):continue
        verify_entry(source)
        dest=release/'lab/experiments/data/baseline-cache'/source.name
        if dest.exists():verify_entry(dest)
        else:shutil.copytree(source,dest)
        count+=1
    return count


def ingest(output, catalog):
    for manifest in Path(output).glob('**/runs/base/baseline-cache-entry.json'):
        # publish verifies files and is safe while independent backup workers overlap.
        publish(manifest.parent,catalog)


def check_launch(release, entry, catalog, gpu):
    """Prevent stale evaluator payloads from spending money on a known baseline."""
    if not entry:return
    release=Path(release)
    script=release/entry
    contents=script.read_text() if script.is_file() else ''
    if 'eval' not in entry and not any(name in contents for name in ('teacher_full_eval','full_replay_eval','shorterbetter.full_eval','shorterbetter.paired_eval')):return
    runner=release/'experiments/study_eval.py'
    if runner.exists() and ('baseline_cache' not in runner.read_text() or not (release/'experiments/baseline_cache.py').exists()):
        raise RuntimeError('Legacy evaluation payload would regenerate baselines; prepare a new cache-enabled release before renting')
    if not runner.exists():return
    for manifest in Path(catalog).glob('*/baseline-cache-entry.json'):
        item=read(manifest)
        if item['contract']['gpu_model']==gpu:
            staged=release/'lab/experiments/data/baseline-cache'/item['key']
            if not staged.exists():
                raise RuntimeError('Baseline cache updated after payload sealing; restage the verified baseline before renting')


def options(command):
    a=list(map(str,command))
    if len(a)<2 or Path(a[1]).name!='evaluate.py' or '--out' not in a:return None
    out=Path(a[a.index('--out')+1])
    if out.name!='base' or '--adapter' in a or '--smoke' in a or '--regrade' in a:return None
    allowed={'--questions','--out','--samples','--seed','--workers','--model','--revision','--sandbox'}
    values={}
    for i in range(2,len(a),2):
        if a[i] not in allowed or i+1==len(a):raise ValueError('Unsupported baseline arguments; review cache contract')
        values[a[i]]=a[i+1]
    # Smaller development protocols are distinct and are not full official baselines.
    if values.get('--samples')!='8' or values.get('--seed')!='9001':return None
    if values.get('--workers','4')!='4':raise ValueError('Baseline grader worker configuration differs')
    return values


def annotate(out, receipt):
    from experiments import study_eval as study
    study.write_json(out/'baseline-reuse.json',receipt)
    provenance=out.parent.parent/'provenance.json'
    if provenance.exists():
        value=read(provenance)
        same_gpu=(not receipt['reused'] or receipt['source'].get('gpu_identity','').split(',')[0]
                  ==receipt.get('comparison_gpu_identity','unknown').split(',')[0])
        value.update(baseline_policy=POLICY,matched_baseline_regenerated=not receipt['reused'],
                     same_physical_worker_required=False,same_physical_worker=same_gpu,
                     baseline_reference=receipt)
        study.write_json(provenance,value)


def status_fields(run):
    path=Path(run)/'baseline-reuse.json'
    if not path.exists():return {}
    receipt=read(path)
    return dict(baseline_reused=receipt['reused'],baseline_cache_key=receipt['key'],
                baseline_source=receipt['source'],gpu_identity=receipt['source'].get('gpu_identity'))


def runtime_context(repo, requested_sandbox):
    import importlib.util
    identity=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name','--format=csv,noheader'],text=True).strip()
    if '\n' in identity:raise ValueError('Official baseline cache requires a single evaluation GPU')
    spec=importlib.util.spec_from_file_location('_baseline_sandbox',Path(repo)/'utils/eval/sandbox.py')
    sandbox=importlib.util.module_from_spec(spec);spec.loader.exec_module(sandbox)
    mode,_=sandbox.choose_mode(requested_sandbox)
    spec=importlib.util.find_spec('vllm')
    processor=Path(next(iter(spec.submodule_search_locations)))/'v1/engine/output_processor.py'
    return identity,mode,versions(),digest(processor)


def run_or_reuse(command, log_path, timeout, runner):
    from experiments import study_eval as study
    args=options(command)
    if args is None:return False
    source=Path(args['--questions']);rows=study.read_jsonl(source/'questions.jsonl')
    if len(rows)!=2759:return False
    model=args.get('--model',study.BASE_MODEL);revision=args.get('--revision',study.BASE_REVISION if model==study.BASE_MODEL else None)
    out=Path(args['--out']);loading='hub' if model==study.BASE_MODEL else 'local'
    if loading=='hub' and revision!=study.BASE_REVISION:raise ValueError('Baseline model revision changed')
    if loading=='local':
        meta=read(out.parent.parent/'baseline-loading.json')
        if (meta['model']!=study.BASE_MODEL or meta['revision']!=study.BASE_REVISION or meta['path']!=model
                or any(digest(Path(model)/f)!=h for f,h in meta['files_sha256'].items())):
            raise ValueError('Local baseline is not the verified pinned model')
    identity,mode,runtime,compat=runtime_context(Path(command[1]).parent,args.get('--sandbox'))
    c=contract(Path(command[1]).parent,source,identity.split(',')[1],runtime,loading=loading,sandbox=mode,compat=compat)
    wanted=key(c)
    catalogs=[study.ROOT/'data/baseline-cache',study.ROOT/'lab/experiments/baseline-cache']
    if out.exists():raise RuntimeError('Preserve existing baseline output; refusing implicit replacement')
    for catalog in catalogs:
        cached=catalog/wanted
        if not cached.exists():continue
        entry=verify_entry(cached,c)
        out.mkdir(parents=True)
        for name in FILES:shutil.copy2(cached/name,out/name)
        result=read(out/'results.json')
        if (result['model'],result['revision'])!=(model,revision):
            shutil.copy2(out/'results.json',out/'results.source.json')
            result.update(model=model,revision=revision)
            study.write_json(out/'results.json',result)
        study.validate_records(out,rows,8,c['cohort_sha256'],'full',model=model,revision=revision)
        receipt=dict(reused=True,key=wanted,source=entry['source'],comparison_gpu_identity=identity,
                     policy=POLICY,source_files_sha256=entry['files_sha256'])
        annotate(out,receipt)
        Path(log_path).parent.mkdir(parents=True,exist_ok=True)
        with Path(log_path).open('a') as log:log.write('BASELINE CACHE HIT '+wanted+'; no baseline inference launched\n')
        return True
    provenance=out.parent.parent/'provenance.json'
    if provenance.exists() and read(provenance).get('config',{}).get('baseline_cache_required'):
        raise RuntimeError('Required verified baseline cache missing or incompatible: '+wanted+'; paid regeneration disabled')
    Path(log_path).parent.mkdir(parents=True,exist_ok=True)
    with Path(log_path).open('a') as log:log.write('BASELINE CACHE MISS '+wanted+' '+json.dumps(c,sort_keys=True)+'\n')
    runner(command,log_path,timeout)
    result=study.validate_records(out,rows,8,c['cohort_sha256'],'full',model=model,revision=revision)
    if result.get('adapter') or result['sandbox']['mode']!=mode:raise ValueError('Baseline result differs from cache contract')
    origin=dict(gpu_identity=identity,finished_at=study.now(),original_run=str(out))
    lease=study.ROOT/'worker-lease.json'
    if lease.exists():origin['job']=read(lease)['job_name']
    entry=dict(schema_version=1,key=wanted,contract=c,source=origin,
               files_sha256={name:digest(out/name) for name in FILES})
    study.write_json(out/'baseline-cache-entry.json',entry)
    verify_entry(out,c)
    annotate(out,dict(reused=False,key=wanted,source=origin,policy=POLICY))
    return True
