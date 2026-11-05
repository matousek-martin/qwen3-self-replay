"""Exhaustive CPU preparation of pinned sources; preserve historical holdouts.

All raw rows receive an accepted/quarantine disposition. Unsupported labels are
never weakened to fit an existing verifier. No quota truncates accepted pools.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import random
import re
import time

from experiments.data_pipeline.pipeline import DuplicateIndex, canonical, normalize
from experiments.shorterbetter.prepare import MODEL, MODEL_REVISION, sha, write_rows
from .prepare import SEED, DOMAINS, base_row, parsed, prompt_for
from .prepare_xlam import SPLITS, hash_check
from experiments.shorterbetter.jsonl import read_rows as rows
from .rewards import IF_TYPES, verify, code_check

DATA_ID = 'shorterbetter-four-domains-full-v1'
SOURCE_FILES = {'gsm8k':'gsm8k.parquet','apps':'apps.jsonl','tulu':'tulu.parquet'}
REFERENCE_SECONDS = 30


def source_records(sources):
    import pyarrow.parquet as pq
    return {'gsm8k':pq.read_table(sources/'gsm8k.parquet').to_pylist(),
            'apps':rows(sources/'apps.jsonl'),
            'tulu':pq.read_table(sources/'tulu.parquet').to_pylist()}


def source_key(source, index, raw):
    return str(raw.get('id', index)) if source in ('apps','tulu') else str(index)


def make_row(source, index, raw):
    if source == 'gsm8k':
        if '####' not in raw['answer']:raise ValueError('missing_numeric_target')
        value=raw['answer'].rsplit('####',1)[-1].strip()
        row=base_row(source,index,raw['question'],{'task':'math','value':value},'\\boxed{'+value+'}')
    elif source == 'apps':
        try:tests=parsed(raw['input_output']);solutions=parsed(raw['solutions'])
        except (ValueError,SyntaxError,TypeError):raise ValueError('invalid_tests_or_solutions') from None
        if (not isinstance(tests,dict) or not isinstance(solutions,list) or not solutions
                or not isinstance(tests.get('inputs'),list) or not tests['inputs']
                or not isinstance(tests.get('outputs'),list) or len(tests['inputs'])!=len(tests['outputs'])):
            raise ValueError('missing_or_unaligned_tests_solutions')
        if len(canonical(tests).encode())>2_000_000:raise ValueError('test_payload_over_2mb')
        if not all(isinstance(s,str) for s in solutions):raise ValueError('invalid_solution_type')
        row=base_row(source,raw.get('id',index),raw['question'],{'task':'coding','tests':tests},solutions,family=raw['url'] or None)
        row.update(starter_code=raw['starter_code'] or '',difficulty=raw['difficulty'])
    else:
        if not raw['constraints']:raise ValueError('missing_instruction_constraints')
        if any(c not in IF_TYPES for c in raw['constraints']):raise ValueError('unsupported_instruction_constraints')
        row=base_row(source,raw['id'],raw['prompt'],{'task':'instructions','checks':[IF_TYPES[c] for c in raw['constraints']]},raw['messages'][-1]['content'])
    if not row['problem']:raise ValueError('empty_problem')
    return row


def split_family(row):
    # Preserve APPS URL/problem grouping. Abstract numeric constants in the other
    # sources; a conservative heuristic, not recovered upstream template lineage.
    if row['source']=='apps':return 'apps:'+row['family']
    abstract=re.sub(r'\b\d+(?:[.,]\d+)*\b','NUMBER',normalize(row['problem']))
    return row['source']+':'+hashlib.sha256(abstract.encode()).hexdigest()


def assign_split(row, registry):
    family=split_family(row)
    if family in registry:
        if len(registry[family])!=1:raise ValueError('historical_template_split_conflict')
        return next(iter(registry[family]))
    bucket=int(hashlib.sha256(family.encode()).hexdigest()[:8],16)%10
    return 'development' if bucket==0 else 'confirmation' if bucket==1 else 'train'


def validate_reference(row):
    start=time.monotonic();attempts=0
    if row['task']!='coding':
        passed=verify(row['reference'],row['answer'])
        return {'id':row['id'],'passed':passed,'reason':None if passed else 'reference_failed_verifier',
                'reference_sha256':hashlib.sha256(row['reference'].encode()).hexdigest(),'seconds':time.monotonic()-start}
    # Every input/output test is used on each attempted source solution.
    # Avoid an unbounded CPU task on sources with hundreds of failing solutions.
    for i,code in enumerate(row['reference']):
        if time.monotonic()-start>=REFERENCE_SECONDS:
            return {'id':row['id'],'passed':False,'reason':'reference_check_time_budget','attempts':attempts,'seconds':time.monotonic()-start}
        attempts+=1
        if code_check(code,json.loads(row['answer'])['tests']):
            return {'id':row['id'],'passed':True,'solution_index':i,'reference_sha256':hashlib.sha256(code.encode()).hexdigest(),
                    'all_tests':len(json.loads(row['answer'])['tests']['inputs']),'attempts':attempts,'seconds':time.monotonic()-start}
    return {'id':row['id'],'passed':False,'reason':'all_source_references_failed','attempts':attempts,'seconds':time.monotonic()-start}


def checked_parent(parent):
    m=json.loads((parent/'manifest.json').read_text());a=json.loads((parent/'audit.json').read_text())
    if a['status']!='passed' or a['manifest_sha256']!=sha(parent/'manifest.json'):raise ValueError('Stale parent audit')
    hash_check(parent,m['files'])
    for f,h in m['verifier_sha256'].items():
        if sha(Path(__file__).with_name(f))!=h:raise ValueError('Verifier changed')
    return m


def inherited(parent):
    out={s:[] for s in SPLITS}
    for s in SPLITS:
        out[s]=[r for r in rows(parent/(s+'.jsonl')) if r['task']!='tools']
        out[s]+=rows(parent/f'xlam-{s}.jsonl')+rows(parent/f'glaive-{s}.jsonl')
    return out


def prepare(parent,sources,out,workers=4):
    from transformers import AutoTokenizer
    if out.exists() and any(out.iterdir()):raise ValueError('Refusing to overwrite output')
    pm=checked_parent(parent);sm=json.loads((sources/'manifest.json').read_text());hash_check(sources,sm)
    if any(sm[k]!=v for k,v in pm['sources'].items() if k!='xlam'):raise ValueError('Parent source changed')
    raw=source_records(sources);output=inherited(parent)
    known={(r['source'],str(r['source_index'])):r for rs in output.values() for r in rs}
    registry=defaultdict(set)
    for s,rs in output.items():
        for r in rs:
            if r['task']!='tools':registry[split_family(r)].add(s)
    benchmark=rows(parent/'benchmark-prompts.jsonl');index=DuplicateIndex(.8)
    for r in benchmark:index.add('benchmark:'+r['id'],r['text'])
    for rs in output.values():
        for r in rs:index.add(r['id'],r['problem'])
    # Previously seen tool prompts remain rejection references even if replaced.
    for r in rows(parent/'historical-tool-prompts.jsonl'):index.add('historical:'+r['id'],r['text'])
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_REVISION,local_files_only=True)
    quarantine=rows(parent/'quarantine.jsonl');candidates=[];dispositions=[]
    def reject(source,key,reason,**kwargs):
        quarantine.append({'source':source,'source_index':key,'reason':reason,**kwargs})
        dispositions.append({'source':source,'source_index':key,'status':'quarantined','reason':reason})
    for source,records in raw.items():
        for i,r in enumerate(records):
            key=source_key(source,i,r)
            if (source,key) in known:
                dispositions.append({'source':source,'source_index':key,'status':'inherited','id':known[source,key]['id']});continue
            try:row=make_row(source,i,r)
            except (ValueError,KeyError,TypeError) as exc:
                reject(source,key,str(exc) if isinstance(exc,ValueError) else 'invalid_schema');continue
            try:split=assign_split(row,registry)
            except ValueError as exc:reject(source,key,str(exc));continue
            hit=index.match(row['problem'])
            if hit:reject(source,key,hit[1],matched_id=hit[0],similarity=hit[2]);continue
            row.update(prompt=prompt_for(row),split=split,source_revision=sm[SOURCE_FILES[source]]['revision'],split_family=split_family(row))
            row['prompt_tokens']=len(tokenizer.apply_chat_template(row['prompt'],tokenize=True,add_generation_prompt=True,enable_thinking=True))
            if row['prompt_tokens']>1500:reject(source,key,'prompt_over_1500');continue
            candidates.append(row)
        print('Scanned',source,len(records),'new candidates',Counter(r['source'] for r in candidates),flush=True)
    # Hash-bound reusable CPU receipts: only this exact input/verifier snapshot.
    staging=out.with_name(out.name+'-verification');staging.mkdir(exist_ok=True)
    binding={'parent_manifest_sha256':sha(parent/'manifest.json'),'sources':sm,
             'preparation_sha256':sha(__file__),'verifier_sha256':pm['verifier_sha256']}
    binding_path=staging/'binding.json'
    if binding_path.exists() and json.loads(binding_path.read_text())!=binding:raise ValueError('Verification cache binding changed')
    binding_path.write_text(json.dumps(binding,indent=2)+'\n')
    receipt_path=staging/'reference-checks.jsonl'
    receipts={r['id']:r for r in rows(receipt_path)} if receipt_path.exists() else {}
    with ProcessPoolExecutor(max_workers=workers) as pool,receipt_path.open('a') as log:
        futures={pool.submit(validate_reference,r):r['id'] for r in candidates if r['id'] not in receipts}
        for n,future in enumerate(as_completed(futures),1):
            result=future.result();receipts[result['id']]=result
            log.write(canonical(result)+'\n');log.flush()
            if n%100==0:print('References checked',n,'/',len(futures),'passed',sum(r['passed'] for r in receipts.values()),flush=True)
    random.Random(SEED).shuffle(candidates)
    for row in candidates:
        source,key=row['source'],str(row['source_index']);receipt=receipts[row['id']]
        if not receipt['passed']:reject(source,key,receipt['reason']);continue
        hit=index.match(row['problem'])
        if hit:reject(source,key,hit[1],matched_id=hit[0],similarity=hit[2]);continue
        row.pop('reference');row.pop('starter_code',None)
        output[row['split']].append(row);index.add(row['id'],row['problem'])
        dispositions.append({'source':source,'source_index':key,'status':'accepted','id':row['id']})
    for rs in output.values():random.Random(SEED).shuffle(rs)
    monitor=rows(parent/'monitor.jsonl')
    # A larger fixed development panel for meaningful per-domain monitoring.
    panel=list(monitor);panel_ids={r['id'] for r in panel}
    for d in DOMAINS:
        extras=[r for r in output['development'] if r['task']==d and r['id'] not in panel_ids]
        panel.extend(extras[:64-sum(r['task']==d for r in panel)])
    files={s+'.jsonl':rs for s,rs in output.items()}
    files.update({'monitor.jsonl':panel,'monitor-smoke.jsonl':monitor,'benchmark-prompts.jsonl':benchmark,
        'quarantine.jsonl':quarantine,'source-dispositions.jsonl':dispositions,
        'reference-checks.jsonl':list(receipts.values()),'historical-tool-prompts.jsonl':rows(parent/'historical-tool-prompts.jsonl')})
    out.mkdir(parents=True,exist_ok=True)
    for f,rs in files.items():write_rows(out/f,rs)
    domains={s:dict(Counter(r['task'] for r in rs)) for s,rs in output.items()}
    manifest={'schema_version':3,'id':DATA_ID,'seed':SEED,'parent_directory':str(parent.resolve()),
        'parent_manifest_sha256':sha(parent/'manifest.json'),'parent_audit_sha256':sha(parent/'audit.json'),
        'source_directory':str(sources.resolve()),'sources':pm['sources'],'raw_counts':{s:len(rs) for s,rs in raw.items()}|{'xlam':60000},
        'counts':{f:len(rs) for f,rs in files.items()},'domain_counts':domains,
        'rejections':{s:dict(Counter(r['reason'] for r in quarantine if r['source']==s)) for s in (*raw,'xlam')},
        'apps_difficulty':{s:dict(Counter(r.get('difficulty','introductory') for r in rs if r['source']=='apps')) for s,rs in output.items()},
        'instruction_checks_observed':dict(Counter(c for r in output['train'] if r['task']=='instructions' for c in json.loads(r['answer'])['checks'])),
        'files':{f:sha(out/f) for f in files},'verifier_sha256':pm['verifier_sha256'],
        'preparation_code_sha256':sha(__file__),'tokenizer_model':MODEL,'tokenizer_revision':MODEL_REVISION,
        'screening':{'method':'normalized exact and5token-shingle containment','threshold':.8,'zero_semantic_overlap_guarantee':False,
            'historical_partitions_preserved':True,'numeric_template_and_apps_url_split_families':True},
        'reference_policy':{'all_published_apps_tests':True,'maximum_source_solution_search_seconds':REFERENCE_SECONDS,'sandbox_wall_timeout_seconds_per_solution':5,'unsupported_instruction_constraints_quarantined':True},
        'sampling':{'no_dataset_size_quota':True,'historical_balanced_sampler_uses_shortest_domain':True,
            'balanced_full_coverage_rollouts_at_64_prompts':(max(domains['train'].values())+15)//16,
            'unbalanced_full_pass_batches_at_64_prompts':(len(output['train'])+63)//64,
            'launch_registered':False},
        'limitations':pm['limitations']+['Full source scan does not mean all raw rows are valid: unsupported labels, long prompts, failed source solutions and overlaps remain quarantined.',
            'IF constraints without machine-readable parameters are not guessed from reference responses; current supported constraint subset is unchanged.',
            'New numeric-template fingerprints and APPS URLs are heuristic grouping, not proven semantic/template isolation.',
            'Reference search budget exhaustion is a technical exclusion, not evidence that the source problem has no correct solution.',
            'All published APPS tests retained; source tests are not a proof of general algorithm correctness.',
            'No full-coverage training or new GPU run launched. A fixed 100/300-rollout budget will not consume every example.']}
    # Remove parent pilot-size limitation, which is superseded by this full pool.
    manifest['limitations']=[x for x in manifest['limitations'] if not x.startswith('Four-domain mix remains')]
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    audit(out)
    return manifest


def audit(out):
    from transformers import AutoTokenizer
    m=json.loads((out/'manifest.json').read_text());hash_check(out,m['files'])
    parent=Path(m['parent_directory']);pm=checked_parent(parent)
    if sha(parent/'manifest.json')!=m['parent_manifest_sha256'] or sha(parent/'audit.json')!=m['parent_audit_sha256']:raise ValueError('Parent changed')
    sources=Path(m['source_directory']);hash_check(sources,{k:v for k,v in m['sources'].items() if k!='xlam'})
    raw=source_records(sources)
    lookup={s:{source_key(s,i,r):(i,r) for i,r in enumerate(rs)} for s,rs in raw.items()}
    prior=inherited(parent);previous={r['id']:r for rs in prior.values() for r in rs}
    registry=defaultdict(set)
    for s,rs in prior.items():
        for r in rs:
            if r['task']!='tools':registry[split_family(r)].add(s)
    receipts={r['id']:r for r in rows(out/'reference-checks.jsonl')}
    index=DuplicateIndex(.8)
    if sha(out/'benchmark-prompts.jsonl')!=sha(parent/'benchmark-prompts.jsonl'):raise ValueError('Benchmark reference changed')
    for r in rows(out/'benchmark-prompts.jsonl'):index.add('benchmark:'+r['id'],r['text'])
    # Screen final rows independently of preparation order.
    seen={};counts={};selected={};families=defaultdict(set);newfamilies=set()
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_REVISION,local_files_only=True)
    for split in SPLITS:
        rs=rows(out/(split+'.jsonl'));counts[split]=dict(Counter(r['task'] for r in rs))
        if counts[split]!=m['domain_counts'][split]:raise ValueError('Count mismatch')
        for r in rs:
            if r['id'] in seen or r['split']!=split or index.match(r['problem']):raise ValueError('Residual overlap/ID/split violation')
            if r['task']!='tools':families[split_family(r)].add(split)
            if r['id'] in previous:
                if r!=previous[r['id']]:raise ValueError('Historical row changed')
            else:
                i,source=lookup[r['source']][str(r['source_index'])];expected=make_row(r['source'],i,source)
                for field in ('id','source','source_index','problem','answer','family'):
                    if r[field]!=expected[field]:raise ValueError('Source mismatch '+field)
                if r['source_revision']!=m['sources'][SOURCE_FILES[r['source']]]['revision']:raise ValueError('Source revision changed')
                if r['split_family']!=split_family(r) or split!=assign_split(r,registry):raise ValueError('Template assignment changed')
                newfamilies.add(split_family(r))
                if r['prompt']!=prompt_for(expected):raise ValueError('Prompt mismatch')
                tokens=len(tokenizer.apply_chat_template(r['prompt'],tokenize=True,add_generation_prompt=True,enable_thinking=True))
                if tokens!=r['prompt_tokens'] or tokens>1500:raise ValueError('Prompt token mismatch')
                receipt=receipts[r['id']]
                if not receipt['passed']:raise ValueError('Unverified target')
                reference=expected['reference'][receipt['solution_index']] if r['task']=='coding' else expected['reference']
                if hashlib.sha256(reference.encode()).hexdigest()!=receipt['reference_sha256']:raise ValueError('Reference mapping changed')
                if r['task']=='coding':
                    if receipt['all_tests']!=len(json.loads(r['answer'])['tests']['inputs']):raise ValueError('Test count changed')
                elif not verify(reference,r['answer']):raise ValueError('Reference failed audit')
            if 'reference' in r:raise ValueError('Reference leaked into payload')
            seen[r['id']]=split;selected[r['source'],str(r['source_index'])]=r;index.add(r['id'],r['problem'])
    if not previous.keys()<=seen.keys():raise ValueError('Historical rows lost')
    if any(len(families[f])>1 for f in newfamilies):raise ValueError('New template crosses splits')
    dispositions=rows(out/'source-dispositions.jsonl');accounted=set()
    for d in dispositions:
        key=(d['source'],d['source_index'])
        if key in accounted or d['source_index'] not in lookup[d['source']]:raise ValueError('Invalid source accounting')
        accounted.add(key)
        if (d['status']!='quarantined')!=(key in selected):raise ValueError('Disposition mismatch')
    if len(accounted)!=sum(len(rs) for rs in raw.values()):raise ValueError('Source rows unaccounted')
    rejected_xlam={str(r['source_index']) for r in rows(out/'quarantine.jsonl') if r['source']=='xlam'}
    accepted_xlam={key[1] for key in selected if key[0]=='xlam'}
    if rejected_xlam&accepted_xlam or len(rejected_xlam|accepted_xlam)!=60000:raise ValueError('xLAM accounting mismatch')
    dev={r['id']:r for r in rows(out/'development.jsonl')}
    for filename,n in [('monitor.jsonl',64),('monitor-smoke.jsonl',4)]:
        panel=rows(out/filename)
        if len({r['id'] for r in panel})!=len(panel) or Counter(r['task'] for r in panel)!={d:n for d in DOMAINS} or any(r!=dev.get(r['id']) for r in panel):raise ValueError('Monitoring panel mismatch')
    result={'status':'passed','manifest_sha256':sha(out/'manifest.json'),'audit_code_sha256':sha(__file__),'domain_counts':counts,
        'raw_rows_accounted':sum(m['raw_counts'].values()),'all_raw_sources_accounted':True,
        'historical_rows_preserved':len(previous),'new_rows':len(seen)-len(previous),
        'residual_exact_lexical_overlap':0,'new_template_split_violations':0,'semantic_overlap_guarantee':False,
        'inherited_template_families_shared':sum(len(v)>1 for v in registry.values()),
        'references':'New source solutions actually checked during preparation; audit verifies receipt/source hashes. Parent references inherited from verified audit.',
        'training_started':False}
    (out/'audit.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('parent','sources','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--audit-only',action='store_true');a=p.parse_args()
    result=audit(a.out) if a.audit_only else prepare(a.parent,a.sources,a.out,a.workers)
    print(json.dumps({k:result[k] for k in ('status','counts','domain_counts','apps_difficulty','rejections','sampling') if k in result},indent=2))
