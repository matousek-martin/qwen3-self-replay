"""Build immutable four-domain data, screen benchmark overlap and verify targets."""
import argparse,ast,collections,hashlib,json,random,re,sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from experiments.data_pipeline.pipeline import DuplicateIndex,canonical,normalize,prompt_text
from experiments.shorterbetter.prepare import sha,write_rows,MODEL,MODEL_REVISION
from .rewards import IF_TYPES,instruction_check,code_check,verify
DOMAINS=('math','coding','tools','instructions');SEED=20261028

def parsed(text):
    try:return json.loads(text)
    except (ValueError,TypeError):return ast.literal_eval(text)

def base_row(source,index,problem,spec,reference,family=None):
    ident=source+'-'+hashlib.sha256((str(index)+'\n'+problem).encode()).hexdigest()[:24]
    return {'id':ident,'source':source,'source_index':index,'problem':problem.strip(),
      'answer':canonical(spec),'task':spec['task'],'reference':reference,'family':family or ident}

def raw_candidates(sources):
    import pyarrow.parquet as pq
    for i,r in enumerate(pq.read_table(sources/'gsm8k.parquet').to_pylist()):
        value=r['answer'].rsplit('####',1)[-1].strip()
        yield base_row('gsm8k',i,r['question'],{'task':'math','value':value},'\\boxed{'+value+'}')
    for i,line in enumerate((sources/'apps.jsonl').open()):
        r=json.loads(line)
        if r['difficulty']!='introductory':continue
        try:tests=parsed(r['input_output']);solutions=parsed(r['solutions'])
        except (ValueError,SyntaxError,TypeError):continue
        if not tests or not solutions or not tests.get('inputs') or len(tests['inputs'])!=len(tests.get('outputs',[])):continue
        # Retain every published test for each selected problem, not a convenient subset.
        if len(tests['inputs'])>100 or len(canonical(tests))>200000:continue
        spec={'task':'coding','tests':tests}
        row=base_row('apps',r.get('id',i),r['question'],spec,solutions[:3],family=r['url'] or None)
        row['starter_code']=r['starter_code'] or '';yield row
    for r in pq.read_table(sources/'tulu.parquet').to_pylist():
        if not r['constraints'] or any(c not in IF_TYPES for c in r['constraints']):continue
        checks=[IF_TYPES[c] for c in r['constraints']]
        reference=r['messages'][-1]['content']
        if not instruction_check(reference,checks):continue
        yield base_row('tulu',r['id'],r['prompt'],{'task':'instructions','checks':checks},reference)
    raw=json.loads((sources/'glaive.json').read_text())
    if isinstance(raw,dict):raw=raw.get('train',raw.get('data'))
    for i,r in enumerate(raw):
        try:
            system=r['system'];start=system.index('{');tail=system[start:];tools=[];dec=json.JSONDecoder()
            while tail.strip():
                v,n=dec.raw_decode(tail.lstrip());tools.append(v);tail=tail.lstrip()[n:].strip()
            if not 1<=len(tools)<=4:continue
            question,rest=r['chat'].split('USER:',1)[1].split('ASSISTANT:',1)
            first=rest.split('<|endoftext|>',1)[0].strip()
            if '<functioncall>' in first:
                call=parsed(first.split('<functioncall>',1)[1].strip())
                if isinstance(call.get('arguments'),str):call['arguments']=parsed(call['arguments'])
                if not isinstance(call.get('arguments'),dict):continue
                calls=[{'name':call['name'],'arguments':call['arguments']}]
                tool=next(t for t in tools if t['name']==call['name'])
                props=tool['parameters'].get('properties',{});req=tool['parameters'].get('required',[])
                if not set(req)<=call['arguments'].keys() or not call['arguments'].keys()<=props.keys():continue
            elif re.search(r"(?:don't|do not|cannot|can't|unable|not have|not able)",first,re.I) and '?' not in first:
                calls=[]
            else:continue
            family=hashlib.sha256(canonical(sorted(tools,key=lambda t:t['name'])).encode()).hexdigest()
            row=base_row('glaive',i,question,{'task':'tools','calls':calls},canonical(calls),family)
            row['tools']=tools;row['tool_case']='call' if calls else 'no_call';yield row
        except (ValueError,KeyError,TypeError,SyntaxError,StopIteration):continue

def check_candidate(row):
    if row['task']=='coding':
        for code in row['reference']:
            if code_check(code,json.loads(row['answer'])['tests']):
                row['reference']=code;return row
        return None
    return row if verify(row['reference'],row['answer']) else None

def prompt_for(row):
    task=row['task'];text=row['problem']
    if task=='math':text+='\n\nPut your final numerical answer within \\boxed{}.'
    elif task=='coding':
        spec=json.loads(row['answer'])['tests']
        text+='\n\nWrite Python 3 code. Return only the complete code in a Python code block after your reasoning.'
        if spec.get('fn_name'):text+=' Implement the function named '+spec['fn_name']+'.'
        else:text+=' Read input from standard input and write the answer to standard output.'
        if row.get('starter_code'):text+='\nStarter code:\n'+row['starter_code']
    elif task=='tools':text+='\n\nAvailable functions:\n'+canonical(row['tools'])+'\n\nAfter reasoning, output only a JSON array of calls, each with "name" and "arguments". If none of the available functions can fulfill the request, output []. Do not execute any functions.'
    return [{'role':'user','content':text}]

def prepare(sources,out,benchmark):
    from transformers import AutoTokenizer
    if out.exists() and any(out.iterdir()):raise ValueError('Refusing to overwrite data')
    out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((sources/'manifest.json').read_text())
    for f,m in manifest.items():
        if sha(sources/f)!=m['sha256']:raise ValueError('Source changed '+f)
    candidates=list(raw_candidates(sources));random.Random(SEED).shuffle(candidates)
    print('Raw candidates',dict(collections.Counter(r['task'] for r in candidates)),flush=True)
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_REVISION,local_files_only=True)
    bench=[{'id':r['id'],'text':prompt_text(r)} for r in map(json.loads,benchmark.read_text().splitlines())]
    index=DuplicateIndex(.8)
    for r in bench:index.add('benchmark:'+r['id'],r['text'])
    pools={d:{'train':[],'development':[],'confirmation':[]} for d in DOMAINS};reject=[]
    # Reference checks happen only on CPU; no LLM API/GPU or benchmark answers used.
    selected=[]
    for row in candidates:
        domain=row['task'];bucket=int(hashlib.sha256(row['family'].encode()).hexdigest()[:8],16)%10
        split='development' if bucket==0 else 'confirmation' if bucket==1 else 'train'
        if domain=='tools':
            same=sum(r['tool_case']==row['tool_case'] for r in pools[domain][split])
            limit=(1400 if split=='train' else 100) if row['tool_case']=='call' else (100 if split=='train' else 25)
            if same>=limit:continue
        elif len(pools[domain][split])>=(1400 if split=='train' else 100):continue
        hit=index.match(row['problem'])
        if hit:
            reject.append({'id':row['id'],'reason':hit[1],'matched_id':hit[0]});continue
        row['prompt']=prompt_for(row)
        tokens=tokenizer.apply_chat_template(row['prompt'],tokenize=True,add_generation_prompt=True,enable_thinking=True)
        if len(tokens)>1500:
            reject.append({'id':row['id'],'reason':'prompt_over_1500'});continue
        row['prompt_tokens']=len(tokens);row['split']=split
        row['source_revision']=manifest[{'gsm8k':'gsm8k.parquet','apps':'apps.jsonl','tulu':'tulu.parquet','glaive':'glaive.json'}[row['source']]]['revision']
        index.add('candidate:'+row['id'],row['problem']);pools[domain][split].append(row);selected.append(row)
    verified=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i,row in enumerate(pool.map(check_candidate,selected,chunksize=4)):
            if row:verified.append(row)
            if i%500==0:print('Checked',i,'accepted',len(verified),flush=True)
    pools={d:{s:[] for s in ('train','development','confirmation')} for d in DOMAINS}
    for row in verified:pools[row['task']][row['split']].append(row)
    output={s:[] for s in ('train','development','confirmation')}
    for d in DOMAINS:
        for split,n in [('train',1000),('development',64),('confirmation',64)]:
            rows=pools[d][split]
            if len(rows)<n:raise ValueError(f'Insufficient verified {d}/{split}: {len(rows)} < {n}')
            if d=='tools':
                # Include source-provided irrelevance, not model-generated negatives.
                negatives=[r for r in rows if r['tool_case']=='no_call'];positives=[r for r in rows if r['tool_case']=='call']
                k=min(n//4,len(negatives));rows=negatives[:k]+positives[:n-k]
                if len(rows)!=n:raise ValueError('Insufficient tool cases')
            else:rows=rows[:n]
            output[split].extend(rows)
    # Multiple-choice tool selection: add only a different schema from the SAME split;
    # candidate gold stays unchanged; no training API schema enters validation.
    for split,rows in output.items():
        toolrows=[r for r in rows if r['task']=='tools']
        for i,row in enumerate(toolrows):
            if i%2==0 and row['tool_case']=='call':
                extra=next((r['tools'][0] for r in toolrows if r['tools'][0]['name'] not in {t['name'] for t in row['tools']}),None)
                if extra:row['tools']=row['tools']+[extra];row['tool_case']='selection';row['prompt']=prompt_for(row)
        for row in rows:
            row.pop('reference');row.pop('starter_code',None)
            row['prompt_tokens']=len(tokenizer.apply_chat_template(row['prompt'],tokenize=True,add_generation_prompt=True,enable_thinking=True))
            if row['prompt_tokens']>1500:raise ValueError('Distractor exceeds prompt cap')
        random.Random(SEED).shuffle(rows)
    monitor=[]
    for d in DOMAINS:
        rows=[r for r in output['development'] if r['task']==d]
        if d=='tools':
            chosen=[next(r for r in rows if r['tool_case']==case) for case in ('no_call','call','selection')]
            chosen+=next([r] for r in rows if r['id'] not in {x['id'] for x in chosen})
        elif d=='instructions':
            chosen=[];seen=set()
            for r in rows:
                checks=tuple(json.loads(r['answer'])['checks'])
                if checks not in seen:chosen.append(r);seen.add(checks)
                if len(chosen)==4:break
        else:chosen=rows[:4]
        monitor.extend(chosen)
    files={s+'.jsonl':r for s,r in output.items()};files.update({'monitor.jsonl':monitor,'quarantine.jsonl':reject,'benchmark-prompts.jsonl':bench})
    for f,rows in files.items():write_rows(out/f,rows)
    result={'schema_version':1,'id':'shorterbetter-four-domains-v1','seed':SEED,'sources':manifest,
      'counts':{f:len(r) for f,r in files.items()},'domain_counts':{s:dict(collections.Counter(r['task'] for r in rows)) for s,rows in output.items()},
      'tool_cases':{s:dict(collections.Counter(r.get('tool_case') for r in rows if r['task']=='tools')) for s,rows in output.items()},
      'instruction_checks':sorted(IF_TYPES.values()),'tokenizer_model':MODEL,'tokenizer_revision':MODEL_REVISION,
      'instruction_checks_observed':dict(collections.Counter(c for r in output['train'] if r['task']=='instructions' for c in json.loads(r['answer'])['checks'])),
      'benchmark_file_sha256':sha(benchmark),'files':{f:sha(out/f) for f in files},
      'screening':{'method':'normalized exact and 5-token-shingle containment','threshold':.8,'zero_semantic_overlap_guarantee':False},
      'reference_checks':'Every selected reference passed its CPU verifier; coding uses all retained source tests in chroot/seccomp sandbox.',
      'limitations':['Glaive substitutes gated xLAM for this pilot.','IF verifies seven constraint families, not semantic topical correctness or all25 IFEval types.','Tulu prompt/template lineage is upstream; source was already synthetic, no new generation performed.','Source tests may be incomplete; screening is heuristic.'],
      'preparation_code_sha256':sha(__file__),
      'verifier_sha256':{f:sha(Path(__file__).with_name(f)) for f in ('rewards.py','code_worker.py')}}
    (out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    audit(out)
    return result

def audit(out):
    m=json.loads((out/'manifest.json').read_text());index=DuplicateIndex(.8);families={};ids=set();counts={}
    for f,h in m['files'].items():
        if sha(out/f)!=h:raise ValueError('Hash mismatch '+f)
    for r in map(json.loads,(out/'benchmark-prompts.jsonl').read_text().splitlines()):index.add('benchmark:'+r['id'],r['text'])
    for split in ['train','development','confirmation']:
        rows=[json.loads(l) for l in (out/(split+'.jsonl')).read_text().splitlines()]
        for r in rows:
            if r['id'] in ids or r['split']!=split or index.match(r['problem']):raise ValueError('Residual overlap or split violation '+r['id'])
            if r['family'] in families and families[r['family']]!=split:raise ValueError('Family split violation')
            families[r['family']]=split;ids.add(r['id']);index.add(r['id'],r['problem'])
            if 'reference' in r or r['prompt_tokens']>1500 or json.loads(r['answer'])['task']!=r['task']:raise ValueError('Schema violation')
        counts[split]=dict(collections.Counter(r['task'] for r in rows))
    dev={r['id']:r for r in map(json.loads,(out/'development.jsonl').read_text().splitlines())}
    monitors=[json.loads(l) for l in (out/'monitor.jsonl').read_text().splitlines()]
    if len(monitors)!=16 or any(r!=dev.get(r['id']) for r in monitors):raise ValueError('Monitor split violation')
    result={'status':'passed','manifest_sha256':sha(out/'manifest.json'),'domain_counts':counts,'residual_exact_lexical_overlap':0,'semantic_overlap_guarantee':False}
    (out/'audit.json').write_text(json.dumps(result,indent=2)+'\n');return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sources',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--benchmark',type=Path,required=True);a=p.parse_args()
    print(json.dumps(prepare(a.sources,a.out,a.benchmark),indent=2))
