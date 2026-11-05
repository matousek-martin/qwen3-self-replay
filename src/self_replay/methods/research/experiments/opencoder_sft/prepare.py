"""Pin, screen and sandbox-check OpenCoder solutions without model generation.

Run with the project's CPU data environment. Benchmark text is rejection-only.
Never writes to the historical replay campaigns or launches a GPU.
"""
import argparse
import ast
import collections
import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from experiments.data_pipeline.pipeline import DuplicateIndex, canonical, normalize
from experiments.shorterbetter.multidomain.rewards import code_check

ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT/'lab/experiments/cloud/replay-scaled-training-20261101t122440z/output/training/checkpoints/group-000768'
REVISION = '7d28f40d579edd7c24402d17d0c7639f991e6f8d'
SOURCE_SHA = '59cc262e240140ca265655a8d14d2a0f28139164a8e532294308692408095759'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')


def rows_write(path, rows):
    Path(path).write_text(''.join(canonical(r)+'\n' for r in rows))


def literals(test, entry):
    """Conservative executable contract: one direct call == literal expected value.

    No eval, test execution, transformations of official benchmark content, or
    silently dropped source tests. Unsupported tests quarantine the whole row.
    """
    tree = ast.parse(test)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assert):
        raise ValueError('unsupported_test')
    expr = tree.body[0].test
    if not isinstance(expr, ast.Compare) or len(expr.ops) != 1 or not isinstance(expr.ops[0], ast.Eq):
        raise ValueError('unsupported_test')
    call = expr.left
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) or call.func.id != entry or call.keywords:
        raise ValueError('unsupported_test')
    args = [ast.literal_eval(a) for a in call.args]
    expected = ast.literal_eval(expr.comparators[0])
    # The existing isolated harness uses JSON transport; preserve types exactly.
    def safe(v):
        if v is None or type(v) in (str, int, float, bool): return True
        if isinstance(v, list): return all(safe(x) for x in v)
        if isinstance(v, dict): return all(type(k) is str and safe(x) for k,x in v.items())
        return False
    if not safe(args) or not safe(expected): raise ValueError('non_json_literal')
    json.dumps([args,expected], allow_nan=False)
    return args, expected


def code_key(code):
    """Structural duplicate heuristic: erase identifier names, retain constants."""
    tree = ast.parse(code)
    class Names(ast.NodeTransformer):
        def visit_Name(self, node): node.id='IDENT'; return node
        def visit_arg(self, node): node.arg='ARG'; return node
        def visit_FunctionDef(self, node):
            self.generic_visit(node); node.name='FUNCTION'
            node.body=[n for n in node.body if not (isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str))]
            return node
    return hashlib.sha256(ast.dump(Names().visit(tree),include_attributes=False).encode()).hexdigest()


def category(code):
    """Transparent syntax-based tags, not ground-truth difficulty or failure labels."""
    t=ast.parse(code); names={n.id for n in ast.walk(t) if isinstance(n,ast.Name)}
    attrs={n.attr for n in ast.walk(t) if isinstance(n,ast.Attribute)}
    if 're' in names or attrs & {'split','join','strip','lower','upper','replace','isalpha'}: return 'strings'
    if any(isinstance(n,(ast.Dict,ast.DictComp,ast.Set,ast.SetComp)) for n in ast.walk(t)) or names & {'dict','set','Counter'}: return 'maps_sets'
    if any(isinstance(n,(ast.List,ast.ListComp,ast.Subscript)) for n in ast.walk(t)) or names & {'list','sorted'}: return 'sequences'
    if any(isinstance(n,(ast.For,ast.While,ast.If)) for n in ast.walk(t)): return 'control_algorithms'
    return 'numeric_other'


def candidate(raw):
    entry=raw['entry_point']
    if not re.fullmatch(r'[A-Za-z_]\w*',entry): raise ValueError('entry_point')
    t=ast.parse(raw['code'])
    if not any(isinstance(n,ast.FunctionDef) and n.name==entry for n in t.body): raise ValueError('missing_function')
    if len(raw['testcase']) < 3: raise ValueError('too_few_tests')
    pairs=[literals(s,entry) for s in raw['testcase']]
    if len({canonical(x[0]) for x in pairs})<3: raise ValueError('too_few_distinct_inputs')
    return {'id':'opencoder-'+str(raw['seq_id']), 'source_index':raw['seq_id'],
        'problem':raw['instruction'], 'code':raw['code'], 'entry_point':entry,
        'tests':{'fn_name':entry,'inputs':[x[0] for x in pairs],'outputs':[x[1] for x in pairs]},
        'category':category(raw['code']), 'code_key':code_key(raw['code'])}


def balanced(rows, n):
    pools=collections.defaultdict(list)
    for r in sorted(rows,key=lambda x:hashlib.sha256(('17:'+x['id']).encode()).hexdigest()): pools[r['category']].append(r)
    out=[]
    while len(out)<n and any(pools.values()):
        for k in sorted(pools):
            if pools[k] and len(out)<n: out.append(pools[k].pop())
    return out


def checked(row):
    ok=code_check(row['code'],row['tests'])
    if ok and code_check('def '+row['entry_point']+'(*args):\n    return None',row['tests']):
        return row, 'non_discriminating_tests'
    return row, 'passed' if ok else 'source_solution_failed'


def build(reg, reservoir=10000, workers=8):
    import pyarrow.parquet as pq
    source=reg/'source/train-00000-of-00001.parquet'
    meta=json.loads((reg/'source/metadata.json').read_text())
    if meta['revision']!=REVISION or sha(source)!=SOURCE_SHA: raise ValueError('Unpinned source')
    out=reg/'data'
    if out.exists(): raise ValueError('Refusing to overwrite an existing prepared dataset')
    out.mkdir()
    if not code_check('def f(x): return x+1',{'fn_name':'f','inputs':[[1]],'outputs':[2]}): raise RuntimeError('Sandbox preflight failed')
    if code_check('def f(x): return x',{'fn_name':'f','inputs':[[1]],'outputs':[2]}): raise RuntimeError('Sandbox accepted bad solution')
    bench=ROOT/'lab/experiments/shorterbetter-multidomain/data-v5-full/benchmark-prompts.jsonl'
    exclusion=DuplicateIndex(.7)
    grams={}
    entry_names=set()
    sources={str(bench.relative_to(ROOT)):sha(bench)}
    for r in map(json.loads,bench.open()):
        exclusion.add('official:'+r['id'],r['text'])
        w=normalize(r['text']).split()
        for i in range(len(w)-9): grams.setdefault(tuple(w[i:i+10]),r['id'])
        entry_names.update(re.findall(r'\bdef\s+([A-Za-z_]\w*)\s*\(',r['text']))
    # Historical development/confirmation questions stay excluded as well.
    for split in ('development','confirmation'):
        f=ROOT/f'lab/experiments/shorterbetter-multidomain/data-v5-full/{split}.jsonl'
        sources[str(f.relative_to(ROOT))]=sha(f)
        for r in map(json.loads,f.open()): exclusion.add(split+':'+r['id'],r['problem'])
    raw=pq.read_table(source).to_pylist()
    raw.sort(key=lambda r:hashlib.sha256(('17:'+str(r['seq_id'])).encode()).hexdigest())
    # Connected components of source-level prompt/entry-point/code families.
    parents=list(range(len(raw))); seen={}
    def find(i):
        while parents[i]!=i: parents[i]=parents[parents[i]]; i=parents[i]
        return i
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b: parents[max(a,b)]=min(a,b)
    candidates=[]; rejected=[]; family_blocked=set(); seen_ids=set()
    for i,r in enumerate(raw):
        try: ck=code_key(r['code'])
        except (SyntaxError,ValueError,TypeError): ck=None
        keys=[('prompt',normalize(r['instruction'])),('entry',r['entry_point'])]
        if ck: keys.append(('code',ck))
        for key in keys:
            if key in seen: union(i,seen[key])
            else: seen[key]=i
        # Even an unusable source row can reveal a contaminated linked family.
        hit=exclusion.match(r['instruction'])
        words=normalize(r['instruction']).split()
        ten=any(tuple(words[j:j+10]) in grams for j in range(len(words)-9))
        contaminated=bool(hit or ten or r['entry_point'] in entry_names)
        if contaminated: family_blocked.add(i)
        try:
            c=candidate(r);c['_i']=i
            if c['id'] in seen_ids: raise ValueError('duplicate_source_id')
            seen_ids.add(c['id'])
            if contaminated: raise ValueError('heldout_overlap')
            if len(c['problem'])>6000 or len(c['code'])>14000: raise ValueError('long_source')
            candidates.append(c)
        except (ValueError,SyntaxError,TypeError,KeyError) as e:
            rejected.append({'id':'opencoder-'+str(r['seq_id']),'reason':str(e).split('\n')[0][:100]})
        if i%20000==0: print('screened',i,'candidates',len(candidates),flush=True)
    blocked_roots={find(i) for i in family_blocked}
    unique=DuplicateIndex(.8); kept=[]; groups=collections.Counter()
    for c in candidates:
        root=find(c.pop('_i')); c['family']='source-component-'+str(raw[root]['seq_id'])
        if root in blocked_roots:
            rejected.append({'id':c['id'],'reason':'heldout_family'});continue
        if unique.match(c['problem']):
            rejected.append({'id':c['id'],'reason':'duplicate_prompt'});continue
        # Limit repetitive variants before CPU verification, without using outcomes.
        if groups[root]>=2:
            rejected.append({'id':c['id'],'reason':'family_cap'});continue
        unique.add(c['id'],c['problem']);groups[root]+=1
        bucket=int(hashlib.sha256(c['family'].encode()).hexdigest()[:8],16)%10
        c['split']='development' if bucket==0 else 'confirmation' if bucket==1 else 'train'
        kept.append(c)
    print('deduplicated',len(kept),'splits',dict(collections.Counter(c['split'] for c in kept)),flush=True)
    # Verify a bounded, stratified reservoir; preserve all filtered candidates for audit.
    to_check=[]
    for split,limit in [('train',reservoir+2000),('development',800),('confirmation',800)]:
        to_check.extend(balanced([c for c in kept if c['split']==split],limit))
    accepted=[]; started=time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i,(r,result) in enumerate(ex.map(checked,to_check)):
            if result=='passed': accepted.append(r)
            else: rejected.append({'id':r['id'],'reason':result})
            if i%500==0: print('verified',i,'passed',len(accepted),'seconds',round(time.time()-started),flush=True)
    chosen={s:balanced([r for r in accepted if r['split']==s],n) for s,n in [('train',reservoir),('development',256),('confirmation',256)]}
    if len(chosen['train'])<2000 or min(len(chosen[s]) for s in ['development','confirmation'])<128:
        rows_write(out/'quarantine.jsonl',rejected)
        raise RuntimeError('Insufficient clean data; do not launch')
    # Preserve source code and tests in grading rows; tests never enter SFT prompts.
    for split,rs in chosen.items(): rows_write(out/(split+'-verified.jsonl'),rs)
    def sft(r):
        return {'id':r['id'],'family':r['family'],'category':r['category'],'split':r['split'],
            'messages':[{'role':'user','content':r['problem']+'\n\nImplement the Python function '+r['entry_point']+'.'},
                        {'role':'assistant','content':'<think>\n\n</think>\n\n```python\n'+r['code'].strip()+'\n```'}],
            'source':'OpenCoder-LLM/opc-sft-stage2/educational_instruct','source_revision':REVISION}
    rows_write(out/'train-reservoir.jsonl',[sft(r) for r in chosen['train']])
    rows_write(out/'train-2000.jsonl',[sft(r) for r in chosen['train'][:2000]])
    rows_write(out/'quarantine.jsonl',rejected)
    rows_write(out/'verification-candidates.jsonl',to_check)
    stats={'raw_rows':len(raw),'distinct_source_instructions':len({r['instruction'] for r in raw}),
        'structural_families':len({find(i) for i in range(len(raw))}), 'filtered_candidates':len(kept),
        'tested':len(to_check),'passed':len(accepted),'selected':{s:len(rs) for s,rs in chosen.items()},
        'category_counts':{s:dict(collections.Counter(r['category'] for r in rs)) for s,rs in chosen.items()},
        'rejection_counts':dict(collections.Counter(r['reason'] for r in rejected)),
        'test_count_histogram':dict(collections.Counter(len(r['tests']['inputs']) for r in chosen['train'])),
        'screening':'normalized exact, 5-token containment>=.7 vs holdouts, any10gram official overlap, official function-name overlap; whole linked family quarantine; internal .8 dedup',
        'family_method':'connected components of normalized prompt, exact entry point, identifier-normalized code AST; max2variants/family',
        'sandbox':'existing chroot+unprivileged UID+seccomp+resource limits; source tests parsed as literal I/O, not executed',
        'limitations':['Lexical and structural heuristics do not guarantee zero semantic overlap.',
            'Provided tests may be incomplete or wrong; no independent oracle.',
            'Restricted literal tests exclude some otherwise valid tasks; distribution is filtered.',
            'Categories are syntax heuristics, not difficulty labels.',
            'No GPU throughput, model improvement or training compatibility measured.'],
        'exclusion_sources':sources,'source_sha256':sha(source),'source_revision':REVISION,
        'files_sha256':{f.name:sha(f) for f in out.glob('*.jsonl')}}
    write(out/'manifest.json',stats); print(json.dumps(stats['selected']),flush=True)
    return stats


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--registration',type=Path,required=True);ap.add_argument('--reservoir',type=int,default=10000);ap.add_argument('--workers',type=int,default=8)
    a=ap.parse_args();build(a.registration,a.reservoir,a.workers)
