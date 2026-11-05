"""CPU-only xLAM replacement; immutable source banks and a balanced4k training mix.

Screening rejects candidates using evaluation prompts; never generates from them.
API/namespace families are assigned before selection. Cross-partition API bundles
are quarantined intact, rather than dropping functions or changing gold calls.
"""
import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re

from experiments.data_pipeline.pipeline import DuplicateIndex, canonical
from experiments.shorterbetter.prepare import MODEL, MODEL_REVISION, sha, write_rows
from .prepare import DOMAINS, SEED, base_row, prompt_for, raw_candidates
from .rewards import verify

SPLITS = ('train', 'development', 'confirmation')
DATA_ID = 'shorterbetter-four-domains-xlam-v1'
NO_CALL_COUNTS = {'train':79, 'development':2, 'confirmation':9}


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def unpack(value):
    return json.loads(value) if isinstance(value, str) else value


def api_family(name):
    if not isinstance(name, str) or not name.strip():
        raise ValueError('invalid_api_name')
    return (name.rsplit('.', 1)[0] if '.' in name else name).casefold()


def family_registry(parent_rows):
    registry = defaultdict(set)
    for split, records in parent_rows.items():
        for row in records:
            for tool in row.get('tools', []):
                registry[api_family(tool['name'])].add(split)
    return dict(registry)


def family_split(families, registry):
    assigned = set()
    for family in families:
        if family in registry:
            if len(registry[family]) != 1:
                raise ValueError('historical_api_split_conflict')
            assigned.update(registry[family])
        else:
            bucket = int(hashlib.sha256(('xlam-api:' + family).encode()).hexdigest()[:8], 16) % 10
            assigned.add('development' if bucket == 0 else 'confirmation' if bucket == 1 else 'train')
    if len(assigned) != 1:
        raise ValueError('cross_partition_api_bundle')
    return assigned.pop()


def type_tree(description):
    if not isinstance(description, str) or len(description) > 200:
        raise ValueError('unsupported_parameter_type')
    text = re.split(r',\s*(?:optional|default)', description, maxsplit=1)[0].strip()
    try:
        tree = ast.parse(text, mode='eval').body
    except SyntaxError:
        raise ValueError('unsupported_parameter_type') from None
    def valid(node):
        if isinstance(node, ast.Name):
            return node.id in ('str','int','float','bool','list','List','dict','Dict','tuple','Tuple','set','Set')
        if not isinstance(node, ast.Subscript) or not isinstance(node.value, ast.Name):
            return False
        name = node.value.id
        children = list(node.slice.elts) if isinstance(node.slice, ast.Tuple) else [node.slice]
        return (name in ('List','list','Tuple','tuple','Dict','dict','Union','Optional','Set','set')
                and 0 < len(children) <= 8 and all(valid(child) for child in children))
    if len(list(ast.walk(tree))) > 40 or not valid(tree):
        raise ValueError('unsupported_parameter_type')
    return tree


def type_matches(value, tree):
    if isinstance(tree, ast.Name):
        name = tree.id
        return {'str': isinstance(value, str), 'int': type(value) is int,
                'float': type(value) in (int, float), 'bool': type(value) is bool,
                'list': isinstance(value, list), 'List': isinstance(value, list),
                'tuple': isinstance(value, list), 'Tuple': isinstance(value, list),
                'set': isinstance(value, list), 'Set': isinstance(value, list),
                'dict': isinstance(value, dict), 'Dict': isinstance(value, dict)}[name]
    name = tree.value.id
    children = list(tree.slice.elts) if isinstance(tree.slice, ast.Tuple) else [tree.slice]
    if name == 'Union':
        return any(type_matches(value, c) for c in children)
    if name == 'Optional':
        return len(children) == 1 and (value is None or type_matches(value, children[0]))
    if name in ('List','list','Set','set'):
        return len(children) == 1 and isinstance(value, list) and all(type_matches(v, children[0]) for v in value)
    if name in ('Tuple','tuple'):
        return isinstance(value, list) and len(value) == len(children) and all(type_matches(v,c) for v,c in zip(value, children))
    if name in ('Dict','dict'):
        return len(children) == 2 and isinstance(value, dict) and all(type_matches(k,children[0]) and type_matches(v,children[1]) for k,v in value.items())
    return False


def xlam_row(raw):
    tools, calls = unpack(raw['tools']), unpack(raw['answers'])
    if not isinstance(tools, list) or not tools or not isinstance(calls, list) or not calls:
        raise ValueError('empty_or_invalid_tools_calls')
    if not isinstance(raw['query'], str) or not raw['query'].strip():
        raise ValueError('invalid_query')
    by_name, schemas = {}, {}
    for tool in tools:
        if set(tool) != {'name','description','parameters'} or tool['name'] in by_name:
            raise ValueError('invalid_or_duplicate_tool')
        api_family(tool['name'])
        params = tool['parameters']
        if not isinstance(params, dict):
            raise ValueError('invalid_parameters')
        schemas[tool['name']] = {name:type_tree(value['type']) for name,value in params.items()}
        by_name[tool['name']] = tool
    for call in calls:
        if (not isinstance(call, dict) or set(call) != {'name','arguments'}
                or call['name'] not in by_name or not isinstance(call['arguments'], dict)):
            raise ValueError('invalid_or_unknown_call')
        params = by_name[call['name']]['parameters']; args = call['arguments']
        required = {name for name,p in params.items() if p.get('required') is True or
                    (p.get('required') is not False and 'default' not in p and 'optional' not in p['type'].lower())}
        if not required <= args.keys() or not args.keys() <= params.keys():
            raise ValueError('missing_or_unknown_argument')
        if any(not type_matches(value, schemas[call['name']][name]) for name,value in args.items()):
            raise ValueError('argument_type_mismatch')
    json.dumps([tools,calls], allow_nan=False)
    row = base_row('xlam', raw['id'], raw['query'], {'task':'tools','calls':calls}, canonical(calls))
    row.update(tools=tools, tool_case='single_call' if len(calls)==1 else 'multiple_calls')
    return row


def hash_check(path, files):
    for name, value in files.items():
        expected = value['sha256'] if isinstance(value, dict) else value
        if sha(path/name) != expected:
            raise ValueError('Source hash mismatch: '+name)


def prepare(parent, sources, xlam, out):
    from transformers import AutoTokenizer
    if out.exists() and any(out.iterdir()):
        raise ValueError('Refusing to overwrite data')
    parent_manifest=json.loads((parent/'manifest.json').read_text())
    parent_audit=json.loads((parent/'audit.json').read_text())
    if parent_audit['status']!='passed' or parent_audit['manifest_sha256']!=sha(parent/'manifest.json'):
        raise ValueError('Parent audit missing/stale')
    hash_check(parent,parent_manifest['files'])
    hash_check(sources,json.loads((sources/'manifest.json').read_text()))
    xm=json.loads((xlam/'manifest.json').read_text());hash_check(xlam,xm['files'])
    for f,h in parent_manifest['verifier_sha256'].items():
        if sha(Path(__file__).with_name(f))!=h:raise ValueError('Parent verifier changed')
    old={s:rows(parent/(s+'.jsonl')) for s in SPLITS};registry=family_registry(old)
    retained={s:[r for r in old[s] if r['task']!='tools'] for s in SPLITS}
    benchmark=rows(parent/'benchmark-prompts.jsonl')
    historical=[{'id':r['id'],'split':s,'text':r['problem']} for s in SPLITS for r in old[s] if r['task']=='tools']
    index=DuplicateIndex(.8)
    for r in benchmark:index.add('benchmark:'+r['id'],r['text'])
    for s in SPLITS:
        for r in old[s]:index.add('historical:'+r['id'],r['problem'])
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_REVISION,local_files_only=True)
    banks={source:{s:[] for s in SPLITS} for source in ('xlam','glaive')}
    reject=[];reasons=Counter()
    def rejected(source, ident, reason, **extra):
        reject.append({'source':source,'source_index':ident,'reason':reason,**extra});reasons[reason]+=1
    def admit(row, revision):
        families=sorted({api_family(t['name']) for t in row['tools']})
        try:split=family_split(families,registry)
        except ValueError as exc:
            rejected(row['source'],row['source_index'],str(exc));return
        hit=index.match(row['problem'])
        if hit:
            rejected(row['source'],row['source_index'],hit[1],matched_id=hit[0],similarity=hit[2]);return
        row.update(api_families=families, family=hashlib.sha256(canonical(families).encode()).hexdigest(),
                   split=split,source_revision=revision)
        row['prompt']=prompt_for(row)
        row['prompt_tokens']=len(tokenizer.apply_chat_template(row['prompt'],tokenize=True,add_generation_prompt=True,enable_thinking=True))
        if row['prompt_tokens']>1500:
            rejected(row['source'],row['source_index'],'prompt_over_1500');return
        if not verify(row['reference'],row['answer']):raise ValueError('Reference fails tool verifier')
        row.pop('reference');index.add(row['id'],row['problem']);banks[row['source']][split].append(row)
    raw=json.loads((xlam/'xlam_function_calling_60k.json').read_text())
    if len(raw)!=60000 or len({r['id'] for r in raw})!=len(raw):raise ValueError('Unexpected source IDs/count')
    random.Random(SEED).shuffle(raw)
    for i,r in enumerate(raw):
        try:row=xlam_row(r)
        except (ValueError,KeyError,TypeError) as exc:
            rejected('xlam',r.get('id'),str(exc) if isinstance(exc,ValueError) else 'invalid_schema');continue
        admit(row,xm['revision'])
        if i%3000==0:print('xLAM checked',i,'accepted',sum(map(len,banks['xlam'].values())),flush=True)
    # Preserve already-screened no-call examples in their historical partitions.
    # Fresh Glaive negatives repeat historical questions; never reassign them.
    for split in SPLITS:
        banks['glaive'][split] = [r for r in old[split] if r['task']=='tools' and r['tool_case']=='no_call']
        if len(banks['glaive'][split]) != NO_CALL_COUNTS[split]:
            raise ValueError('Unexpected inherited no-call count')
    print('Source banks', {k:{s:len(v) for s,v in d.items()} for k,d in banks.items()},flush=True)
    selected={s:list(retained[s]) for s in SPLITS}
    for split in SPLITS:
        n=1000 if split=='train' else 64
        negative_count=NO_CALL_COUNTS[split]
        counts={'single_call':(n-negative_count)//2,'multiple_calls':n-negative_count-(n-negative_count)//2}
        for case,count in counts.items():
            candidates=[r for r in banks['xlam'][split] if r['tool_case']==case]
            if len(candidates)<count:raise ValueError(f'Insufficient {split}/{case}: {len(candidates)}')
            selected[split].extend(candidates[:count])
        if len(banks['glaive'][split])<negative_count:raise ValueError(f'Insufficient no-call {split}')
        selected[split].extend(banks['glaive'][split][:negative_count])
        random.Random(SEED).shuffle(selected[split])
    monitor=[r for r in rows(parent/'monitor.jsonl') if r['task']!='tools']
    devtools=[r for r in selected['development'] if r['task']=='tools']
    chosen=[next(r for r in devtools if r['tool_case']==c) for c in ('single_call','multiple_calls','no_call')]
    chosen.append(next(r for r in devtools if r['id'] not in {v['id'] for v in chosen}))
    monitor.extend(chosen)
    files={s+'.jsonl':v for s,v in selected.items()}
    files.update({f'{source}-{s}.jsonl':rs for source,bank in banks.items() for s,rs in bank.items()})
    files.update({'monitor.jsonl':monitor,'benchmark-prompts.jsonl':benchmark,
                  'historical-tool-prompts.jsonl':historical,'quarantine.jsonl':reject})
    out.mkdir(parents=True,exist_ok=True)
    for f,rs in files.items():write_rows(out/f,rs)
    manifest={'schema_version':2,'id':DATA_ID,'seed':SEED,'parent_directory':str(parent.resolve()),
        'parent_manifest_sha256':sha(parent/'manifest.json'),'parent_audit_sha256':sha(parent/'audit.json'),
        'sources':{**parent_manifest['sources'],'xlam':xm},'xlam_source_directory':str(xlam.resolve()),
        'legacy_source_directory':str(sources.resolve()),'counts':{f:len(rs) for f,rs in files.items()},
        'domain_counts':{s:dict(Counter(r['task'] for r in rs)) for s,rs in selected.items()},
        'tool_cases':{s:dict(Counter(r['tool_case'] for r in rs if r['task']=='tools')) for s,rs in selected.items()},
        'bank_counts':{source:{s:len(rs) for s,rs in bank.items()} for source,bank in banks.items()},
        'raw_xlam_rows':len(raw),'inherited_no_call_counts':NO_CALL_COUNTS,'rejections':dict(reasons),
        'files':{f:sha(out/f) for f in files},'verifier_sha256':parent_manifest['verifier_sha256'],
        'preparation_code_sha256':sha(__file__),'shared_preparation_code_sha256':sha(Path(__file__).with_name('prepare.py')),
        'tokenizer_model':MODEL,'tokenizer_revision':MODEL_REVISION,'benchmark_file_sha256':parent_manifest['benchmark_file_sha256'],
        'screening':{'method':'normalized exact and5token-shingle containment','threshold':.8,'zero_semantic_overlap_guarantee':False,
                     'reject_against_all_historical_questions':True,'new_xlam_api_family_disjoint':True,'inherited_glaive_exact_schema_families_disjoint':True,
                     'inherited_glaive_api_names_may_overlap':True,'cross_partition_xlam_api_bundles_quarantined':True},
        'reference_checks':'xLAM call names/required arguments/types and verifier self-check; source APIs NOT executed locally. Non-tool rows inherit unchanged verified parent references.',
        'limitations':['Lexical screening is heuristic; zero semantic overlap is not guaranteed.',
          'API/namespace grouping cannot reconstruct all upstream synthetic template families.',
          'Glaive no-call examples inherit exact parent rows/splits and schema-family separation; seven API names overlap between these historical splits. New xLAM APIs with historical split conflicts are quarantined.',
          'Glaive no-call labels use upstream refusals; local semantic correctness is not independently established.',
          'Development has only two inherited no-call examples; report this subgroup separately and do not claim reliable no-call generalization.',
          'xLAM typing subset is conservative; unsupported schemas and mismatches quarantined, never repaired using benchmark answers.',
          'Four-domain mix remains1000training questions/domain; source banks are larger but do not authorize a larger run.',
          'Instruction domain retains six represented constraint types and no semantic content judge.',
          'Historical development questions were already inspected; confirmation is separately held out.']}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    audit(out)
    return manifest


def audit(out):
    """Fresh disk read verifies provenance, references, separation and selected subsets."""
    from transformers import AutoTokenizer
    m=json.loads((out/'manifest.json').read_text());hash_check(out,m['files'])
    parent=Path(m['parent_directory']);xm=Path(m['xlam_source_directory'])
    if sha(parent/'manifest.json')!=m['parent_manifest_sha256'] or sha(parent/'audit.json')!=m['parent_audit_sha256']:
        raise ValueError('Parent changed')
    hash_check(parent,json.loads((parent/'manifest.json').read_text())['files'])
    hash_check(Path(m['legacy_source_directory']),{k:v for k,v in m['sources'].items() if k!='xlam'})
    hash_check(xm,m['sources']['xlam']['files'])
    for f,h in m['verifier_sha256'].items():
        if sha(Path(__file__).with_name(f))!=h:raise ValueError('Verifier changed')
    old={s:rows(parent/(s+'.jsonl')) for s in SPLITS};registry=family_registry(old)
    if sha(out/'benchmark-prompts.jsonl')!=sha(parent/'benchmark-prompts.jsonl'):
        raise ValueError('Frozen benchmark prompt reference changed')
    historical=[{'id':r['id'],'split':s,'text':r['problem']} for s in SPLITS for r in old[s] if r['task']=='tools']
    if rows(out/'historical-tool-prompts.jsonl')!=historical:
        raise ValueError('Historical rejection reference changed')
    raw={r['id']:r for r in json.loads((xm/'xlam_function_calling_60k.json').read_text())}
    negative_sources={r['source_index']:r for r in raw_candidates(Path(m['legacy_source_directory']))
                      if r['source']=='glaive' and r['tool_case']=='no_call'}
    index=DuplicateIndex(.8)
    for r in rows(out/'benchmark-prompts.jsonl'):index.add('benchmark:'+r['id'],r['text'])
    for r in rows(out/'historical-tool-prompts.jsonl'):index.add('historical:'+r['id'],r['text'])
    for split in SPLITS:
        for r in old[split]:
            if r['task']!='tools':index.add(r['id'],r['problem'])
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_REVISION,local_files_only=True)
    seen,api_splits,bank={}, {}, {}
    for source in ('xlam',):
        for split in SPLITS:
            for r in rows(out/f'{source}-{split}.jsonl'):
                if r['id'] in seen or index.match(r['problem']):raise ValueError('Residual question overlap')
                fs=sorted({api_family(t['name']) for t in r['tools']})
                if r['split']!=split or fs!=r['api_families'] or family_split(fs,registry)!=split:raise ValueError('API assignment mismatch')
                for family in fs:
                    if family in api_splits and api_splits[family]!=split:raise ValueError('API family leakage')
                    api_splits[family]=split
                if r['source']!='xlam' or r['source_revision']!=m['sources']['xlam']['revision']:raise ValueError('Source revision mismatch')
                expected=xlam_row(raw[r['source_index']])
                for field in ('id','problem','answer','tools','tool_case'):
                    if r[field]!=expected[field]:raise ValueError('Source mapping mismatch: '+field)
                if 'reference' in r or r['prompt']!=prompt_for(r) or not verify(expected['reference'],r['answer']):raise ValueError('Target/prompt mismatch')
                tokens=len(tokenizer.apply_chat_template(r['prompt'],tokenize=True,add_generation_prompt=True,enable_thinking=True))
                if tokens!=r['prompt_tokens'] or tokens>1500:raise ValueError('Token cap mismatch')
                seen[r['id']]=split;bank[r['id']]=r;index.add(r['id'],r['problem'])
    legacy_families=defaultdict(set)
    legacy_api_splits=defaultdict(set)
    for split in SPLITS:
        inherited=rows(out/f'glaive-{split}.jsonl')
        expected_rows=[r for r in old[split] if r['task']=='tools' and r['tool_case']=='no_call']
        if inherited!=expected_rows or len(inherited)!=NO_CALL_COUNTS[split]:
            raise ValueError('Inherited no-call rows changed')
        for r in inherited:
            expected=negative_sources[r['source_index']]
            for field in ('id','problem','answer','tools','tool_case'):
                if r[field]!=expected[field]:raise ValueError('Inherited source mismatch')
            if not verify(expected['reference'],r['answer']):raise ValueError('Invalid inherited reference')
            legacy_families[r['family']].add(split)
            for t in r['tools']:legacy_api_splits[api_family(t['name'])].add(split)
            if r['id'] in seen:raise ValueError('Duplicate inherited ID')
            seen[r['id']]=split;bank[r['id']]=r
    if any(len(v)>1 for v in legacy_families.values()):raise ValueError('Inherited schema family leakage')
    counts={}
    for split in SPLITS:
        rs=rows(out/(split+'.jsonl'));parent_non={r['id']:r for r in old[split] if r['task']!='tools'}
        if {r['id']:r for r in rs if r['task']!='tools'}!=parent_non:raise ValueError('Non-tool domains changed')
        if len({r['id'] for r in rs})!=len(rs):raise ValueError('Duplicate selected ID')
        for r in rs:
            if r['task']=='tools' and (r!=bank.get(r['id']) or seen[r['id']]!=split):raise ValueError('Selected bank/split mismatch')
        counts[split]=dict(Counter(r['task'] for r in rs))
        if counts[split]!={d:(1000 if split=='train' else 64) for d in DOMAINS}:raise ValueError('Unbalanced domains')
        cases=Counter(r['tool_case'] for r in rs if r['task']=='tools')
        n=1000 if split=='train' else 64;neg=NO_CALL_COUNTS[split]
        expected_cases={'single_call':(n-neg)//2,'multiple_calls':n-neg-(n-neg)//2,'no_call':neg}
        if cases!=expected_cases:raise ValueError('Tool case coverage mismatch')
    dev={r['id']:r for r in rows(out/'development.jsonl')};monitor=rows(out/'monitor.jsonl')
    if Counter(r['task'] for r in monitor)!={d:4 for d in DOMAINS} or any(r!=dev.get(r['id']) for r in monitor):raise ValueError('Monitor split mismatch')
    rejected_xlam={r['source_index'] for r in rows(out/'quarantine.jsonl') if r['source']=='xlam'}
    retained_xlam={r['source_index'] for r in bank.values() if r['source']=='xlam'}
    if rejected_xlam & retained_xlam or rejected_xlam|retained_xlam != set(raw):raise ValueError('Incomplete source accounting')
    result={'status':'passed','manifest_sha256':sha(out/'manifest.json'),'domain_counts':counts,
        'new_xlam_residual_exact_lexical_overlap':0,'new_xlam_api_family_split_violations':0,'new_xlam_historical_question_leaks':0,
        'inherited_no_call_rows_unchanged':True,'inherited_no_call_schema_family_violations':0,
        'inherited_no_call_shared_api_names':sum(len(v)>1 for v in legacy_api_splits.values()),
        'non_tool_rows_unchanged':True,'all60000_xlam_rows_accounted':True,'semantic_overlap_guarantee':False,
        'verified_source_bank_rows':len(bank),'api_families':len(api_splits)}
    (out/'audit.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('parent','sources','xlam','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--audit-only',action='store_true');a=p.parse_args()
    result=audit(a.out) if a.audit_only else prepare(a.parent,a.sources,a.xlam,a.out)
    print(json.dumps({k:result[k] for k in ('status','counts','domain_counts','bank_counts','rejections') if k in result},indent=2))
