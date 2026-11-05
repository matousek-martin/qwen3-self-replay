"""Prepare the complete paper-source dataset, without reading benchmark answers.

CPU only. Never overwrite prior artifacts. Benchmark prompts are used solely as
rejection references; neither they nor upstream solutions enter training files.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from experiments.data_pipeline.pipeline import DuplicateIndex, normalize, prompt_text
from .upstream import load_verifier

MODEL = 'Qwen/Qwen3-0.6B'
MODEL_REVISION = 'c1899de289a04d12100db370d81485cdf75e47ca'
DATA_REVISION = 'b6ae8c60f5c1f2b594e2140b91c49c9ad0949e29'
INSTRUCTION = r'Please reason step by step, and put your final answer within \boxed{}.'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(row):
    return json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def write_rows(path, rows):
    with Path(path).open('x') as out:
        for row in rows:
            out.write(canonical(row) + '\n')


def prepare(sources, benchmark, out, tokenizer, validation_count=1000, confirmation_count=1000):
    sources, benchmark, out = Path(sources), Path(benchmark), Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Refusing nonempty destination')
    source_manifest = json.loads((sources / 'manifest.json').read_text())
    for name, item in source_manifest.items():
        if sha(sources / name) != item['sha256']:
            raise ValueError('Source hash mismatch: ' + name)
    verify = load_verifier(sources)
    # Project prompts immediately; downstream filtering never receives graders,
    # benchmark reference answers, tests, or other benchmark fields.
    benchmark_prompts = []
    for line in benchmark.read_text().splitlines():
        row = json.loads(line)
        benchmark_prompts.append({'id': row['id'], 'text': prompt_text(row)})
    index = DuplicateIndex(0.8)
    for row in benchmark_prompts:
        index.add('benchmark:' + row['id'], row['text'])
    records = json.loads((sources / 'deepscaler.json').read_text())
    accepted, rejected = [], []
    for source_index, raw in enumerate(records):
        problem, answer = raw.get('problem'), raw.get('answer')
        identity = hashlib.sha256(canonical({'problem': problem, 'answer': answer}).encode()).hexdigest()
        identifier = 'deepscaler-' + identity[:24]
        reason = None
        if not isinstance(problem, str) or not problem.strip() or not isinstance(answer, str) or not answer.strip():
            reason = 'empty_or_invalid_question_or_gold'
        elif not verify('\\boxed{' + answer + '}', answer):
            reason = 'gold_fails_upstream_verifier_self_check'
        if reason:
            rejected.append({'source_index': source_index, 'id': identifier, 'reason': reason})
            continue
        hit = index.match(problem)
        if hit:
            rejected.append({'source_index': source_index, 'id': identifier,
                             'reason': hit[1], 'matched_id': hit[0], 'similarity': hit[2]})
            continue
        prompt = [{'role': 'user', 'content': problem.strip() + '\n\n' + INSTRUCTION}]
        ids = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True,
                                            enable_thinking=True)
        if len(ids) > 1500:
            rejected.append({'source_index': source_index, 'id': identifier,
                             'reason': 'overlong_prompt', 'prompt_tokens': len(ids)})
            continue
        # Screen against ALL retained rows before splitting, so lexical near
        # duplicates cannot be distributed across train/dev/confirmation.
        index.add('retained:' + identifier, problem)
        accepted.append({'id': identifier, 'prompt': prompt, 'problem': problem,
                         'answer': answer, 'task': 'math', 'source': 'deepscaler',
                         'source_revision': DATA_REVISION, 'source_index': source_index,
                         'prompt_tokens': len(ids)})
    rng = random.Random(20261026)
    rng.shuffle(accepted)
    if len(accepted) < validation_count + confirmation_count + 19200:
        raise ValueError('Insufficient retained data for the registered one-pass training exposure')
    dev = [dict(r, split='development') for r in accepted[:validation_count]]
    confirm = [dict(r, split='confirmation') for r in accepted[validation_count:validation_count + confirmation_count]]
    train = [dict(r, split='train') for r in accepted[validation_count + confirmation_count:]]
    out.mkdir(parents=True, exist_ok=True)
    files = {'train.jsonl': train, 'development.jsonl': dev,
             'monitor.jsonl': dev[:256], 'confirmation.jsonl': confirm,
             'quarantine.jsonl': rejected, 'benchmark-prompts.jsonl': benchmark_prompts}
    for name, rows in files.items():
        write_rows(out / name, rows)
    manifest = {
        'schema_version': 1, 'source_repo': 'agentica-org/DeepScaleR-Preview-Dataset',
        'source_revision': DATA_REVISION, 'declared_license': 'mit',
        'license_note': 'Upstream card declaration; no independent legal/source-license audit.',
        'source_sha256': sha(sources / 'deepscaler.json'),
        'source_card_sha256': sha(sources / 'deepscaler_card.md'),
        'benchmark_file_sha256': sha(benchmark),
        'tokenizer_model': MODEL, 'tokenizer_revision': MODEL_REVISION,
        'seed': 20261026, 'raw_rows': len(records),
        'counts': {k: len(v) for k, v in files.items()},
        'rejections': dict(Counter(r['reason'] for r in rejected)),
        'screening': {'threshold': 0.8, 'method': 'normalized exact and five-token-shingle containment',
                      'guarantees_zero_semantic_overlap': False},
        'solution_column_used': False, 'synthetic_generation': False,
        'monitor_is_subset_of_development': True,
        'official_benchmarks_are_not_new_untouched_holdout': True,
        'files': {name: sha(out / name) for name in files},
        'preparation_code_sha256': sha(__file__),
        'screening_code_sha256': sha(Path(__file__).parents[1] / 'data_pipeline/pipeline.py'),
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def audit(out):
    """Re-read every output and independently check boundary violations."""
    out = Path(out)
    manifest = json.loads((out / 'manifest.json').read_text())
    for name, expected in manifest['files'].items():
        if sha(out / name) != expected:
            raise ValueError('Prepared file hash mismatch: ' + name)
    index = DuplicateIndex(0.8)
    benchmark_ids = set()
    for line in (out / 'benchmark-prompts.jsonl').read_text().splitlines():
        row = json.loads(line)
        benchmark_ids.add(row['id'])
        index.add('benchmark:' + row['id'], row['text'])
    seen, counts = set(), {}
    for name, split in [('confirmation.jsonl', 'confirmation'),
                        ('development.jsonl', 'development'), ('train.jsonl', 'train')]:
        count = 0
        for line in (out / name).read_text().splitlines():
            row = json.loads(line)
            if row['id'] in seen or row['id'] in benchmark_ids or row['split'] != split:
                raise ValueError('Duplicate ID or split contamination')
            if 'solution' in row or row['source_revision'] != DATA_REVISION:
                raise ValueError('Unexpected solution/provenance')
            expected = row['problem'].strip() + '\n\n' + INSTRUCTION
            if row['prompt'] != [{'role': 'user', 'content': expected}]:
                raise ValueError('Prompt/answer separation failed')
            if index.match(row['problem']):
                raise ValueError('Residual exact/near duplicate: ' + row['id'])
            index.add(split + ':' + row['id'], row['problem'])
            seen.add(row['id'])
            count += 1
        counts[split] = count
    dev = {json.loads(l)['id'] for l in (out / 'development.jsonl').read_text().splitlines()}
    monitor = [json.loads(l)['id'] for l in (out / 'monitor.jsonl').read_text().splitlines()]
    if len(monitor) != 256 or not set(monitor) <= dev:
        raise ValueError('Invalid monitor subset')
    return {'status': 'passed', 'counts': counts, 'manifest_sha256': sha(out / 'manifest.json'),
            'exact_or_lexical_overlap': 0, 'semantic_overlap_guarantee': False,
            'checker_reuses_screening_primitive': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources')
    parser.add_argument('--benchmarks')
    parser.add_argument('--out', required=True)
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    if not args.audit_only:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
        prepare(args.sources, args.benchmarks, args.out, tokenizer)
    result = audit(args.out)
    Path(args.out, 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
