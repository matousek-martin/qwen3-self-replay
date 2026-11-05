"""Pinned shared prompt preparation and verified rollout selection; no GPU use."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import re
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
BASE_REVISION = 'c1899de289a04d12100db370d81485cdf75e47ca'
SOURCES = {
    'gsm8k': ('openai/gsm8k', 'main', 'train'),
    'skywork': ('Skywork/Skywork-OR1-RL-Data', None, 'math'),
    'commonsense_qa': ('tau/commonsense_qa', None, 'train'),
    'arc_easy': ('allenai/ai2_arc', 'ARC-Easy', 'train'),
}
MATH = 'Put the final answer on the last line, like: Final answer: 42'
CHOICE = 'Answer with the label of the correct option. Put it on the last line, like: Final answer: B'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    with Path(path).open() as stream:
        for i, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f'{path}:{i}: invalid JSON') from exc


def write_jsonl(path, rows):
    with Path(path).open('x') as stream:
        for row in rows:
            stream.write(canonical(row) + '\n')


def fresh_directory(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f'Refusing to overwrite nonempty output: {path}')
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize(text):
    # Remove only known boilerplate. Never inspect benchmark answers or traces.
    for suffix in (MATH, CHOICE,
        "The last line of your reply must start with 'Final answer:' followed by the number alone."):
        text = text.replace(suffix, '')
    return ' '.join(re.findall(r'\w+', text.casefold()))


def prompt_text(row):
    messages = row.get('prompt', row.get('messages'))
    if not isinstance(messages, list):
        raise ValueError('Expected prompt or messages list')
    return '\n'.join(m['content'] for m in messages if m.get('role') == 'user')


def shingles(text):
    words = normalize(text).split()
    return {tuple(words[i:i + 5]) for i in range(len(words) - 4)}


class DuplicateIndex:
    """Exact normalization + 5-gram containment; lexical, not semantic detection."""
    def __init__(self, threshold=0.8):
        if not 0 < threshold <= 1:
            raise ValueError('threshold must be in (0, 1]')
        self.threshold = threshold
        self.exact = {}
        self.tokens = []
        self.ids = []
        self.inverted = defaultdict(set)

    def add(self, identifier, text):
        value = normalize(text)
        index = len(self.ids)
        parts = shingles(text)
        self.exact[value] = identifier
        self.ids.append(identifier)
        self.tokens.append(parts)
        for part in parts:
            self.inverted[part].add(index)

    def match(self, text):
        value = normalize(text)
        if value in self.exact:
            return self.exact[value], 'exact', 1.0
        parts = shingles(text)
        if not parts:
            return None
        candidates = Counter()
        for part in parts:
            candidates.update(self.inverted.get(part, ()))
        for index, overlap in candidates.most_common():
            # Require at least three common 5-grams to avoid tiny phrase matches.
            denominator = min(len(parts), len(self.tokens[index]))
            score = overlap / denominator if denominator else 0
            if overlap >= 3 and score >= self.threshold:
                return self.ids[index], 'lexical_near', round(score, 6)
        return None


def resolve_sources(lock_path):
    lock_path = Path(lock_path)
    if lock_path.exists():
        lock = json.loads(lock_path.read_text())
    else:
        lock = {'schema_version': 1, 'resolved_at': datetime.now(timezone.utc).isoformat(), 'sources': {}}
        for name, (repo, config, split) in SOURCES.items():
            with urllib.request.urlopen('https://huggingface.co/api/datasets/' + repo, timeout=60) as response:
                metadata = json.load(response)
            lock['sources'][name] = {'repo': repo, 'config': config, 'split': split,
                'revision': metadata['sha'], 'license': (metadata.get('cardData') or {}).get('license'),
                'source_url': 'https://huggingface.co/datasets/' + repo}
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text(json.dumps(lock, indent=2) + '\n')
    for name, (repo, config, split) in SOURCES.items():
        item = lock['sources'][name]
        if (item['repo'], item['config'], item['split']) != (repo, config, split):
            raise ValueError(f'Source lock definition mismatch: {name}')
        if not re.fullmatch(r'[0-9a-f]{40}', item['revision']):
            raise ValueError(f'Source revision is not pinned: {name}')
    return lock


def source_rows(name, spec):
    from datasets import load_dataset
    kwargs = {'path': spec['repo'], 'split': spec['split'], 'revision': spec['revision']}
    if spec['config']:
        kwargs['name'] = spec['config']
    for raw in load_dataset(**kwargs):
        if name == 'gsm8k':
            question, answer = raw['question'], raw['answer'].split('####')[-1].strip().replace(',', '')
            task, instruction = 'math', MATH
        elif name == 'skywork':
            gold = json.loads(raw['reward_model']['ground_truth'])
            if len(gold) != 1 or not re.fullmatch(r'-?\d+', gold[0].strip()):
                continue
            if raw['extra_info']['model_difficulty']['DeepSeek-R1-Distill-Qwen-1.5B'] > 8:
                continue
            question, answer = raw['prompt'][-1]['content'], gold[0].strip()
            task, instruction = 'math', MATH
        else:
            choices = raw['choices']
            question = raw['question'] + '\n\n' + '\n'.join(
                f'{label}. {text}' for label, text in zip(choices['label'], choices['text']))
            answer, task, instruction = raw['answerKey'], 'multiple_choice', CHOICE
        prompt = [{'role': 'user', 'content': question + '\n\n' + instruction}]
        row = {'prompt': prompt, 'answer': answer, 'task': task, 'source': name,
               'source_revision': spec['revision']}
        row['id'] = name + '-' + digest(row)[:24]
        yield row


def split_sources(grouped, seed=0, val_per_source=200):
    """Same RNG order/split semantics as the contest helper, before filtering."""
    rng = random.Random(seed)
    train, val = [], []
    for name in SOURCES:
        rows = list(grouped[name])
        rng.shuffle(rows)
        val.extend(dict(row, split='val') for row in rows[:val_per_source])
        train.extend(dict(row, split='train') for row in rows[val_per_source:])
    rng.shuffle(train)
    return train, val


def screen(train, val, benchmark_rows, threshold=0.8):
    index = DuplicateIndex(threshold)
    for row in benchmark_rows:
        index.add('eval:' + row['id'], prompt_text(row))
    kept = {'train': [], 'val': []}
    exclusions = []
    # Validation has priority; screening training against it prevents leakage.
    for split, rows in [('val', val), ('train', train)]:
        for row in rows:
            text = prompt_text(row)
            hit = index.match(text)
            if hit:
                exclusions.append({'id': row['id'], 'split': split, 'matched_id': hit[0],
                                   'reason': hit[1], 'similarity': hit[2]})
            else:
                kept[split].append(row)
                index.add(split + ':' + row['id'], text)
    return kept['train'], kept['val'], exclusions


def balanced_pilot(rows, count, seed):
    if count <= 0 or count > len(rows):
        raise ValueError(f'pilot size must be 1..{len(rows)}')
    rng = random.Random(seed)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['source']].append(row)
    for group in grouped.values():
        rng.shuffle(group)
    selected = []
    while len(selected) < count:
        for name in sorted(grouped):
            if grouped[name] and len(selected) < count:
                selected.append(grouped[name].pop())
    rng.shuffle(selected)
    return selected


def prepare(args):
    out = fresh_directory(args.out)
    if not args.benchmarks.is_file():
        raise ValueError('Frozen benchmark questions are required for contamination screening')
    lock = resolve_sources(args.lock)
    grouped = {name: list(source_rows(name, lock['sources'][name])) for name in SOURCES}
    train, val = split_sources(grouped, args.seed, args.val_per_source)
    original = {'train': len(train), 'val': len(val)}
    train, val, excluded = screen(train, val, read_jsonl(args.benchmarks), args.near_threshold)
    pilot = balanced_pilot(train, args.pilot_size, args.seed)
    for filename, rows in [('train.jsonl', train), ('val.jsonl', val), ('pilot.jsonl', pilot),
                           ('exclusions.jsonl', excluded)]:
        write_jsonl(out / filename, rows)
    (out / 'sources.lock.json').write_text(json.dumps(lock, indent=2) + '\n')
    manifest = {'schema_version': 1, 'created_at': datetime.now(timezone.utc).isoformat(),
                'seed': args.seed, 'val_per_source_before_screening': args.val_per_source,
                'pilot_size': len(pilot), 'expected_rollouts_per_prompt': 4,
                'base_model': 'Qwen/Qwen3-0.6B', 'base_revision': BASE_REVISION,
                'before_screening': original, 'counts': {'train': len(train), 'val': len(val), 'excluded': len(excluded)},
                'train_sources': dict(Counter(r['source'] for r in train)),
                'val_sources': dict(Counter(r['source'] for r in val)),
                'pilot_sources': dict(Counter(r['source'] for r in pilot)),
                'benchmark_file_sha256': file_hash(args.benchmarks),
                'screening': {'method': 'normalized exact and five-token-shingle containment',
                              'threshold': args.near_threshold, 'semantic_paraphrases_detected': False},
                'code_sha256': file_hash(__file__),
                'files': {p.name: file_hash(p) for p in out.iterdir() if p.is_file()}}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


def grader():
    path = ROOT / 'ThinkingCap-Test/utils/train/rewards.py'
    spec = importlib.util.spec_from_file_location('thinkingcap_rewards', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.is_task_correct


def valid_reasoning(completion):
    if not isinstance(completion, str) or completion.count('</think>') != 1:
        return False
    reasoning, answer = completion.split('</think>')
    if reasoning.startswith('<think>'):
        reasoning = reasoning[len('<think>'):]
    return bool(reasoning.strip() and answer.strip()) and '<think>' not in reasoning + answer


def select_rows(prompts, rollouts, seed=0, revision=BASE_REVISION, compatibility_scores=None):
    prompt_map = {row['id']: row for row in prompts}
    if len(prompt_map) != len(prompts):
        raise ValueError('Duplicate prompt IDs')
    if any(row.get('split') != 'train' for row in prompts):
        raise ValueError('Only training prompts may enter solution selection')
    is_correct = grader()
    accepted = defaultdict(list)
    seen = set()
    audit = []
    for row in rollouts:
        identity = (row.get('prompt_id'), row.get('sample_id'))
        if identity in seen:
            raise ValueError(f'Duplicate rollout identity: {identity}')
        seen.add(identity)
        prompt = prompt_map.get(row.get('prompt_id'))
        if prompt is None:
            raise ValueError(f'Unknown prompt_id: {row.get("prompt_id")}')
        if row.get('sample_id') is None:
            raise ValueError('sample_id required')
        if row.get('model_revision') != revision:
            raise ValueError('Rollout model revision mismatch')
        generation = row.get('generation', {})
        if not all(k in generation for k in ['temperature', 'top_p', 'top_k', 'max_new_tokens', 'seed']):
            raise ValueError('Missing generation provenance')
        tokens = row.get('total_tokens')
        if type(tokens) is not int or tokens <= 0 or tokens > generation['max_new_tokens']:
            raise ValueError('Invalid total_tokens')
        if type(row.get('truncated')) is not bool:
            raise ValueError('Explicit truncated boolean required')
        if row.get('finish_reason') not in ['stop', 'eos', 'length']:
            raise ValueError('finish_reason must be stop, eos or length')
        reason = None
        if row['truncated'] or row['finish_reason'] == 'length':
            reason = 'truncated'
        elif not valid_reasoning(row.get('completion')):
            reason = 'malformed_reasoning'
        elif not is_correct(row['completion'], prompt['answer'], prompt['task']):
            reason = 'incorrect'
        else:
            accepted[prompt['id']].append(row)
        audit.append({'prompt_id': prompt['id'], 'sample_id': row['sample_id'],
                      'accepted': reason is None, 'reason': reason})
    scores = None
    if compatibility_scores is not None:
        expected = {(identifier, row['sample_id']): row for identifier, candidates in accepted.items() for row in candidates}
        scores = {}
        for row in compatibility_scores:
            key = (row.get('prompt_id'), row.get('sample_id'))
            if key in scores or key not in expected:
                raise ValueError('Duplicate, unknown, or ineligible compatibility score: ' + str(key))
            value = row.get('verified_answer_mean_logprob')
            if type(value) not in (int, float) or not math.isfinite(value) or value > 0:
                raise ValueError('Invalid compatibility log probability')
            if row.get('model_revision') != revision or row.get('total_tokens') != expected[key]['total_tokens']:
                raise ValueError('Compatibility score provenance differs from rollout')
            scores[key] = value
        if set(scores) != set(expected):
            raise ValueError(f'Incomplete compatibility scores: expected {len(expected)}, received {len(scores)}; refusing biased subset')
    short, random_correct, pairs, compatible = [], [], [], []
    rng = random.Random(seed)
    for identifier in sorted(accepted):
        candidates = sorted(accepted[identifier], key=lambda r: (r['total_tokens'], str(r['sample_id'])))
        prompt = prompt_map[identifier]
        common = {'prompt_id': identifier, 'prompt': prompt['prompt'], 'answer': prompt['answer'],
                  'task': prompt['task'], 'source': prompt['source']}
        for destination, chosen in [(short, candidates[0]), (random_correct, rng.choice(candidates))]:
            destination.append(dict(common, completion=chosen['completion'], total_tokens=chosen['total_tokens'],
                                    sample_id=chosen['sample_id'], model_revision=chosen['model_revision']))
        if scores is not None:
            chosen = min(candidates, key=lambda row: (-scores[(identifier, row['sample_id'])], row['total_tokens'], str(row['sample_id'])))
            compatible.append(dict(common, completion=chosen['completion'], total_tokens=chosen['total_tokens'],
                                   sample_id=chosen['sample_id'], model_revision=chosen['model_revision'],
                                   compatibility_mean_logprob=scores[(identifier, chosen['sample_id'])]))
        if candidates[-1]['total_tokens'] > candidates[0]['total_tokens'] and candidates[-1]['completion'] != candidates[0]['completion']:
            pairs.append(dict(common, chosen=candidates[0]['completion'], rejected=candidates[-1]['completion'],
                              chosen_sample_id=candidates[0]['sample_id'], rejected_sample_id=candidates[-1]['sample_id'],
                              chosen_tokens=candidates[0]['total_tokens'], rejected_tokens=candidates[-1]['total_tokens']))
    outputs = {'sft-short.jsonl': short, 'sft-random.jsonl': random_correct, 'dpo.jsonl': pairs,
               'selection-audit.jsonl': audit}
    if scores is not None:
        outputs['sft-compatible.jsonl'] = compatible
    return outputs


def select(args):
    out = fresh_directory(args.out)
    prompts, rollouts = list(read_jsonl(args.prompts)), list(read_jsonl(args.rollouts))
    scores = None
    score_manifest = None
    if args.compatibility_scores:
        score_manifest = json.loads(args.compatibility_scores.with_suffix('.manifest.json').read_text())
        if score_manifest.get('prompts_sha256') != file_hash(args.prompts) or score_manifest.get('rollouts_sha256') != file_hash(args.rollouts):
            raise ValueError('Compatibility scoring input hashes differ')
        if score_manifest.get('revision') != args.revision:
            raise ValueError('Compatibility scoring model revision differs')
        scores = list(read_jsonl(args.compatibility_scores))
    outputs = select_rows(prompts, rollouts, args.seed, args.revision, scores)
    for filename, rows in outputs.items():
        write_jsonl(out / filename, rows)
    counts = Counter(r['prompt_id'] for r in rollouts)
    manifest = {'schema_version': 1, 'seed': args.seed, 'model_revision': args.revision,
                'prompts_sha256': file_hash(args.prompts), 'rollouts_sha256': file_hash(args.rollouts),
                'verifier_sha256': file_hash(ROOT / 'ThinkingCap-Test/utils/train/rewards.py'),
                'code_sha256': file_hash(__file__), 'input_prompts': len(prompts),
                'input_rollouts': len(rollouts), 'rollout_counts': dict(Counter(counts.get(p['id'], 0) for p in prompts)),
                'counts': {name: len(rows) for name, rows in outputs.items()},
                'files': {p.name: file_hash(p) for p in out.iterdir()}}
    if args.compatibility_scores:
        manifest['compatibility_scores_sha256'] = file_hash(args.compatibility_scores)
        manifest['compatibility_scoring'] = score_manifest
        manifest['compatibility_selection'] = 'maximum verified-answer mean log probability; tie shortest; all eligible candidates required'
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


def validation_pilot(args):
    rows = list(read_jsonl(args.validation))
    if any(row.get('split') != 'val' for row in rows):
        raise ValueError('Validation pilot requires only held-out validation rows')
    pilot = balanced_pilot(rows, args.size, args.seed)
    manifest_path = args.out.with_suffix('.manifest.json')
    if args.out.exists() or manifest_path.exists():
        raise ValueError('Validation pilot output already exists')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out, pilot)
    manifest = {'schema_version': 1, 'seed': args.seed, 'size': len(pilot),
                'validation_sha256': file_hash(args.validation), 'pilot_sha256': file_hash(args.out),
                'sources': dict(Counter(row['source'] for row in pilot)),
                'scope': 'held-out training-source validation; not contest evaluation',
                'code_sha256': file_hash(__file__)}
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--lock', type=Path, default=Path(__file__).with_name('sources.lock.json'))
    prep.add_argument('--benchmarks', type=Path, required=True)
    prep.add_argument('--out', type=Path, required=True)
    prep.add_argument('--seed', type=int, default=0)
    prep.add_argument('--val-per-source', type=int, default=200)
    prep.add_argument('--pilot-size', type=int, default=512)
    prep.add_argument('--near-threshold', type=float, default=0.8)
    prep.set_defaults(func=prepare)
    choose = sub.add_parser('select')
    choose.add_argument('--prompts', type=Path, required=True)
    choose.add_argument('--rollouts', type=Path, required=True)
    choose.add_argument('--out', type=Path, required=True)
    choose.add_argument('--seed', type=int, default=0)
    choose.add_argument('--revision', default=BASE_REVISION)
    choose.add_argument('--compatibility-scores', type=Path)
    choose.set_defaults(func=select)
    val = sub.add_parser('validation-pilot')
    val.add_argument('--validation', type=Path, required=True)
    val.add_argument('--out', type=Path, required=True)
    val.add_argument('--size', type=int, default=64)
    val.add_argument('--seed', type=int, default=0)
    val.set_defaults(func=validation_pilot)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
