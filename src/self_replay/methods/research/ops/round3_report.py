"""Paired question-cluster bootstrap on the independently held-out development set."""
import json
import math
from pathlib import Path
import numpy as np


def load(path):
    rows = [json.loads(x) for x in Path(path).read_text().splitlines()]
    if len(rows) != 1536 or len({(r['prompt_id'], r['attempt']) for r in rows}) != 1536:
        raise ValueError('Incomplete eight-sample development evaluation')
    groups = {}
    for row in rows:
        groups.setdefault(row['prompt_id'], []).append(row)
    if len(groups) != 192 or any(len(v) != 8 for v in groups.values()):
        raise ValueError('Incomplete question coverage')
    return groups


def compare(base_path, candidate_path):
    base, candidate = load(base_path), load(candidate_path)
    if set(base) != set(candidate): raise ValueError('Unpaired questions')
    if not any(base[k] != candidate[k] for k in base):
        raise ValueError('No measurable adapter effect')
    sources = sorted({v[0]['source'] for v in base.values()})
    domains = {}; batches = []
    rng = np.random.default_rng(20261027)
    for source in sources:
        keys = sorted(k for k in base if base[k][0]['source'] == source)
        if len(keys) != 64: raise ValueError('Unexpected development source coverage')
        pairs = []
        for k in keys:
            b = sorted(base[k], key=lambda x: x['attempt']); c = sorted(candidate[k], key=lambda x: x['attempt'])
            if any(x['seed'] != y['seed'] for x, y in zip(b, c)): raise ValueError('Unpaired sampling seeds')
            pairs.append([sum(r['correct'] for r in b)/8, sum(r['correct'] for r in c)/8,
                          sum(r['total_tokens'] for r in b)/8, sum(r['total_tokens'] for r in c)/8])
        values = np.asarray(pairs, dtype=float)
        observed = values.mean(axis=0)
        boot = values[rng.integers(0, len(keys), size=(10000, len(keys)))].mean(axis=1)
        batches.append(boot)
        domains[source] = {'accuracy_delta': float(observed[1]-observed[0]),
            'token_reduction': float(1-observed[3]/observed[2]),
            'baseline_accuracy': float(observed[0]), 'candidate_accuracy': float(observed[1]),
            'intervals': {'accuracy_delta': np.quantile(boot[:,1]-boot[:,0],[.025,.975]).tolist(),
                          'token_ratio': np.quantile(boot[:,3]/boot[:,2],[.025,.975]).tolist()}}
    deltas = np.stack([b[:,1]-b[:,0] for b in batches]).mean(axis=0)
    ratios = np.exp(np.stack([np.log(b[:,3]/b[:,2]) for b in batches]).mean(axis=0))
    average = {'accuracy_delta': sum(v['accuracy_delta'] for v in domains.values())/len(domains),
               'token_reduction': 1-math.exp(sum(math.log1p(-v['token_reduction']) for v in domains.values())/len(domains)),
               'intervals': {'accuracy_delta': np.quantile(deltas,[.025,.975]).tolist(),
                             'token_ratio': np.quantile(ratios,[.025,.975]).tolist()}}
    return {'average': average, 'domains': domains, 'bootstrap': {'samples': 10000, 'seed': 20261027,
        'unit': 'question, all8paired samples kept together; stratified by source',
        'scope': 'Within one training seed; no inferred combined across-seed CI.'}}
