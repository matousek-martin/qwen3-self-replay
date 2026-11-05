"""Prompt-cluster development metrics; these are not official benchmark scores."""
import random
import statistics
from collections import defaultdict
from .core import DOMAINS

def summarize(records, baseline=None):
    out={}
    for domain in ('overall',)+DOMAINS:
        rows=[r for r in records if domain=='overall' or r['task']==domain]
        if not rows: continue
        value=dict(accuracy=statistics.mean(r['correct'] for r in rows),
                   mean_tokens=statistics.mean(r['total_tokens'] for r in rows),
                   median_tokens=statistics.median(r['total_tokens'] for r in rows),
                   truncation=statistics.mean(r['truncated'] for r in rows),
                   questions=len({r['id'] for r in rows}),responses=len(rows))
        if baseline is not None:
            bs={(r['id'],r['sample']):r for r in baseline if domain=='overall' or r['task']==domain}
            if set(bs)!={(r['id'],r['sample']) for r in rows}:raise ValueError('Mismatched monitor pairs')
            grouped=defaultdict(list)
            for r in rows:grouped[r['id']].append(float(r['correct'])-float(bs[r['id'],r['sample']]['correct']))
            delta=[statistics.mean(x) for x in grouped.values()];rng=random.Random(42)
            boot=sorted(statistics.mean(rng.choices(delta,k=len(delta))) for _ in range(1000))
            value.update(accuracy_delta=statistics.mean(delta),accuracy_delta_ci=[boot[24],boot[974]],
                         token_saving=1-value['mean_tokens']/statistics.mean(r['total_tokens'] for r in bs.values()),
                         interval_method='paired_prompt_cluster_percentile_1000')
        out[domain]=value
    return out
