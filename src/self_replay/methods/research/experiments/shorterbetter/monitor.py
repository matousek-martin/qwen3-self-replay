"""Read-only internal validation report; never reads official benchmark scores."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import statistics


def summarize(path, expected_questions=None, task=None):
    groups = defaultdict(list)
    paths = path if isinstance(path, (list, tuple)) else [path]
    for item in paths:
        with Path(item).open() as stream:
            for line in stream:
                row = json.loads(line)
                if row['split'] == 'development' and (task is None or row.get('task') == task):
                    groups[(row['optimizer_step'], row['prompt_id'])].append(row)
    checkpoints = defaultdict(dict)
    for (step, identifier), rows in groups.items():
        if len(rows) != 8:
            raise ValueError(f'Incomplete or repeated validation: step={step}, id={identifier}, n={len(rows)}')
        checkpoints[step][identifier] = {
            'accuracy': statistics.mean(r['correct'] for r in rows),
            'completed_accuracy': statistics.mean(r['correct'] and not r['truncated']
                                                  and r['closed_thinking'] for r in rows),
            'tokens': statistics.mean(r['total_tokens'] for r in rows),
            'truncated': statistics.mean(r['truncated'] for r in rows),
            'invalid_thinking': statistics.mean(not r['closed_thinking'] for r in rows),
        }
    if 0 not in checkpoints:
        raise ValueError('Step-zero validation is required')
    baseline = checkpoints[0]
    if expected_questions is not None and len(baseline) != expected_questions:
        raise ValueError('Incomplete baseline: unexpected question count')
    report = []
    for step, values in sorted(checkpoints.items()):
        if set(values) != set(baseline):
            raise ValueError('Unmatched validation questions')
        ids = sorted(values)
        deltas = [values[i]['accuracy'] - baseline[i]['accuracy'] for i in ids]
        completed_deltas = [values[i]['completed_accuracy'] - baseline[i]['completed_accuracy'] for i in ids]
        rng = random.Random(20261026)
        boot = sorted(statistics.mean(rng.choices(deltas, k=len(deltas))) for _ in range(2000))
        rng = random.Random(20261026)
        completed_boot = sorted(statistics.mean(rng.choices(completed_deltas, k=len(ids))) for _ in range(2000))
        rng = random.Random(20261026)
        token_boot = []
        for _ in range(2000):
            sampled = rng.choices(ids, k=len(ids))
            token_boot.append(1 - sum(values[i]['tokens'] for i in sampled)
                              / sum(baseline[i]['tokens'] for i in sampled))
        token_boot.sort()
        report.append({'optimizer_step': step, 'rollout_batches': step / 4,
            'questions': len(ids), 'accuracy': statistics.mean(v['accuracy'] for v in values.values()),
            'accuracy_delta_pp': 100 * statistics.mean(deltas),
            'paired_question_accuracy_ci95_pp': [100 * boot[49], 100 * boot[1949]],
            'completed_accuracy': statistics.mean(v['completed_accuracy'] for v in values.values()),
            'completed_accuracy_delta_pp': 100 * statistics.mean(completed_deltas),
            'completed_accuracy_ci95_pp': [100 * completed_boot[49], 100 * completed_boot[1949]],
            'mean_total_tokens': statistics.mean(v['tokens'] for v in values.values()),
            'token_saving': 1 - sum(v['tokens'] for v in values.values()) / sum(v['tokens'] for v in baseline.values()),
            'token_saving_ci95': [token_boot[49], token_boot[1949]],
            'truncation_rate': statistics.mean(v['truncated'] for v in values.values()),
            'invalid_thinking_rate': statistics.mean(v['invalid_thinking'] for v in values.values())})
    return {'scope': 'Repeatedly inspected internal monitoring; not final evidence of efficacy',
            'checkpoints': report}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('rollouts')
    args = p.parse_args()
    print(json.dumps(summarize(args.rollouts), indent=2))
