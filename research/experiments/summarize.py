"""Paired, question-cluster pilot summaries. These are not contest scores."""
from collections import defaultdict
import argparse
import json
import math
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments/offline_training'))
from score import is_task_correct, valid_reasoning


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def paired_metrics(prompts, baseline, candidate, draws=2000):
    def index(records):
        indexed = {}
        for row in records:
            key = (row['prompt_id'], row['sample_id'])
            if key in indexed:
                raise ValueError('Duplicate validation response')
            indexed[key] = row
        return indexed
    base, cand = index(baseline), index(candidate)
    expected = {(p['id'], s) for p in prompts for s in range(2)}
    if set(base) != expected or set(cand) != expected:
        raise ValueError('Mismatched or incomplete paired validation responses')
    grouped = defaultdict(list)
    for prompt in prompts:
        values = []
        for sample in range(2):
            pair = []
            for response in [base[(prompt['id'], sample)], cand[(prompt['id'], sample)]]:
                correct = (not response['truncated'] and valid_reasoning(response['completion'])
                           and is_task_correct(response['completion'], prompt['answer'], prompt['task']))
                if response['total_tokens'] <= 0:
                    raise ValueError('Nonpositive token count')
                pair.extend([int(correct), response['total_tokens']])
            values.append(pair)
        grouped[prompt['source']].append(tuple(sum(v[i] for v in values) for i in range(4)))

    def metrics(groups):
        differences, ratios = [], []
        for group in groups.values():
            bcorrect, btokens, ccorrect, ctokens = (sum(v[i] for v in group) for i in range(4))
            differences.append((ccorrect - bcorrect) / (2 * len(group)))
            ratios.append(ctokens / btokens)
        return (100 * sum(differences) / len(differences),
                100 * (1 - math.exp(sum(math.log(r) for r in ratios) / len(ratios))))

    point = metrics(grouped)
    rng = random.Random(20261023)
    samples = [metrics({k: [rng.choice(v) for _ in v] for k, v in grouped.items()}) for _ in range(draws)]
    intervals = []
    for i in range(2):
        ordered = sorted(s[i] for s in samples)
        intervals.append([ordered[int(.025 * (draws - 1))], ordered[int(.975 * (draws - 1))]])
    return {'accuracy_delta_pp': point[0], 'accuracy_delta_95ci': intervals[0],
            'total_tokens_saved_percent': point[1], 'tokens_saved_95ci': intervals[1],
            'bootstrap': 'paired question clusters, stratified by training source', 'bootstrap_draws': draws}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', type=Path, default=ROOT / 'lab/experiments/campaign.json')
    campaign_path = parser.parse_args().campaign
    campaign = json.loads(campaign_path.read_text())
    outputs = [Path(j['output']) for j in campaign['jobs']]
    base_path = next((p / 'validation/base.jsonl' for p in outputs if (p / 'validation/base.jsonl').exists()), None)
    if base_path is None:
        raise SystemExit('No base validation is available yet')
    base_manifest = json.loads(base_path.with_suffix('.manifest.json').read_text())
    prompts = read(ROOT / 'lab/experiments/data/val-pilot.jsonl')
    results = []
    for folder in outputs:
        for path in (folder / 'validation').glob('*.score.json'):
            name = path.name.removesuffix('.score.json')
            if name == 'base':
                continue
            rollouts = path.with_name(name + '.jsonl')
            manifest = json.loads(rollouts.with_suffix('.manifest.json').read_text())
            for key in ['prompt_sha256', 'samples', 'seed', 'max_new_tokens', 'max_model_len', 'backend',
                        'temperature', 'top_p', 'top_k', 'enable_thinking', 'revision', 'limit']:
                if manifest[key] != base_manifest[key]:
                    raise ValueError('Validation settings differ for ' + name + ': ' + key)
            training = folder / 'training' / name
            result_path = training / 'result.json'
            if result_path.exists():
                complete = json.loads(result_path.read_text()).get('complete', False)
            else:
                result_path = training / 'status.json'
                complete = result_path.exists() and json.loads(result_path.read_text()).get('status') == 'completed'
            results.append({'name': name, 'training_complete': complete,
                            **paired_metrics(prompts, read(base_path), read(rollouts))})
    report = {'scope': '64-question training-source pilot validation; not official contest scores',
              'caution': 'Small exploratory comparisons, no multiple-comparison correction. An interval containing zero does not establish equal accuracy. Math/choice results do not establish transfer to coding, tools or instruction following.',
              'baseline': str(base_path), 'candidates': results}
    target = ROOT / 'lab/experiments/pilot-comparison.json'
    target.write_text(json.dumps(report, indent=2) + '\n')
    print(target)


if __name__ == '__main__':
    main()
