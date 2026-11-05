"""Remote pilot roles. All comparisons use frozen training-source validation.

These are inexpensive exploratory runs, not contest benchmark results.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
OUT = ROOT / 'output'
OFFLINE = ROOT / 'experiments/offline_training'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run(*args):
    cmd = [sys.executable, *map(str, args)]
    print('RUN', ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def generate(prompts, output, adapter=None, samples=2, limit=0):
    args = [OFFLINE / 'generate.py', '--prompts', prompts, '--output', output,
            '--samples', samples, '--max-new-tokens', 4096, '--batch-size', 16,
            '--max-seconds', 0]
    if adapter:
        args += ['--adapter', adapter]
    if limit:
        args += ['--limit', limit]
    run(*args)
    if not json.loads(output.with_suffix('.progress.json').read_text()).get('complete'):
        raise RuntimeError('Generation did not finish: ' + str(output))


def validate(name, adapter=None):
    output = OUT / 'validation' / (name + '.jsonl')
    generate(DATA / 'val-pilot.jsonl', output, adapter)
    run(OFFLINE / 'score.py', '--prompts', DATA / 'val-pilot.jsonl', '--rollouts', output,
        '--output', output.with_suffix('.score.json'), '--expected-samples', 2)


def bootstrap():
    rollouts = OUT / 'shared/rollouts.jsonl'
    generate(DATA / 'pilot.jsonl', rollouts, samples=4)
    run(ROOT / 'experiments/data_pipeline/pipeline.py', 'select', '--prompts', DATA / 'pilot.jsonl',
        '--rollouts', rollouts, '--out', OUT / 'shared/selected')
    for method, filename in [('sft', 'sft-short.jsonl'), ('dpo', 'dpo.jsonl')]:
        if not rows(OUT / 'shared/selected' / filename):
            write(OUT / 'smoke' / (method + '-skipped.json'), {'reason': 'No verified examples for this method'})
            continue
        run(OFFLINE / 'train.py', '--method', method, '--data', OUT / 'shared/selected' / filename,
            '--output', OUT / 'smoke' / method, '--steps', 1, '--grad-accum', 1,
            '--max-seconds', 600)
        smoke = json.loads((OUT / 'smoke' / method / 'result.json').read_text())
        if not smoke['complete']:
            raise RuntimeError(method + ' GPU smoke did not complete')
        generate(DATA / 'val-pilot.jsonl', OUT / 'smoke' / (method + '-reload.jsonl'),
                 smoke['adapter'], samples=1, limit=4)
    counts = defaultdict(list)
    by_id = {r['id']: r for r in rows(DATA / 'pilot.jsonl')}
    for response in rows(rollouts):
        counts[by_id[response['prompt_id']]['source']].append(response['total_tokens'])
    write(OUT / 'shared/references.json', {k: sum(v) / len(v) for k, v in counts.items()})
    validate('base')
    run(OFFLINE / 'score_compatibility.py', '--prompts', DATA / 'pilot.jsonl',
        '--rollouts', rollouts, '--output', OUT / 'shared/compatibility.jsonl', '--max-seconds', 3600)
    # The selector re-verifies answers; compatibility scores never substitute for correctness.
    run(ROOT / 'experiments/data_pipeline/pipeline.py', 'select', '--prompts', DATA / 'pilot.jsonl',
        '--rollouts', rollouts, '--out', OUT / 'shared/compatible',
        '--compatibility-scores', OUT / 'shared/compatibility.jsonl')


def sft(name, dataset, method='sft'):
    target = OUT / 'training' / name
    run(OFFLINE / 'train.py', '--method', method, '--data', dataset, '--output', target,
        '--steps', 64, '--grad-accum', 8, '--max-seconds', 0, '--checkpoint-every', 16)
    result = json.loads((target / 'result.json').read_text())
    validate(name, result['adapter'])
    if not result['complete']:
        raise RuntimeError(name + ' stopped early; checkpoint and validation retained, matched-step comparison incomplete')
    return result


def rl(name, variant, optimizer='dr_grpo', references=False):
    target = OUT / 'training' / name
    args = ['-m', 'experiments.rl_training.train', '--data', DATA / 'pilot.jsonl', '--out', target,
            '--variant', variant, '--optimizer', optimizer, '--max-steps', 128,
            '--max-seconds', 0, '--max-new-tokens', 2048, '--final-cap', 1024,
            '--group-size', 4, '--save-every', 32]
    if references:
        args += ['--references', DATA / 'shared/references.json']
    run(*args)
    validate(name, target / 'adapter')
    result = json.loads((target / 'status.json').read_text())
    if result['status'] != 'completed' or result['completed_groups'] != 128:
        raise RuntimeError(name + ' stopped early; checkpoint retained, matched rollout budget incomplete')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('role', choices=['bootstrap', 'rl-controls', 'sft', 'dpo', 'rl-shaping', 'rl-preserve'])
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    start = time.time()
    results = {}
    if args.role == 'bootstrap':
        bootstrap()
    elif args.role == 'rl-controls':
        run('-m', 'experiments.rl_training.train', '--data', DATA / 'pilot.jsonl',
            '--out', OUT / 'smoke/rl', '--max-steps', 8, '--max-seconds', 900,
            '--variant', 'correctness', '--optimizer', 'dr_grpo')
        smoke = json.loads((OUT / 'smoke/rl/status.json').read_text())
        if smoke['optimizer_updates'] < 1 or smoke['status'] != 'completed':
            raise RuntimeError('Bounded RL smoke produced no verified optimizer update or timed out; inspect reward density before expansion')
        generate(DATA / 'val-pilot.jsonl', OUT / 'smoke/rl-reload.jsonl', OUT / 'smoke/rl/adapter', samples=1, limit=4)
        results['grpo'] = rl('grpo', 'correctness', 'grpo')
        results['dr-grpo'] = rl('dr-grpo', 'correctness')
    elif args.role == 'sft':
        results['sft-short'] = sft('sft-short', DATA / 'shared/selected/sft-short.jsonl')
        results['sft-random'] = sft('sft-random', DATA / 'shared/selected/sft-random.jsonl')
        results['sft-compatible'] = sft('sft-compatible', DATA / 'shared/compatible/sft-compatible.jsonl')
    elif args.role == 'dpo':
        pairs = rows(DATA / 'shared/selected/dpo.jsonl')
        if not pairs:
            raise RuntimeError('No verified preference pairs; no DPO run launched')
        eligible = {r['prompt_id'] for r in pairs}
        matched = [r for r in rows(DATA / 'shared/selected/sft-short.jsonl') if r['prompt_id'] in eligible]
        path = OUT / 'dpo-matched-sft.jsonl'
        path.write_text(''.join(json.dumps(r) + '\n' for r in matched))
        results['sft-dpo-matched'] = sft('sft-dpo-matched', path)
        results['dpo'] = sft('dpo', DATA / 'shared/selected/dpo.jsonl', 'dpo')
    elif args.role == 'rl-shaping':
        for variant in ['fixed', 'adaptive', 'difficulty']:
            results[variant] = rl('rl-' + variant, variant, references=True)
    elif args.role == 'rl-preserve':
        for variant in ['progressive', 'replay']:
            results[variant] = rl('rl-' + variant, variant)
    write(OUT / 'role-result.json', {'role': args.role, 'elapsed_seconds': time.time() - start,
          'scope': 'training-source pilot validation; not a contest evaluation or paper reproduction',
          'results': results})


if __name__ == '__main__':
    main()
