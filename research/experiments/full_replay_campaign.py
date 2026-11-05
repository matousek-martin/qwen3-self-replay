"""Durable full-evaluation controller and W&B publisher; credentials stay here."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

from . import cloud
from . import full_replay_eval
from . import study_eval as study

ROOT = study.ROOT


def benchmark_metrics(results):
    groups = {}
    for dataset, value in results['results'].items():
        group = 'bfcl' if dataset.startswith('bfcl_') else dataset
        groups.setdefault(group, []).append(value)
    output = {}
    for name, values in groups.items():
        count = sum(v['n'] for v in values)
        output[name] = {'responses': count}
        for metric in ('accuracy', 'mean_output_tokens', 'mean_reasoning_tokens', 'truncation_rate'):
            output[name][metric] = sum(v[metric] * v['n'] for v in values) / count
    output['macro'] = {key: sum(v[key] for v in output.values()) / len(output)
                       for key in ('accuracy', 'mean_output_tokens', 'mean_reasoning_tokens', 'truncation_rate')}
    return output


def final_metrics(output):
    comparison = json.loads((output / 'comparison.json').read_text())
    if comparison['ungraded_benchmarks'] or any(comparison['full_evaluation'].values()):
        raise ValueError('Incomplete official comparison')
    metrics, table = {}, []
    results = {name: benchmark_metrics(json.loads((output / 'runs' / name / 'results.json').read_text()))
               for name in ('base', 'rl-replay')}
    for name, groups in results.items():
        for domain, values in groups.items():
            for key, value in values.items():
                metrics[f'{name}/{domain}/{key}'] = value
    for domain in results['base']:
        base, candidate = (results[name][domain] for name in ('base', 'rl-replay'))
        thinking = 100 * (1 - candidate['mean_reasoning_tokens'] / base['mean_reasoning_tokens'])
        metrics[f'comparison/{domain}/thinking_token_reduction_percent'] = thinking
        table.append([domain, base['accuracy'] * 100, candidate['accuracy'] * 100,
                      100 * (candidate['accuracy'] - base['accuracy']),
                      base['mean_output_tokens'], candidate['mean_output_tokens'], thinking,
                      base['truncation_rate'], candidate['truncation_rate']])
    for domain, values in dict(comparison['results'], macro=comparison['average']).items():
        prefix = f'comparison/{domain}/'
        for key, unit in [('accuracy_delta', 'pp'), ('token_reduction', 'percent')]:
            metrics[prefix + key + '_' + unit] = values[key] * 100
            for side, value in zip(('lower', 'upper'), values['intervals'][key]):
                metrics[prefix + key + '_ci_' + side + '_' + unit] = value * 100
    interval = comparison['average']['intervals']['accuracy_delta']
    metrics['conclusion/zero_regression_established'] = interval[0] >= 0
    metrics['conclusion/one_pp_noninferiority_supported'] = interval[0] >= -0.01
    metrics['evaluation/complete'] = True
    return metrics, table


def progress(output):
    values = {}
    for name in ('base', 'rl-replay'):
        records = output / 'runs' / name / 'records.jsonl'
        if records.exists():
            with records.open() as stream:
                values[f'progress/{name}/saved_responses'] = sum(1 for _ in stream)
        status = output / 'status' / (name + '.json')
        if status.exists():
            values[f'progress/{name}/graded_complete'] = json.loads(status.read_text())['state'] == 'complete'
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--publish-only', action='store_true', help='Resume W&B publication without renting compute')
    args = parser.parse_args()
    if not args.name.startswith('full-replay-') or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-' for c in args.name):
        raise ValueError('Unsafe job name')
    archive = ROOT / 'lab/experiments/full-replay' / args.name
    archive.mkdir(parents=True, exist_ok=True)
    state_path = archive / 'campaign.json'
    previous = json.loads(state_path.read_text()) if state_path.exists() else None
    if previous and not args.publish_only:
        raise RuntimeError('Existing campaign; inspect it rather than allocate twice')
    config_path = ROOT / 'lab/experiments/data/study/full-replay-evaluation.json'
    config, _, rows, _, hashes, lock = full_replay_eval.inputs(config_path)
    with cloud.ledger() as ledger:
        commitment = cloud.liability(ledger['jobs'])
        existing = ledger['jobs'].get(args.name)
    if existing and not args.publish_only:
        raise RuntimeError('Job already exists in budget ledger')
    if not args.publish_only and commitment + 7.8 > cloud.USABLE:
        raise RuntimeError('Insufficient remaining budget for initial six-hour reservation')
    for line in (ROOT / '.env').read_text().splitlines():
        line = line.strip().removeprefix('export ')
        if '=' in line and not line.startswith('#'):
            name, value = line.split('=', 1)
            if name in ('WANDB_API_KEY', 'WANDB_ENTITY', 'WANDB_BASE_URL'):
                os.environ[name] = shlex.split(value, comments=True)[0]
    if not os.environ.get('WANDB_API_KEY'):
        raise RuntimeError('Online W&B credentials required before allocation')
    import wandb
    wandb_config = dict(config, questions=len(rows), model=study.BASE_MODEL, revision=study.BASE_REVISION,
                        candidate_lock=lock, adapter_files_sha256=hashes, fresh_holdout=False,
                        full_contest_evaluation=True, budget_ceiling_usd=cloud.CEILING,
                        conservative_commitment_before_launch_usd=commitment,
                        initial_reservation_usd=7.8, training_performed=False)
    run = wandb.init(project='thinkingcap', entity=os.environ.get('WANDB_ENTITY'),
                     name=args.name, job_type='full-paired-evaluation', config=wandb_config,
                     id=previous['wandb_id'] if previous else None,
                     resume='must' if previous else None, dir=str(archive), mode='online',
                     settings=wandb.Settings(disable_git=True, disable_code=True, save_code=False,
                                             x_disable_stats=True))
    state = previous or dict(job=args.name, created_at=study.now(), state='starting',
                             wandb_url=run.url, wandb_id=run.id, config=wandb_config)
    study.write_json(state_path, state)
    print('W&B:', run.url, flush=True)
    job_root = cloud.STATE / args.name
    output = job_root / 'output/full-evaluation'
    try:
        if not args.publish_only:
            with (archive / 'cloud-controller.log').open('a') as stream:
                process = subprocess.Popen([sys.executable, '-m', 'experiments.cloud', 'run',
                    '--name', args.name, '--entry', 'experiments/run-full-replay-eval.sh',
                    '--gpu', 'NVIDIA GeForce RTX 4090', '--hours', '6'], cwd=ROOT,
                    stdout=stream, stderr=subprocess.STDOUT)
                state.update(state='running', controller_pid=process.pid)
                study.write_json(state_path, state)
                while process.poll() is None:
                    try:
                        run.log(progress(output))
                    except Exception as exc:
                        print('W&B progress update failed:', type(exc).__name__, flush=True)
                    time.sleep(45)
                if process.returncode:
                    raise RuntimeError('Cloud controller failed; inspect cloud-controller.log and budget ledger')
        result = cloud.verify_backup(args.name)
        if result['exit_code'] != 0:
            raise RuntimeError('Worker failed; partial outputs preserved')
        with cloud.ledger() as ledger:
            job = ledger['jobs'][args.name]
        if not job.get('released_at') or not job.get('backup_verified'):
            raise RuntimeError('Await verified backup and confirmed GPU release before publication')
        for name in ('base', 'rl-replay'):
            study.validate_records(output / 'runs' / name, rows, 8,
                json.loads((output / 'provenance.json').read_text())['cohort_sha256'], 'full')
        metrics, table = final_metrics(output)
        metrics['cost/conservative_job_usd'] = job['charged_estimate_usd']
        run.log(metrics)
        run.summary.update(metrics)
        run.log({'benchmark_comparison': wandb.Table(columns=[
            'domain', 'baseline_accuracy_percent', 'replay_accuracy_percent', 'accuracy_delta_pp',
            'baseline_output_tokens', 'replay_output_tokens', 'thinking_token_reduction_percent',
            'baseline_truncation_rate', 'replay_truncation_rate'], data=table)})
        artifact = wandb.Artifact(args.name, type='evaluation', metadata={
            'full_evaluation': True, 'fresh_holdout': False, 'questions': 2759, 'samples': 8})
        artifact.add_dir(str(output), name='full-evaluation')
        for name in ('SHA256SUMS', 'result.json', 'exit.json', 'gpu.txt', 'packages.txt',
                     'vllm-compat.json', 'vllm-lora-smoke.json'):
            artifact.add_file(str(job_root / 'output' / name), name=name)
        artifact.add_file(str(job_root / 'source/manifest.json'), name='source-manifest.json')
        artifact.add_file(str(job_root / 'source/requirements.lock'), name='requirements.lock')
        run.log_artifact(artifact).wait()
        state.update(state='complete', completed_at=study.now(), gpu_released=True,
                     backup_verified=True, metrics=metrics)
        study.write_json(state_path, state)
        run.finish()
    except BaseException as exc:
        state.update(state='failed', error=str(exc), updated_at=study.now())
        study.write_json(state_path, state)
        run.finish(exit_code=1)
        raise


if __name__ == '__main__':
    main()
