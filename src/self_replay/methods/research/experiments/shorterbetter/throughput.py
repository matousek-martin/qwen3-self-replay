"""Forecast only the actually measured hardware/backend, never ideal GPU scaling."""
import json
import math
from pathlib import Path


def read_rows(path):
    from .jsonl import read_rows as read_jsonl
    return read_jsonl(path) if Path(path).exists() else []


def forecast(output, protocol, lease):
    output = Path(output)
    steps = {r['optimizer_step']: r['time'] for r in read_rows(output/'steps.jsonl')}
    cadence = protocol['monitoring']['every_rollout_batches'] * 4
    intervals = [steps[end]-steps[end-4] for end in sorted(steps)
                 if end % 4 == 0 and end >= 12 and end-4 in steps and (end-4) % cadence != 0]
    evaluations = [r for r in read_rows(output/'rank-000/metrics.jsonl') if 'eval_runtime' in r]
    if len(intervals) < 5 or not evaluations:
        return {'status': 'insufficient_measurements', 'measured_rollouts': len(intervals)}
    if any(not math.isfinite(x) or x <= 0 for x in intervals):
        raise ValueError('Invalid timing samples')
    rollout = max(intervals)
    evaluation = max(r['eval_runtime'] for r in evaluations)
    setup = max(0, evaluations[0]['time']-evaluations[0]['eval_runtime']-lease['created_at'])
    # Checkpoint/export and archive reserve is explicitly a planning allowance,
    # not a measurement; 20% further margin avoids promising a hard SLA.
    checkpoint_and_archive_allowance = 900
    target = protocol.get('training', {}).get('rollout_batches', 300)
    validations = target // protocol['monitoring']['every_rollout_batches'] + 1
    seconds = setup + 1.2 * (target*rollout + validations*evaluation) + checkpoint_and_archive_allowance
    return {'status': 'measured_projection', 'gpu': lease['gpu'], 'gpu_count': lease['gpu_count'],
        'measured_rollouts': len(intervals), 'slowest_measured_rollout_seconds': rollout,
        'slowest_validation_seconds': evaluation, 'setup_seconds': setup,
        'checkpoint_and_archive_allowance_seconds': checkpoint_and_archive_allowance,
        'variability_multiplier': 1.2, 'projected_full_hours': seconds/3600,
        'fits_eight_hours_estimate': seconds <= 8*3600,
        'projected_reserved_cost_usd': seconds/3600*lease['budget_rate'],
        'scope': f'One seed/arm, {target} rollouts, {validations} internal validations; no final competition suite',
        'limitation': 'Forecast from early batches; not a runtime guarantee or efficacy result'}
