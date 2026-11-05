"""Bounds and measurements for the three explicitly requested hardware trials."""
import statistics
import math
from collections import Counter
from .throughput import read_rows

SCOPE = 'shorterbetter_hardware_comparison_v1'
# Per-job limits chosen to bound four rollouts plus setup/evaluation/archive.
SETUPS = {
    'a100x2': ('NVIDIA A100-SXM4-80GB', 2, 18., 1.79),
    'h100x2': ('NVIDIA H100 80GB HBM3', 2, 32., 3.99),
    'h100x4': ('NVIDIA H100 80GB HBM3', 4, 62., 3.99),
}
ROLLOUTS = 4
MAX_HOURS = 3
# Separate follow-up pilot; keep the original three-setup campaign unchanged.
VAST_A100X4 = ('NVIDIA A100-SXM4-80GB', 4, 15., 1.20)


def validate(protocol):
    e, t = protocol['execution'], protocol['training']
    setup = VAST_A100X4 if e.get('hardware_comparison') == 'vast_a100x4' else SETUPS.get(e.get('hardware_comparison'))
    if not setup:
        raise ValueError('Unknown hardware comparison setup')
    gpu, count, cap, _ = setup
    if (e.get('gpu') != gpu or e.get('world_size') != count
            or not e.get('multidomain') or not e.get('scaled')
            or e.get('phase') != 'benchmark' or e.get('accelerated_multidomain_benchmark')
            or e.get('single_gpu_chunked_projection') or not e.get('deterministic_ddp')
            or e.get('shutdown_reserve_seconds') != 1200
            or not e.get('terminal_evaluation_required')
            or t['rollout_batches'] != ROLLOUTS or t['prompts_per_rollout_batch'] != 64
            or t['generations_per_prompt'] != 8 or t['prompts_per_optimizer_minibatch'] != 16
            or t['max_completion_tokens'] != 6000 or t.get('parameter_dtype') != 'float32'
            or t.get('optimizer_updates_total') != ROLLOUTS*4
            or t.get('training_completions_per_arm') != ROLLOUTS*512
            or protocol.get('data', {}).get('cover_all_training_rows') is not True
            or protocol.get('budget', {}).get('scope') != SCOPE
            or protocol['budget'].get('proposed_total_pilot_cap_usd') != cap
            or protocol.get('authorization', {}).get('full_training_authorized') is not False):
        raise ValueError('Unregistered hardware comparison dimensions')
    return setup


def measure(output, lease):
    """Whole rollout wall time; explicitly exclude the first warmup rollout."""
    starts = read_rows(output/'rollout-starts.jsonl')
    ends = {r['optimizer_step']: r['time'] for r in read_rows(output/'steps.jsonl')}
    durations = [ends[r['optimizer_step']+4]-r['time'] for r in starts if r['optimizer_step']+4 in ends]
    if any(not math.isfinite(t) or t <= 0 for t in durations):
        raise ValueError('Invalid rollout durations')
    steady = durations[1:]
    groups = {}
    tokens = 0
    step_tokens = {}
    for rank in range(lease['gpu_count']):
        for row in read_rows(output/f'rank-{rank:03d}'/'rollouts.jsonl'):
            if row['split'] == 'train':
                groups.setdefault(row['optimizer_step'], []).append(row['prompt_id'])
                tokens += row['total_tokens']
                step_tokens[row['optimizer_step']] = step_tokens.get(row['optimizer_step'], 0) + row['total_tokens']
    expected_steps = [r['optimizer_step'] for r in starts if r['optimizer_step']+4 in ends]
    exposure_valid = all(len(groups.get(step, [])) == 512 and len(set(groups.get(step, []))) == 64
                         and set(Counter(groups[step]).values()) == {8} for step in expected_steps)
    if expected_steps and not exposure_valid:
        raise ValueError('Incomplete measured rollout groups')
    rate = lease.get('rate')
    if rate is not None and (not math.isfinite(rate) or rate <= 0):
        raise ValueError('Invalid provider rate')
    result = {'status': 'measured' if len(steady) == 3 else 'incomplete',
              'gpu': lease['gpu'], 'gpu_count': lease['gpu_count'],
              'rollout_seconds': durations, 'warmup_rollouts_excluded': 1,
              'steady_rollouts': len(steady), 'prompt_ids_by_step': {str(k): sorted(v) for k, v in groups.items()},
              'total_generated_tokens': tokens, 'generated_tokens_by_step': step_tokens,
              'rate_usd_per_hour': rate,
              'limitations': ['Three post-warmup iterations are an early speed estimate, not convergence evidence.',
                             'Generated tokens may differ between GPU types despite identical prompts and seeds.',
                             'Iteration price excludes setup, validation and archive; full rental reported separately.']}
    if steady:
        median = statistics.median(steady)
        result.update(median_rollout_seconds=median, min_rollout_seconds=min(steady), max_rollout_seconds=max(steady),
                      iteration_cost_usd=median/3600*rate if rate is not None else None,
                      training_only_100_hours=median*100/3600,
                      training_only_300_hours=median*300/3600)
    return result


def compare(results):
    """Report a timing/cost comparison, never a claim about model quality."""
    if len(results) != 3 or any(r.get('status') != 'measured' for r in results):
        return {'status': 'incomplete', 'recommendation': 'Finish all three valid hardware measurements.'}
    if len({(r['gpu'], r['gpu_count']) for r in results}) != 3:
        raise ValueError('Duplicate hardware configurations')
    if not results[0].get('comparison_identity') or any(r.get('comparison_identity') != results[0]['comparison_identity'] for r in results[1:]):
        raise ValueError('Different or missing model/data/source identity')
    if any(r['prompt_ids_by_step'] != results[0]['prompt_ids_by_step'] for r in results[1:]):
        raise ValueError('Hardware runs used different training questions')
    fastest = min(results, key=lambda r: r['median_rollout_seconds'])
    priced = all(r.get('iteration_cost_usd') is not None for r in results)
    cheapest = min(results, key=lambda r: r['iteration_cost_usd']) if priced else None
    noisy = any((r['max_rollout_seconds']-r['min_rollout_seconds'])/r['median_rollout_seconds'] > .2 for r in results)
    spans_overlap = any(fastest['max_rollout_seconds'] >= r['min_rollout_seconds'] for r in results if r is not fastest)
    token_totals = [r['total_generated_tokens'] for r in results]
    different_work = max(token_totals) > 1.1*min(token_totals)
    label = lambda r: {'gpu': r['gpu'], 'count': r['gpu_count']} if r else None
    return {'status': 'needs_more_measurement' if noisy or spans_overlap or different_work else 'initial_comparison',
            'fastest_observed': label(fastest), 'lowest_iteration_cost_observed': label(cheapest),
            'high_timing_variability': noisy, 'timing_ranges_overlap': spans_overlap,
            'generated_token_work_differs_over_10pct': different_work,
            'decision_rule': 'Prefer lowest iteration cost meeting the desired runtime; no runtime target has been fixed.',
            'limitation': 'Descriptive ranges from three samples, not statistical confidence intervals or model-quality evidence.'}
