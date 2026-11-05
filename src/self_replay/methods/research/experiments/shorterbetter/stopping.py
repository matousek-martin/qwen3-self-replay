"""Registered operational stopping rules, not a sequential efficacy test.

Repeated pointwise CIs are descriptive. Early stopping is a spending decision;
success on this reused development panel still needs independent confirmation.
"""
import math

DEFAULTS = {
    'regression_min_rollouts': 20, 'regression_patience': 3,
    'regression_margin_pp': 3.0, 'collapse_increase': .15,
    'minimum_rollouts': 100, 'plateau_evaluations': 5,
    'stagnation_patience': 5, 'accuracy_tolerance_pp': 2.0,
    'target_token_saving': .10, 'minimum_token_saving_lcb': .05,
    'max_truncation_increase': .02, 'max_invalid_increase': .02,
    'plateau_token_range': .02, 'plateau_accuracy_range_pp': 1.0,
    'promising_token_saving': .05, 'promising_accuracy_gain_pp': 1.0,
}


def apply_quality_policy(decision, protocol):
    """Keep quality diagnostics while honoring a registered monitoring-only run."""
    if protocol.get('stopping', {}).get('quality_stopping_enabled', True):
        return decision
    if (not protocol.get('execution', {}).get('multidomain_full_training')
            or not protocol.get('authorization', {}).get('quality_stopping_disabled_by_user')):
        raise ValueError('Quality stopping override requires explicit full-run authorization')
    quality_reasons = {'persistent_regression', 'no_useful_progress_within_window',
                       'useful_plateau_needs_confirmation'}
    result = dict(decision, quality_stopping_enabled=False)
    if decision['reason'] in quality_reasons:
        result.update(stop=False, reason='quality_monitoring_only',
                      quality_stop_recommendation=dict(decision))
    return result


def decide(report, settings):
    c = {**DEFAULTS, **settings}
    rows = report['checkpoints']
    if not rows or rows[0]['optimizer_step'] != 0:
        raise ValueError('Stopping needs a complete fresh baseline')
    base, last = rows[0], rows[-1]
    for row in rows:
        for key in ('accuracy', 'completed_accuracy', 'mean_total_tokens', 'token_saving',
                    'truncation_rate', 'invalid_thinking_rate'):
            if not math.isfinite(row[key]):
                raise ValueError('Nonfinite monitoring data')
    def quality(r):
        return (r['truncation_rate'] <= base['truncation_rate'] + c['max_truncation_increase']
                and r['invalid_thinking_rate'] <= base['invalid_thinking_rate'] + c['max_invalid_increase'])
    def eligible(r):
        return (quality(r) and r['paired_question_accuracy_ci95_pp'][0] >= -c['accuracy_tolerance_pp']
                and r['completed_accuracy_ci95_pp'][0] >= -c['accuracy_tolerance_pp']
                and r['token_saving'] >= c['target_token_saving']
                and r['token_saving_ci95'][0] >= c['minimum_token_saving_lcb'])
    def promising(r):
        preserved = (r['accuracy_delta_pp'] >= -c['accuracy_tolerance_pp']
                     and r['completed_accuracy_delta_pp'] >= -c['accuracy_tolerance_pp'])
        return quality(r) and preserved and (
            r['token_saving'] >= c['promising_token_saving'] or
            (r['accuracy_delta_pp'] >= c['promising_accuracy_gain_pp'] and r['token_saving'] >= 0))
    def regressed(r):
        return (r['paired_question_accuracy_ci95_pp'][1] < -c['regression_margin_pp']
                or r['completed_accuracy_ci95_pp'][1] < -c['regression_margin_pp']
                or r['truncation_rate'] > base['truncation_rate'] + c['collapse_increase']
                or r['invalid_thinking_rate'] > base['invalid_thinking_rate'] + c['collapse_increase'])
    eligible_rows = [r for r in rows[1:] if eligible(r)]
    best = min(eligible_rows, key=lambda r: (r['mean_total_tokens'], r['optimizer_step']), default=None)
    result = {'stop': False, 'reason': 'collecting_evidence',
              'optimizer_step': last['optimizer_step'], 'rollout_batches': last['rollout_batches'],
              'best_eligible_step': best['optimizer_step'] if best else None,
              'scope': 'Operational development stopping; no final efficacy or mathematical convergence claim'}
    tail = [r for r in rows[1:] if r['rollout_batches'] >= c['regression_min_rollouts']]
    n = c['regression_patience']
    if len(tail) >= n and all(regressed(r) for r in tail[-n:]):
        return {**result, 'stop': True, 'reason': 'persistent_regression'}
    if last['rollout_batches'] < c['minimum_rollouts']:
        return result
    n = c['plateau_evaluations']
    window = rows[1:][-n:]
    if len(window) == n and all(eligible(r) for r in window):
        token_range = (max(r['mean_total_tokens'] for r in window) - min(r['mean_total_tokens'] for r in window)) / base['mean_total_tokens']
        accuracy_range = 100 * (max(r['accuracy'] for r in window) - min(r['accuracy'] for r in window))
        if token_range <= c['plateau_token_range'] and accuracy_range <= c['plateau_accuracy_range_pp']:
            return {**result, 'stop': True, 'reason': 'useful_plateau_needs_confirmation'}
    n = c['stagnation_patience']
    if len(rows) > n and not any(promising(r) for r in rows[-n:]):
        return {**result, 'stop': True, 'reason': 'no_useful_progress_within_window'}
    return {**result, 'reason': 'improving_or_not_yet_stable'}
