"""Equation (1), ShorterBetter arXiv:2504.21370v4.

Lengths are generated token counts including EOS, never retokenized text.
Wrong and truncated rollouts remain in their original groups. Correctness is
supplied by the pinned verifier; formatting diagnostics do not alter reward.
"""
from collections import defaultdict
import math


def sol_rewards(correct, lengths, alpha=2.0, length_beta=0.001):
    correct, lengths = list(correct), list(lengths)
    if not correct or len(correct) != len(lengths):
        raise ValueError('Expected nonempty, equally sized groups')
    if any(type(x) is not bool for x in correct):
        raise ValueError('Correctness must be Boolean')
    if any(type(x) is not int or x <= 0 for x in lengths):
        raise ValueError('Lengths must be positive integer token counts')
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError('alpha must be finite and positive')
    if not math.isfinite(length_beta) or length_beta < 0:
        raise ValueError('length_beta must be finite and nonnegative')
    successes = [n for ok, n in zip(correct, lengths) if ok]
    target = min(successes) if successes else sum(lengths) / len(lengths)
    return [alpha * int(ok) - length_beta * abs(n - target)
            for ok, n in zip(correct, lengths)], target


def grouped_rewards(ids, correct, lengths, group_size=8, **kwargs):
    if not len(ids) == len(correct) == len(lengths) or not ids:
        raise ValueError('Mismatched rollout columns')
    groups = defaultdict(list)
    for i, identifier in enumerate(ids):
        groups[identifier].append(i)
    out = [None] * len(ids)
    targets = {}
    for identifier, positions in groups.items():
        if len(positions) != group_size:
            raise ValueError(f'Incomplete/duplicated group {identifier}: {len(positions)}')
        rewards, target = sol_rewards([correct[i] for i in positions],
                                      [lengths[i] for i in positions], **kwargs)
        targets[identifier] = target
        for i, value in zip(positions, rewards):
            out[i] = value
    return out, targets
