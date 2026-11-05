"""Pure-Python reward and normalization definitions, independent of GPU libraries."""

import math
import random
import statistics

VARIANTS = ("correctness", "fixed", "adaptive", "difficulty", "progressive", "replay")


def reasoning_is_complete(text):
    """Qwen's prompt supplies <think>; accept either raw or reconstructed output."""
    if text.count("</think>") != 1:
        return False
    reasoning, answer = text.split("</think>")
    reasoning = reasoning.strip()
    if reasoning.startswith("<think>"):
        reasoning = reasoning[len("<think>"):].strip()
    return bool(reasoning and answer.strip()) and "<think>" not in reasoning and "<think>" not in answer


def random_streams(seed):
    """Keep auxiliary replay draws independent of the common prompt schedule."""
    return random.Random(seed), random.Random(seed + 104729)


def rewards(successes, lengths, variant="correctness", coefficient=0.1, reference=1024,
            replay_reward="correctness"):
    """Our experimental rewards, not reproductions of paper-specific objectives.

    Correct answers always beat wrong answers in raw reward when coefficient < 1.
    All-wrong groups get zeros and never supply an adaptive target.
    """
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    if replay_reward not in ("correctness", "fixed"):
        raise ValueError("replay_reward must be correctness or fixed")
    if variant == "replay":
        variant = replay_reward
    if not successes or len(successes) != len(lengths):
        raise ValueError("Need equally sized, nonempty success and length groups")
    if not 0 <= coefficient < 1 or reference <= 0:
        raise ValueError("Require 0 <= coefficient < 1 and positive reference")
    if any(length <= 0 for length in lengths):
        raise ValueError("Generated lengths must be positive")
    correct_lengths = [n for ok, n in zip(successes, lengths) if ok]
    if variant in ("correctness", "progressive") or not correct_lengths:
        return [float(ok) for ok in successes]
    if variant == "adaptive":
        target = min(correct_lengths)
        penalties = [min(max(n / target - 1, 0), 1) for n in lengths]
    else:
        penalties = [min(n / reference, 1) for n in lengths]
    # A group's current success rate is a noisy difficulty estimate, not a label.
    if variant == "difficulty":
        coefficient *= sum(successes) / len(successes)
    return [float(ok) * (1 - coefficient * penalty) for ok, penalty in zip(successes, penalties)]


def advantages(values, scale=True):
    if len(values) < 2:
        raise ValueError("A relative group requires at least two samples")
    if any(not math.isfinite(x) for x in values):
        raise ValueError("Rewards must be finite")
    mean = statistics.mean(values)
    # Sample standard deviation matches the original group estimator.
    denominator = statistics.stdev(values) + 1e-4 if scale else 1.0
    return [(x - mean) / denominator for x in values]


def loss_denominator(optimizer, length, maximum, group_size):
    if optimizer not in ("grpo", "dr_grpo"):
        raise ValueError("optimizer must be grpo or dr_grpo")
    if min(length, maximum, group_size) <= 0 or length > maximum:
        raise ValueError("Invalid completion length or group size")
    return group_size * (length if optimizer == "grpo" else maximum)


def training_cap(step, max_steps, initial, final, variant):
    if initial <= 0 or final <= 0 or final > initial:
        raise ValueError("Require 0 < final <= initial training cap")
    if variant != "progressive" or max_steps <= 1:
        return initial
    fraction = min(max(step / (max_steps - 1), 0), 1)
    return max(final, round(initial + fraction * (final - initial)))
