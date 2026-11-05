"""Group-local RL rewards for the replay ablation; no replay state or I/O.

The caller supplies training-verifier successes (including format/truncation
checks) and generated total-token counts for ONE on-policy prompt group.
Replay admission, RNG, buffer contents, weight and imitation loss stay owned
by the trainer. This is not a validation-accuracy controller.
"""

import math

from .objectives import rewards

REPLAY_REWARDS = ("correctness", "fixed", "adaptive")
REWARD_DEFINITION = "existing-group-adaptive-v1"


def replay_rewards(successes, lengths, replay_reward="correctness",
                   coefficient=0.1, reference=1024):
    """Return existing reward semantics without routing through variant=replay.

    fixed:    ok * (1 - coefficient * min(length / reference, 1))
    adaptive: ok * (1 - coefficient * clip(length / target - 1, 0, 1))
    target is the shortest correct completion in this group only. All-wrong
    groups return zeros without constructing a target. A single correct answer
    receives 1 in adaptive mode. There is no memory between calls.

    Delegate to the canonical objectives implementation to prevent formula
    drift. Reject malformed/nonfinite inputs before they can reach autograd.
    The historical correctness default and fixed values remain unchanged for
    valid trainer inputs. Reference stays required/validated in adaptive mode
    for a shared configuration, but does not determine its reward.
    """
    if replay_reward not in REPLAY_REWARDS:
        raise ValueError(f"Unknown replay reward: {replay_reward}")
    successes, lengths = tuple(successes), tuple(lengths)
    if not successes or len(successes) != len(lengths):
        raise ValueError("Need equally sized, nonempty success and length groups")
    if any(type(ok) is not bool for ok in successes):
        raise ValueError("Successes must be verified booleans")
    if any(isinstance(n, bool) or not isinstance(n, (int, float))
           or not math.isfinite(n) or n <= 0 for n in lengths):
        raise ValueError("Generated lengths must be finite and positive")
    if (not math.isfinite(coefficient) or not 0 <= coefficient < 1
            or not math.isfinite(reference) or reference <= 0):
        raise ValueError("Require finite 0 <= coefficient < 1 and positive reference")
    return rewards(successes, lengths, variant=replay_reward,
                   coefficient=coefficient, reference=reference)
