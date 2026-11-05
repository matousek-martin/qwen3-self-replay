"""CPU reward tests; optional torch tests differentiate synthetic logps only."""

import itertools
import math
import random
import unittest

from .adaptive_replay import REPLAY_REWARDS, replay_rewards
from .objectives import advantages, loss_denominator, rewards

try:
    import torch
except ImportError:
    torch = None


class AdaptiveReplayTests(unittest.TestCase):
    def test_all_wrong_has_no_target_even_after_correct_group(self):
        replay_rewards([True, True], [10, 20], "adaptive")
        self.assertEqual(replay_rewards([False] * 4, [1, 10, 100, 2048], "adaptive"), [0] * 4)

    def test_one_correct_is_target_wrong_shortest_ignored(self):
        self.assertEqual(replay_rewards([False, True, False], [1, 500, 100], "adaptive"), [0, 1, 0])

    def test_equal_lengths_and_zero_variance(self):
        for mode in REPLAY_REWARDS:
            values = replay_rewards([True] * 4, [100] * 4, mode)
            for scale in (False, True):
                self.assertEqual(advantages(values, scale), [0] * 4)
        self.assertEqual(replay_rewards([True] * 4, [100] * 4, "adaptive"), [1] * 4)

    def test_penalty_boundaries(self):
        values = replay_rewards([True] * 6, [100, 150, 199, 200, 201, 2048], "adaptive", 0.2)
        for actual, expected in zip(values, [1, 0.9, 0.802, 0.8, 0.8, 0.8]):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(replay_rewards([True] * 3, [50, 100, 200], "fixed", 0.2, 100), [0.9, 0.8, 0.8])

    def test_zero_coefficient_and_correctness_default(self):
        for mode in REPLAY_REWARDS:
            self.assertEqual(replay_rewards([True, False, True], [50, 1, 2048], mode, 0), [1, 0, 1])
        self.assertEqual(replay_rewards([True, False], [2048, 1]), [1, 0])

    def test_exact_parity_with_existing_rewards(self):
        rng = random.Random(7041)
        for successes in itertools.product((False, True), repeat=4):
            for _ in range(20):
                lengths = [rng.randint(1, 2048) for _ in successes]
                coefficient, reference = rng.choice([0, 0.05, 0.1, 0.999]), rng.choice([100, 1024, 4096])
                for mode in REPLAY_REWARDS:
                    actual = replay_rewards(successes, lengths, mode, coefficient, reference)
                    expected = rewards(successes, lengths, mode, coefficient, reference)
                    self.assertEqual(actual, expected)
                    if mode != "adaptive":
                        self.assertEqual(actual, rewards(successes, lengths, "replay", coefficient, reference, mode))

    def test_stateless_and_reference_independent_adaptive(self):
        successes, lengths = [True, False, True], [100, 1, 150]
        before = (successes[:], lengths[:], random.getstate())
        expected = replay_rewards(successes, lengths, "adaptive", reference=1)
        replay_rewards([True, True], [1000, 2048], "adaptive")
        self.assertEqual(expected, replay_rewards(successes, lengths, "adaptive", reference=9999))
        self.assertEqual(before, (successes, lengths, random.getstate()))

    def test_correct_answers_always_beat_wrong_and_advantages_finite(self):
        for mode in REPLAY_REWARDS:
            values = replay_rewards([False, True, True, True], [1, 10, 20, 2048], mode, 0.999999)
            self.assertTrue(all(value > values[0] for value in values[1:]))
            for scale in (False, True):
                adv = advantages(values, scale)
                self.assertTrue(all(math.isfinite(value) for value in adv))
                self.assertLess(adv[0], 0)

    def test_invalid_groups_and_nonfinite_parameters(self):
        for successes, lengths in (([], []), ([True], [1, 2]), ([1], [1]), ([True], [0]),
                                   ([True], [-1]), ([True], [True]), ([True], [float('nan')]),
                                   ([False], [float('inf')])):
            with self.subTest(successes=successes, lengths=lengths), self.assertRaises(ValueError):
                replay_rewards(successes, lengths, "adaptive")
        for name, values in (("coefficient", [-1, 1, float('nan'), float('inf')]),
                             ("reference", [0, -1, float('nan'), float('inf')])):
            for value in values:
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    replay_rewards([False, False], [1, 2], "adaptive", **{name: value})
        with self.assertRaises(ValueError):
            replay_rewards([True, False], [1, 2], "validation-controller")


@unittest.skipIf(torch is None, "CPU torch required for synthetic gradient checks")
class AdaptiveReplayGradientTests(unittest.TestCase):
    def test_policy_and_replay_gradient_matches_analytic_derivative(self):
        # No model, optimizer step, downloads, GPU, or training data involved.
        for mode in ("fixed", "adaptive"):
            for successes in ([False] * 4, [True, False, False, False], [True] * 4):
                for lengths in ([4] * 4, [2, 3, 4, 8]):
                    with self.subTest(mode=mode, successes=successes, lengths=lengths):
                        advs = advantages(replay_rewards(successes, lengths, mode), scale=False)
                        policy = [torch.full((n,), -0.7, dtype=torch.float64, requires_grad=True) for n in lengths]
                        replay = torch.tensor([-0.3, -0.8, -1.2], dtype=torch.float64, requires_grad=True)
                        loss = -0.1 * replay.mean()
                        for lp, adv in zip(policy, advs):
                            loss = loss - (lp - lp.detach()).exp().sum() * adv / loss_denominator("dr_grpo", len(lp), 8, 4)
                        loss.backward()
                        self.assertTrue(torch.isfinite(loss))
                        for lp, adv in zip(policy, advs):
                            self.assertTrue(torch.isfinite(lp.grad).all())
                            torch.testing.assert_close(lp.grad, torch.full_like(lp, -adv / 32))
                        torch.testing.assert_close(replay.grad, torch.full_like(replay, -0.1 / 3))


if __name__ == "__main__":
    unittest.main()
