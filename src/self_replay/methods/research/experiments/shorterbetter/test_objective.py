import ast
import math
from pathlib import Path
import random
import unittest

from .objective import sol_rewards, grouped_rewards
from .upstream import load_verifier

SOURCES = Path(__file__).resolve().parents[2] / 'lab/experiments/shorterbetter-preparation-20261026/sources'


class RewardTests(unittest.TestCase):
    def test_mixed_group_penalizes_short_wrong_and_long_correct(self):
        values, target = sol_rewards([False, True, True], [10, 100, 1000])
        self.assertEqual(target, 100)
        self.assertEqual(values, [-0.09, 2.0, 1.1])

    def test_all_wrong_uses_mean_and_provides_nonzero_signal(self):
        values, target = sol_rewards([False] * 3, [10, 20, 90])
        self.assertEqual(target, 40)
        self.assertEqual(values, [-0.03, -0.02, -0.05])

    def test_equal_lengths_zero_variance(self):
        self.assertEqual(sol_rewards([True] * 8, [100] * 8)[0], [2.0] * 8)
        self.assertEqual(sol_rewards([False] * 8, [100] * 8)[0], [0.0] * 8)

    def test_control_keeps_correctness_scale(self):
        self.assertEqual(sol_rewards([True, False], [900, 10], length_beta=0)[0], [2.0, 0.0])

    def test_not_clipped_and_correct_may_have_negative_reward(self):
        self.assertLess(sol_rewards([True, True], [1, 6000])[0][1], 0)

    def test_group_order_preserved(self):
        values, targets = grouped_rewards(['a', 'b', 'a', 'b'],
                                         [True, True, False, True], [10, 20, 30, 40], 2)
        self.assertEqual(values, [2.0, 2.0, -0.02, 1.98])
        self.assertEqual(targets, {'a': 10, 'b': 20})

    def test_incomplete_groups_fail(self):
        with self.assertRaises(ValueError):
            grouped_rewards(['a'] * 7, [True] * 7, [10] * 7)

    def test_invalid_inputs(self):
        for correct, lengths in [([], []), ([1], [10]), ([True], [0]), ([True], [1.5])]:
            with self.assertRaises(ValueError):
                sol_rewards(correct, lengths)
        for value in [float('nan'), float('inf'), -1]:
            with self.assertRaises(ValueError):
                sol_rewards([True], [10], length_beta=value)

    def test_parity_with_pinned_authors_scalar_function(self):
        tree = ast.parse((SOURCES / 'shorterbetter_reward.py').read_text())
        function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                        and n.name == 'sb_compute_score')
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), 'upstream', 'exec'), namespace)
        oracle = namespace['sb_compute_score']
        rng = random.Random(83)
        for _ in range(500):
            correct = [bool(rng.getrandbits(1)) for _ in range(8)]
            lengths = [rng.randrange(1, 6001) for _ in range(8)]
            values, target = sol_rewards(correct, lengths)
            for ok, length, actual in zip(correct, lengths, values):
                self.assertEqual(actual, oracle(target, length, ok))

    def test_upstream_verifier_does_not_accept_arbitrary_text(self):
        verify = load_verifier(SOURCES)
        self.assertTrue(verify(r'Answer: \boxed{\frac{1}{2}}', r'\frac{1}{2}'))
        self.assertFalse(verify(r'Answer: \boxed{3}', '2'))
        self.assertFalse(verify('The answer is 2', '2'))
        self.assertFalse(verify(r'\boxed{2', '2'))


if __name__ == '__main__':
    unittest.main()
