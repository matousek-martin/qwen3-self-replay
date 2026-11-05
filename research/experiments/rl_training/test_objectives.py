import json
import tempfile
import unittest
from pathlib import Path

from .objectives import advantages, loss_denominator, random_streams, reasoning_is_complete, rewards, training_cap
from .train import load_rows, verifier


class RewardTests(unittest.TestCase):
    def test_replay_fixed_reward_matches_fixed_control(self):
        for successes in ([True, False, True, True], [True] * 4, [False] * 4):
            for reference in (300, 1200):
                lengths = [100, 300, 700, 2048]
                expected = rewards(successes, lengths, "fixed", 0.1, reference)
                replay = rewards(successes, lengths, "replay", 0.1, reference, replay_reward="fixed")
                self.assertEqual(replay, expected)
        self.assertEqual(rewards([True, False], [100, 200], "replay"), [1, 0])

    def test_replay_draws_do_not_change_prompt_order_across_epochs(self):
        control_rng, _ = random_streams(17)
        prompt_rng, replay_rng = random_streams(17)
        control, replay = list(range(512)), list(range(512))
        for epoch in range(3):
            control_rng.shuffle(control)
            prompt_rng.shuffle(replay)
            self.assertEqual(control, replay)
            for _ in range(512):
                replay_rng.choice([1, 2, 3])

    def test_wrong_short_answer_does_not_become_target(self):
        self.assertEqual(rewards([False, True, True], [1, 100, 200], "adaptive"), [0, 1, 0.9])

    def test_all_wrong_and_equal_groups_have_zero_advantage(self):
        for variant in ("correctness", "fixed", "adaptive", "difficulty", "progressive", "replay"):
            self.assertEqual(rewards([False, False], [1, 1000], variant), [0, 0])
        self.assertEqual(advantages([0, 0]), [0, 0])
        self.assertEqual(advantages([1, 1], False), [0, 0])

    def test_wrong_answers_keep_negative_relative_signal(self):
        shaped = rewards([True, False], [2000, 1], "fixed", 0.2, 100)
        self.assertEqual(shaped, [0.8, 0])
        self.assertGreater(advantages(shaped)[0], 0)
        self.assertLess(advantages(shaped)[1], 0)

    def test_difficulty_weakens_pressure_when_success_sparse(self):
        fixed = rewards([True, False, False, False], [512] * 4, "fixed", reference=1024)
        adapted = rewards([True, False, False, False], [512] * 4, "difficulty", reference=1024)
        self.assertGreater(adapted[0], fixed[0])

    def test_dr_denominator_is_length_independent(self):
        self.assertEqual(loss_denominator("dr_grpo", 100, 2048, 4), loss_denominator("dr_grpo", 1000, 2048, 4))
        self.assertNotEqual(loss_denominator("grpo", 100, 2048, 4), loss_denominator("grpo", 1000, 2048, 4))

    def test_progressive_caps_only_affect_training_variant(self):
        self.assertEqual(training_cap(9, 10, 2048, 512, "progressive"), 512)
        self.assertEqual(training_cap(9, 10, 2048, 512, "fixed"), 2048)

    def test_nonempty_closed_reasoning_required(self):
        self.assertTrue(reasoning_is_complete("Compute 1+1.\n</think>\nFinal answer: 2"))
        self.assertTrue(reasoning_is_complete("<think>Compute 1+1.</think>\nFinal answer: 2"))
        for text in ("Final answer: 2", "<think></think>\nFinal answer: 2", "<think>unfinished", "a</think>b</think>c"):
            self.assertFalse(reasoning_is_complete(text))

    def test_uses_actual_contest_training_verifier(self):
        check = verifier()
        self.assertTrue(check("<think>one plus one</think>\nFinal answer: 2", "2", "math"))
        self.assertFalse(check("<think>one plus one</think>\nFinal answer: 3", "2", "math"))

    def test_validation_rows_cannot_enter_training(self):
        row = {"id": "a", "prompt": [{"role": "user", "content": "one plus one"}], "answer": "2", "task": "math", "source": "gsm8k", "split": "val"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            path.write_text(json.dumps(row) + "\n")
            with self.assertRaises(ValueError):
                load_rows(path)


if __name__ == "__main__":
    unittest.main()
