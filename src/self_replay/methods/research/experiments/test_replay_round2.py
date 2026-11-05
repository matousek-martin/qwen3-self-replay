"""Verify the registered grid preserves the prior replay control."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from experiments import replay_round2 as batch
from experiments.rl_training.train import make_parser


class ReplayRoundTwoTests(unittest.TestCase):
    def test_grid_has_three_matched_seeds_and_one_factor_changes(self):
        grid = batch.trials()
        self.assertEqual(len({t['name'] for t in grid}), 9)
        for seed in batch.SEEDS:
            by = {t['recipe']: t for t in grid if t['seed'] == seed}
            self.assertEqual(set(by), set(batch.RECIPES))
            self.assertEqual(by['control']['length_coefficient'], 0.1)
            self.assertEqual(by['control']['replay_weight'], 0.1)
            self.assertEqual(by['weaker-length']['length_coefficient'], 0.05)
            self.assertEqual(by['weaker-length']['replay_weight'], by['control']['replay_weight'])
            self.assertEqual(by['stronger-replay']['length_coefficient'], by['control']['length_coefficient'])
            self.assertEqual(by['stronger-replay']['replay_weight'], 0.2)

    def test_control_matches_archived_training_parameters(self):
        archive = next((batch.ROOT / 'lab/experiments/cloud').glob('study-rl-replay-*/output/training/rl-replay/config.json'))
        previous = json.loads(archive.read_text())
        # Historical snapshots predate the explicit admission option; their
        # trainer implemented all(successes). Do not modify archived configs.
        previous.setdefault('replay_admission', 'all_correct')
        trial = next(t for t in batch.trials() if t['name'] == 'control-seed17')
        command = batch.training_command(trial, Path('/data'), Path('/output'))
        actual = vars(make_parser().parse_args(command[3:]))
        for key, value in actual.items():
            if key not in ('data', 'out', 'references'):
                self.assertEqual(value, previous[key], key)

    def test_changed_trial_grid_is_rejected_before_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'plan.json'
            plan = {'trials': batch.trials(), 'max_parallel': 3,
                    'gpu': 'NVIDIA GeForce RTX 4090', 'total_campaign_ceiling_usd': 200}
            path.write_text(json.dumps(plan))
            batch.load_plan(path)
            plan['trials'][0]['replay_weight'] = 0.3
            path.write_text(json.dumps(plan))
            with self.assertRaises(ValueError):
                batch.load_plan(path)


if __name__ == '__main__':
    unittest.main()
