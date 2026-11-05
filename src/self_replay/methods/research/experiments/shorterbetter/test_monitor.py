import json
from pathlib import Path
import tempfile
import unittest

from .monitor import summarize
from .config import training_arguments

ROOT = Path(__file__).resolve().parents[2]


class DiagnosticTests(unittest.TestCase):
    def test_pilot_has_expected_budget_and_cadence(self):
        protocol = json.loads((ROOT / 'studies/shorterbetter/pilot-protocol.json').read_text())
        config = training_arguments(protocol, 'shorterbetter', 17, '/tmp/sb-pilot-test')
        self.assertEqual(config['max_steps'], 80)
        self.assertEqual(config['save_steps'], 20)
        self.assertEqual(config['eval_steps'], 20)
        self.assertEqual(config['generation_batch_size'], 512)
        self.assertEqual(protocol['budget']['proposed_total_pilot_cap_usd'], 20)
        with self.assertRaises(ValueError):
            training_arguments(protocol, 'correctness-control', 17, '/tmp/unregistered')

    def fixture(self, path, duplicate=False):
        # Deliberately synthetic test data, never published as measured results.
        rows = []
        for step in (0, 20):
            for identifier in ('a', 'b'):
                for _ in range(8):
                    rows.append({'optimizer_step': step, 'prompt_id': identifier,
                        'split': 'development', 'correct': True, 'total_tokens': 100 if step == 0 else 20,
                        'truncated': step == 20, 'closed_thinking': True})
        if duplicate:
            rows.append(rows[0])
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))

    def test_short_truncated_answers_do_not_look_like_preserved_completion_accuracy(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'rollouts.jsonl'
            self.fixture(path)
            report = summarize(path)
            last = report['checkpoints'][-1]
            self.assertEqual(last['accuracy_delta_pp'], 0)
            self.assertEqual(last['completed_accuracy_delta_pp'], -100)
            self.assertAlmostEqual(last['token_saving'], 0.8)
            self.assertEqual(last['truncation_rate'], 1)

    def test_duplicate_validation_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'rollouts.jsonl'
            self.fixture(path, duplicate=True)
            with self.assertRaises(ValueError):
                summarize(path)

    def test_renderer_only_creates_file_from_supplied_report(self):
        try:
            import matplotlib
        except ImportError:
            self.skipTest('Run also in the isolated preparation environment with matplotlib')
        from .plot import render
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'rollouts.jsonl'
            self.fixture(path)
            destination = render(summarize(path), Path(tmp) / 'diagnostic.png')
            self.assertTrue(destination.read_bytes().startswith(b'\x89PNG'))


if __name__ == '__main__':
    unittest.main()
