import json
from pathlib import Path
import tempfile
import unittest

from experiments.shorterbetter.completion_control import seed_continuation


class ContinuationTests(unittest.TestCase):
    def test_preserves_history_and_excludes_discarded_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old, new = root / 'old', root / 'new'
            checkpoint = old / 'training/checkpoint-20'
            checkpoint.mkdir(parents=True)
            (checkpoint / 'trainer_state.json').write_text(json.dumps({'global_step': 20}))
            for filename in ['optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'model.safetensors']:
                (checkpoint / filename).write_bytes(b'SYNTHETIC TEST FIXTURE')
            rows = [{'optimizer_step': step, 'split': split} for step, split in
                    [(0, 'development'), (0, 'train'), (16, 'train'), (20, 'development'), (20, 'train')]]
            source = old / 'training/rollouts.jsonl'
            source.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            original = source.read_bytes()
            (old / 'training/budget-stop.jsonl').write_text('{"optimizer_step":21}\n')
            (old / 'training/steps.jsonl').write_text('{"optimizer_step":20}\n{"optimizer_step":21}\n')
            seed_continuation(old, new, 20)
            seed_continuation(old, new, 20)  # interrupted preparation can be repeated
            self.assertEqual(source.read_bytes(), original)
            actual = [json.loads(line) for line in (new / 'training/rollouts.jsonl').read_text().splitlines()]
            self.assertEqual(actual, rows[:-1])
            self.assertEqual((new / 'training/steps.jsonl').read_text(), '{"optimizer_step":20}\n')
            self.assertFalse((new / 'training/budget-stop.jsonl').exists())
            self.assertEqual((new / 'training/checkpoint-20/optimizer.pt').read_bytes(), b'SYNTHETIC TEST FIXTURE')

    def test_missing_optimizer_refuses_continuation(self):
        with tempfile.TemporaryDirectory() as tmp:
            old = Path(tmp) / 'old'
            checkpoint = old / 'training/checkpoint-20'
            checkpoint.mkdir(parents=True)
            (checkpoint / 'trainer_state.json').write_text('{"global_step":20}')
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                seed_continuation(old, Path(tmp) / 'new', 20)


if __name__ == '__main__':
    unittest.main()
