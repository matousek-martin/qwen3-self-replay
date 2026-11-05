import unittest
from experiments.summarize import paired_metrics


class SummaryTests(unittest.TestCase):
    def test_identical_answers_and_half_tokens(self):
        prompts = [{'id': 'x', 'source': 'math', 'answer': '2', 'task': 'math'}]
        base = [{'prompt_id': 'x', 'sample_id': i, 'completion': '<think>1+1=2</think>\nFinal answer: 2',
                 'truncated': False, 'total_tokens': 100} for i in range(2)]
        candidate = [dict(r, total_tokens=50) for r in base]
        metrics = paired_metrics(prompts, base, candidate, draws=20)
        self.assertEqual(metrics['accuracy_delta_95ci'], [0, 0])
        self.assertEqual(metrics['tokens_saved_95ci'], [50, 50])

    def test_unpaired_samples_rejected(self):
        with self.assertRaises(ValueError):
            paired_metrics([{'id': 'x'}], [], [], draws=20)


if __name__ == '__main__':
    unittest.main()
