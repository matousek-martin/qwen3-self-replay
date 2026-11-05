"""Protect full-cohort reporting from subset and aggregation mistakes."""
import json
from pathlib import Path
import tempfile
import unittest

from experiments.full_replay_campaign import benchmark_metrics, final_metrics
from experiments.study_eval import check_lock, sha256


class FullReplayTests(unittest.TestCase):
    def test_bfcl_is_pooled_before_equal_domain_average(self):
        def value(n, accuracy):
            return dict(n=n, accuracy=accuracy, mean_output_tokens=100,
                        mean_reasoning_tokens=80, truncation_rate=0)
        metrics = benchmark_metrics({'results': {
            'bfcl_simple_python': value(400, 1), 'bfcl_multiple': value(200, 0),
            'bfcl_irrelevance': value(240, 0), 'mbpp': value(378, 1),
            'svamp': value(1000, 0), 'ifeval': value(541, 0)}})
        self.assertAlmostEqual(metrics['bfcl']['accuracy'], 400 / 840)
        self.assertAlmostEqual(metrics['macro']['accuracy'], (400 / 840 + 1) / 4)

    def test_subset_comparison_cannot_be_published_as_full(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / 'comparison.json').write_text(json.dumps({
                'ungraded_benchmarks': {},
                'full_evaluation': {'base': ['study subset'], 'candidate': ['study subset']}}))
            with self.assertRaisesRegex(ValueError, 'Incomplete official comparison'):
                final_metrics(output)

    def test_changed_adapter_rejected_by_original_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'lock.json'
            lock = {'schema_version': 1, 'candidate': 'rl-replay', 'primary_candidate': 'rl-replay',
                    'adapter_files_sha256': {'weights': 'original'},
                    'split_manifest_sha256': 'frozen', 'locked_at': '2026-10-24'}
            path.write_text(json.dumps(lock))
            with self.assertRaisesRegex(ValueError, 'changed after locking'):
                check_lock(path, 'rl-replay', {'weights': 'modified'}, 'frozen')


if __name__ == '__main__':
    unittest.main()
