import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('notify_teacher', Path(__file__).with_name('notify_teacher_campaigns.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CompletionTests(unittest.TestCase):
    def setUp(self):
        self.target = {'seeds': [17, 29, 43], 'arms': ['student', 'teacher'], 'job_prefix': 'full-v2-'}
        self.campaign = {'state': 'development_complete', 'trials': {
            str(s): {'state': 'complete', 'job': f'full-v2-{s}'} for s in self.target['seeds']}}
        self.results = {'complete': True, 'comparisons': [
            {'seed': s, 'arm': a} for s in self.target['seeds'] for a in self.target['arms']]}
        self.ledger = {'jobs': {f'full-v2-{s}': {'state': 'complete', 'backup_verified': True,
                                              'released_at': 123} for s in self.target['seeds']}}

    def ready(self):
        return module.completion_ready(self.target, self.campaign, self.results, self.ledger)

    def test_ready(self):
        self.assertTrue(self.ready())

    def test_preparation_cannot_notify(self):
        self.campaign = {'state': 'prepared_inputs', 'trials': {}}
        self.assertFalse(self.ready())

    def test_partial_results_cannot_notify(self):
        self.results['comparisons'].pop()
        self.assertFalse(self.ready())

    def test_unverified_or_unreleased_cannot_notify(self):
        for field in ('released_at', 'backup_verified'):
            with self.subTest(field=field):
                old = self.ledger['jobs']['full-v2-17'][field]
                self.ledger['jobs']['full-v2-17'][field] = None
                self.assertFalse(self.ready())
                self.ledger['jobs']['full-v2-17'][field] = old

    def test_orphan_attempt_cannot_notify(self):
        self.ledger['jobs']['full-v2-capacity-retry'] = {'state': 'running'}
        self.assertFalse(self.ready())

    def test_wrong_arm_or_duplicate_cannot_notify(self):
        self.results['comparisons'][-1] = self.results['comparisons'][0]
        self.assertFalse(self.ready())

    def test_uncertain_delivery_is_not_retried(self):
        calls = []
        def sender(message):
            calls.append(message)
            raise TimeoutError('Simulated uncertain delivery')
        with tempfile.TemporaryDirectory() as directory:
            alerts = module.Alerts(Path(directory) / 'delivery.sqlite3', 'Test', sender=sender)
            self.assertFalse(alerts.emit('done', 'Completion', 'Completed'))
            self.assertFalse(alerts.emit('done', 'Completion', 'Completed'))
            self.assertEqual(len(calls), 1)
            self.assertEqual(alerts.db.execute('SELECT delivery FROM incidents').fetchone()[0], 'uncertain')


if __name__ == '__main__':
    unittest.main()
