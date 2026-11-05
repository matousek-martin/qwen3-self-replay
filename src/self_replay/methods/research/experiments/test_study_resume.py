import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
import study_evaluation_campaign as campaign


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.study, self.cloud = self.root / 'study', self.root / 'cloud'
        self.study.mkdir()
        self.cloud.mkdir()
        self.data = self.root / 'data'
        self.data.mkdir()
        self.patchers = [patch.object(campaign, 'STUDY', self.study), patch.object(campaign, 'STATE', self.cloud),
                         patch.object(campaign, 'DATA', self.data)]
        for patcher in self.patchers:
            patcher.start()

    def tearDown(self):
        for patcher in self.patchers:
            patcher.stop()
        self.temp.cleanup()

    def seed(self, state='needs_review', released=True, evaluated=None):
        value = {'state': state, 'batches': [{'job': 'old', 'success': False}],
                 'evaluated': evaluated or [], 'started_at': 'old'}
        (self.study / 'evaluation-campaign.json').write_text(json.dumps(value))
        (self.cloud / 'ledger.json').write_text(json.dumps({'jobs': {'old': {'released_at': 1 if released else None}}}))

    def test_active_campaign_is_not_resumed(self):
        self.seed(state='running')
        with patch.object(sys, 'argv', ['driver', '--resume']), patch.object(campaign, 'evaluate') as evaluate:
            with self.assertRaises(SystemExit):
                campaign.main()
            evaluate.assert_not_called()

    def test_missing_study_result_is_failure_even_after_successful_shell_exit(self):
        name = 'study-eval-00-test'
        (self.cloud / 'ledger.json').write_text(json.dumps({'jobs': {
            name: {'state': 'complete', 'backup_verified': True, 'released_at': 1}}}))
        with patch.object(campaign.subprocess, 'run', return_value=SimpleNamespace(returncode=0)), \
                patch.object(campaign, 'analyze') as analyze:
            result = campaign.evaluate(['sft-short'], 'test', 0)
        self.assertFalse(result['success'])
        self.assertTrue(result['cleanup_confirmed'])
        self.assertIn('study-result.json', result['error'])
        analyze.assert_not_called()

    def test_unreleased_rental_is_not_duplicated(self):
        self.seed(released=False)
        with patch.object(sys, 'argv', ['driver', '--resume']), patch.object(campaign, 'evaluate') as evaluate:
            with self.assertRaises(SystemExit):
                campaign.main()
            evaluate.assert_not_called()

    def test_resume_keeps_history_and_skips_completed_candidates(self):
        done = campaign.INITIAL[:-1] + campaign.RL
        self.seed(evaluated=done)
        with patch.object(sys, 'argv', ['driver', '--resume']), patch.object(campaign, 'evaluate', return_value={'job': 'new', 'success': True}) as evaluate:
            campaign.main()
        self.assertEqual(evaluate.call_args.args[0], [campaign.INITIAL[-1]])
        self.assertEqual(evaluate.call_args.args[2], 1)
        result = json.loads((self.study / 'evaluation-campaign.json').read_text())
        self.assertEqual(result['batches'][0]['job'], 'old')
        self.assertEqual(result['state'], 'development_complete')
        self.assertEqual(set(result['evaluated']), set(campaign.INITIAL + campaign.RL))


if __name__ == '__main__':
    unittest.main()
