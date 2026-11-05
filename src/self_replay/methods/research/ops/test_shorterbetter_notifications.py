"""Notification rendering without sending Telegram messages."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ops.notify_shorterbetter_scaled import format_notification, send_html, notices
from project_alerts import Alerts, format_message


class NotificationFormatTests(unittest.TestCase):
    def test_validation_only_suppresses_baseline_and_progress_but_keeps_alerts(self):
        with tempfile.TemporaryDirectory() as t:
            out=Path(t)/'output';(out/'training').mkdir(parents=True)
            (out/'phase.json').write_text(json.dumps({'phase':'main_training'}))
            (out/'training/progress.json').write_text(json.dumps({'rollout_batches':10}))
            curve=out/'training/learning-curve.json'
            baseline={'optimizer_step':0,'rollout_batches':0}
            curve.write_text(json.dumps({'checkpoints':[baseline]}))
            job={'state':'running','created_at':0}
            # Even ten finished training iterations do not constitute a validation.
            self.assertEqual(notices('test',job,out,now=8000,validation_only=True),[])
            curve.write_text(json.dumps({'checkpoints':[baseline,{'optimizer_step':40,'rollout_batches':10}]}))
            events=notices('test',job,out,now=8000,validation_only=True)
            self.assertEqual([e[0] for e in events],['eval-40'])
            (out/'training/failure-rank-000.json').write_text(json.dumps({'category':'numerical_failure'}))
            job.update(last_account_credit=20,incremental_funding={'warning_credit_usd':30,'stop_credit_usd':12})
            events=notices('test',job,out,now=8000,validation_only=True)
            self.assertEqual({e[0] for e in events},{'eval-40','failure','credit-low'})

    def test_live_technical_progress_does_not_count_as_main_training(self):
        with tempfile.TemporaryDirectory() as t:
            out=Path(t)/'output';out.mkdir()
            (out/'training').mkdir()
            (out/'training/progress.json').write_text(json.dumps({'rollout_batches':99}))
            (out/'phase.json').write_text(json.dumps({'phase':'deterministic_resume_reference'}))
            snapshot={'observed_at':100,'files':{'phase.json':{'phase':'gpu_resume_verification'},
                'verification-resumed/progress.json':{'optimizer_step':7},'training/progress.json':None}}
            (out.parent/'notification-live.json').write_text(json.dumps(snapshot))
            p={'execution':{'multidomain_full_training':True},'training':{'rollout_batches':300}}
            message=format_notification('Test','Progress','ignored',{'created_at':0},out,p,now=120)
            self.assertIn('Fáze: test obnovy checkpointu před tréninkem',message)
            self.assertIn('Technická kontrola: 7 / 8 aktualizací',message)
            self.assertIn('Iterations (trénink): 0 / 300',message)
            self.assertNotIn('Progress:',message)
            stale=format_notification('Test','Progress','ignored',{'created_at':0},out,p,now=1000)
            self.assertIn('živé spojení není aktuální',stale)

    def test_main_phase_drops_technical_counter(self):
        with tempfile.TemporaryDirectory() as t:
            out=Path(t)/'output';(out/'training').mkdir(parents=True)
            (out/'phase.json').write_text(json.dumps({'phase':'main_training'}))
            (out/'training/progress.json').write_text(json.dumps({'rollout_batches':1.25}))
            p={'execution':{'multidomain_full_training':True},'training':{'rollout_batches':300}}
            message=format_notification('Test','Progress','ignored',{'created_at':0},out,p,now=120)
            self.assertIn('Fáze: hlavní trénink',message)
            self.assertIn('Iterations (trénink): 1.25 / 300',message)
            self.assertNotIn('Technická kontrola:',message)

    def test_partial_training_progress_is_distinct_from_last_evaluation(self):
        with tempfile.TemporaryDirectory() as t:
            out = Path(t); (out/'training').mkdir()
            a = dict(rollout_batches=0, optimizer_step=0, accuracy=.5, mean_total_tokens=1000, truncation_rate=.1)
            b = dict(a, rollout_batches=1, optimizer_step=4, accuracy=.625, mean_total_tokens=800)
            curve = {'checkpoints': [a, b], 'domains': {name: {'checkpoints': [a, b]}
                for name in ('math', 'coding', 'tools', 'instructions')}}
            (out/'training/learning-curve.json').write_text(json.dumps(curve))
            (out/'training/progress.json').write_text(json.dumps({'rollout_batches': 1.5}))
            message = format_notification('ThinkingCap <test>', 'Progress', 'Running & healthy',
                {'created_at': 100, 'gpu_count': 4, 'gpu': 'NVIDIA A100', 'provider': 'vast'}, out,
                {'training': {'rollout_batches': 4}}, now=3761)
            self.assertIn('<b>Project: ThinkingCap &lt;test&gt;</b>', message)
            self.assertNotIn('Progress:', message)
            self.assertNotIn('Running & healthy', message)
            self.assertIn('Iterations: 1.5 / 4\nTime elapsed: 01:01:01\nTimestamp:', message)
            self.assertIn('Metriky z validace: iterace 1', message)
            self.assertIn('Accuracy: 50.0% → 62.5% (+12.5 p. b.)\nTokeny / odpověď: 1000 → 800', message)
            self.assertEqual(message.count('Accuracy:'), 5)
            self.assertNotIn('Malý interní panel', message)
            self.assertNotIn('Problem:', message)
            self.assertLess(len(message.encode('utf-16-le'))//2, 4096)
            events = notices('test', {'created_at': 100}, out, now=3761)
            self.assertEqual([e[0] for e in events], ['eval-4'])

    def test_final_elapsed_stops_at_release_and_missing_total_is_unknown(self):
        with tempfile.TemporaryDirectory() as t:
            message = format_notification('Test', 'Results', 'Done',
                {'created_at': 100, 'released_at': 160}, Path(t), now=10000)
            self.assertIn('Summary: Done', message)
            self.assertIn('Time elapsed: 00:01:00', message)
            self.assertIn('Iterations: 0 / ?', message)

    def test_shared_default_format_and_dedup_remain_unchanged(self):
        with tempfile.TemporaryDirectory() as t:
            sent = []
            alerts = Alerts(Path(t)/'delivery.db', 'Other project', sent.append)
            self.assertTrue(alerts.emit('one', 'Technical', 'An issue'))
            self.assertTrue(sent[0].startswith('Project: Other project\nType: Technical\nProblem: An issue\n'))
            self.assertFalse(alerts.emit('one', 'Technical', 'An issue'))
            alerts.formatter = lambda *args: '<b>Custom message</b>'
            self.assertTrue(alerts.emit('two', 'Progress', 'Update'))
            self.assertEqual(sent[-1], '<b>Custom message</b>')

    def test_html_sender_uses_configured_helper(self):
        with patch('ops.notify_shorterbetter_scaled.subprocess.run') as run:
            run.return_value.returncode = 0
            send_html('<b>Project: test</b>')
            self.assertEqual(run.call_args.args[0], ['/usr/local/bin/codex-telegram', 'send', '<b>Project: test</b>', '--parse-mode', 'HTML'])


if __name__ == '__main__':
    unittest.main()
