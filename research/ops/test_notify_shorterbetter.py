import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ops import notify_shorterbetter as notify


class NotificationTests(unittest.TestCase):
    def test_only_main_verified_results_are_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            (output / 'training').mkdir()
            rows = [dict(optimizer_step=step, questions=128, accuracy=acc,
                         mean_total_tokens=tokens, truncation_rate=.1)
                    for step, acc, tokens in [(0, .3, 5000), (20, .35, 4000)]]
            (output / 'training/learning-curve.json').write_text(json.dumps({'checkpoints': rows}))
            (output / 'training/training-finished.json').write_text(json.dumps(
                {'optimizer_steps': 20, 'training_completed': False}))
            job = {'state': 'complete', 'charged_estimate_usd': 10}
            launch = {'prior_cost_estimate_usd': 3.4952}
            message, evidence = notify.summarize(output, job, launch, True)
            self.assertIn('předčasně', message)
            self.assertIn('30.0% → 35.0%', message)
            self.assertIn('13.50 USD', message)
            self.assertLessEqual(len(message), 500)
            self.assertFalse(evidence['completed'])
            message, evidence = notify.summarize(output, job, launch, False)
            self.assertNotIn('35.0%', message)
            self.assertIn('nelze posoudit', message)

    def test_running_job_does_not_send_or_call_provider(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / 'notification'
            directory.mkdir()
            (directory / 'config.json').write_text(json.dumps({'telegram_authorized': True, 'job_name': 'pilot'}))
            (root / 'launch.json').write_text(json.dumps({'job_name': 'pilot'}))
            (root / 'ledger.json').write_text(json.dumps({'jobs': {'pilot': {'state': 'running'}}}))
            with patch.object(notify, 'DIRECTORY', directory), patch.object(notify.cloud, 'STATE', root), \
                    patch.object(notify.cloud, 'list_pods') as provider, \
                    patch.object(notify.Alerts, 'emit') as emit, patch('sys.argv', ['notify']):
                notify.main()
                provider.assert_not_called()
                emit.assert_not_called()

    def test_uncertain_delivery_is_not_retried_even_if_results_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / 'notification'
            directory.mkdir()
            (directory / 'config.json').write_text(json.dumps({'telegram_authorized': True, 'job_name': 'pilot'}))
            (root / 'launch.json').write_text(json.dumps({'job_name': 'pilot'}))
            def failed_sender(message):
                raise TimeoutError('Synthetic timeout')
            alerts = notify.Alerts(directory / 'delivery.sqlite3', 'test', sender=failed_sender)
            alerts.emit('pilot', 'Results', 'Synthetic fixture')
            alerts.db.close()
            with patch.object(notify, 'DIRECTORY', directory), \
                    patch.object(notify.Alerts, 'emit') as emit, patch('sys.argv', ['notify']):
                notify.main()  # no ledger needed: durable intent blocks every later send
                emit.assert_not_called()


if __name__ == '__main__':
    unittest.main()
