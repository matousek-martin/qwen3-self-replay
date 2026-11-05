import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from .config import training_arguments
from .distributed import rewards_for_parts
from .hardware import allocation_spec, verify_worker, SCOPE, ENTRY
from .objective import grouped_rewards
from .stopping import decide, DEFAULTS
from ops.notify_shorterbetter_scaled import notices
from unittest.mock import patch, Mock


def point(iteration, saving=0., delta=0., ci=None):
    ci = ci or [delta - .5, delta + .5]
    return {'optimizer_step': iteration * 4, 'rollout_batches': iteration, 'questions': 128,
        'accuracy': .5 + delta / 100, 'accuracy_delta_pp': delta,
        'paired_question_accuracy_ci95_pp': ci, 'completed_accuracy': .48 + delta / 100,
        'completed_accuracy_delta_pp': delta, 'completed_accuracy_ci95_pp': ci,
        'mean_total_tokens': 5000 * (1-saving), 'token_saving': saving,
        'token_saving_ci95': [saving-.01, saving+.01], 'truncation_rate': .2,
        'invalid_thinking_rate': .1}


class ScaledTests(unittest.TestCase):
    def test_deterministic_resume_diagnostic_is_scoped_and_keeps_global_batches(self):
        p=json.loads((Path(__file__).resolve().parents[2]/'studies/shorterbetter/protocol.json').read_text())
        p['execution']={'scaled':True,'phase':'benchmark','world_size':4,
                        'deterministic_training':True,'resume_diagnostic':True}
        settings=training_arguments(p,'shorterbetter',17,'/tmp/none')
        self.assertTrue(settings['full_determinism'])
        self.assertEqual(settings['eval_strategy'],'no')
        self.assertEqual(settings['save_steps'],4)
        self.assertEqual(settings['generation_batch_size'],512)
        self.assertEqual(settings['gradient_accumulation_steps']*4,128)
        p['execution']['phase']='training'
        with self.assertRaises(ValueError):training_arguments(p,'shorterbetter',17,'/tmp/none')

    def test_engineering_phase_preserves_the_registered_full_protocol(self):
        from .launch_checks import engineering_protocol
        p = json.loads((Path(__file__).resolve().parents[2] / 'studies/shorterbetter/protocol.json').read_text())
        p['execution'] = {'phase': 'training', 'gated_full_training': True}
        e = engineering_protocol(p)
        self.assertEqual(p['training']['rollout_batches'], 300)
        self.assertEqual(e['training']['rollout_batches'], 8)
        self.assertEqual(e['monitoring']['every_rollout_batches'], 7)
        self.assertEqual(e['model'], p['model'])
        self.assertEqual(e['reward'], p['reward'])
        self.assertEqual(e['data'], p['data'])

    def test_global_groups_cross_rank_boundaries(self):
        rows = [('a' if i < 8 else 'b', i % 3 == 0, 100+i) for i in range(16)]
        expected, _ = grouped_rewards(*zip(*rows))
        parts = [rows[:3], rows[3:9], rows[9:]]
        actual = sum([rewards_for_parts(parts, rank)[0] for rank in range(3)], [])
        self.assertEqual(actual, expected)
        with self.assertRaises(ValueError):
            rewards_for_parts([rows[:7]], 0)
        with self.assertRaisesRegex(ValueError, 'advantage groups'):
            rewards_for_parts([rows[::2] + rows[1::2]], 0)

    def test_world_size_does_not_change_exposure_or_minibatches(self):
        p = json.loads((Path(__file__).resolve().parents[2] / 'studies/shorterbetter/protocol.json').read_text())
        for world in (1, 2, 4, 8):
            p['execution'] = {'world_size': world}
            args = training_arguments(p, 'shorterbetter', 17, '/tmp/none')
            self.assertEqual(args['gradient_accumulation_steps'] * world, 128)
            self.assertEqual(args['generation_batch_size'], 512)
            self.assertEqual(args['max_steps'], 1200)

    def test_scope_cannot_relax_other_jobs_or_wrong_gpu_count(self):
        args = SimpleNamespace(hardware_scope=SCOPE, gpu_count=4, gpu='NVIDIA H100 80GB HBM3',
            name='shorterbetter-scaled-test', entry=ENTRY, max_cost_usd=35)
        self.assertEqual(allocation_spec(args)['count'], 4)
        args.name = 'unrelated-job'
        with self.assertRaises(ValueError): allocation_spec(args)
        lease = {'hardware_scope': SCOPE, 'gpu_count': 4, 'gpu': args.gpu, 'job_name': 'shorterbetter-scaled-test'}
        verify_worker([args.gpu]*4, lease)
        with self.assertRaises(RuntimeError): verify_worker([args.gpu]*2, lease)
        with self.assertRaises(RuntimeError): verify_worker(['NVIDIA GeForce RTX 4090']*4, lease)

    def test_noisy_early_drop_does_not_abort(self):
        report = {'checkpoints': [point(0), point(10, delta=-2, ci=[-5, 1]), point(20)]}
        self.assertFalse(decide(report, DEFAULTS)['stop'])

    def test_regression_requires_three_evaluations(self):
        rows = [point(0)] + [point(i, delta=-5) for i in (20,30,40)]
        self.assertFalse(decide({'checkpoints': rows[:-1]}, DEFAULTS)['stop'])
        self.assertEqual(decide({'checkpoints': rows}, DEFAULTS)['reason'], 'persistent_regression')

    def test_stagnation_is_not_declared_before_100(self):
        rows = [point(i) for i in range(0, 101, 10)]
        self.assertFalse(decide({'checkpoints': rows[:-1]}, DEFAULTS)['stop'])
        self.assertEqual(decide({'checkpoints': rows}, DEFAULTS)['reason'], 'no_useful_progress_within_window')

    def test_short_but_bad_or_uncertain_outputs_are_not_success(self):
        rows = [point(0)] + [point(i, saving=.3, delta=-4, ci=[-8, 0]) for i in range(10,101,10)]
        decision = decide({'checkpoints': rows}, DEFAULTS)
        self.assertIsNone(decision['best_eligible_step'])
        self.assertNotEqual(decision['reason'], 'useful_plateau_needs_confirmation')
        rows = [point(0)] + [point(i, saving=.3) for i in range(10,101,10)]
        for r in rows[1:]: r['truncation_rate'] = .3
        self.assertIsNone(decide({'checkpoints': rows}, DEFAULTS)['best_eligible_step'])

    def test_useful_plateau_and_progress_are_distinct(self):
        rows = [point(0)] + [point(i, saving=.2) for i in range(10,101,10)]
        self.assertEqual(decide({'checkpoints': rows}, DEFAULTS)['reason'], 'useful_plateau_needs_confirmation')
        rows = [point(0)] + [point(i, saving=i/500) for i in range(10,101,10)]
        self.assertFalse(decide({'checkpoints': rows}, DEFAULTS)['stop'])

    def test_incomplete_metrics_fail_closed(self):
        with self.assertRaises(ValueError): decide({'checkpoints': [point(10)]}, DEFAULTS)
        p = point(0); p['accuracy'] = float('nan')
        with self.assertRaises(ValueError): decide({'checkpoints': [p]}, DEFAULTS)

    def test_notifications_distinguish_stop_and_release(self):
        with tempfile.TemporaryDirectory() as t:
            out = Path(t); (out/'training').mkdir()
            (out/'training/decision.json').write_text(json.dumps({'stop': True, 'reason': 'persistent_regression'}))
            job = {'state': 'running', 'created_at': 100, 'gpu_count': 4, 'gpu': 'H100'}
            events = notices('test', job, out, 200)
            self.assertIn('stop', [e[0] for e in events])
            self.assertNotIn('final', [e[0] for e in events])
            job.update(released_at=200, backup_verified=True, charged_estimate_usd=1.2)
            self.assertIn('final', [e[0] for e in notices('test', job, out, 200)])
            (out/'training/failure-rank-001.json').write_text(json.dumps({'category': 'numerical_failure'}))
            events = notices('test', job, out, 200)
            self.assertIn('failure', [e[0] for e in events])
            self.assertIn('numerická chyba', next(e[2] for e in events if e[0] == 'final'))

    def test_uncertain_telegram_delivery_is_not_retried(self):
        import ops.notify_shorterbetter_scaled as notifier
        from project_alerts import Alerts
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); notification = root/'notify'; notification.mkdir()
            name = 'shorterbetter-scaled-test'
            (notification/'jobs.json').write_text(json.dumps({name: {'telegram_authorized': True}}))
            (root/'ledger.json').write_text(json.dumps({'jobs': {name: {'state': 'running',
                'created_at': 10**12, 'hardware_scope': SCOPE, 'gpu': 'H100', 'gpu_count': 4}}}))
            sender = Mock(side_effect=RuntimeError('uncertain network response'))
            with patch.object(notifier, 'DIRECTORY', notification), patch.object(notifier.cloud, 'STATE', root), \
                 patch.object(notifier, 'Alerts', side_effect=lambda p, n, **kwargs: Alerts(p, n, sender)), \
                 patch('sys.argv', ['notifier']):
                notifier.main(); notifier.main()
            self.assertEqual(sender.call_count, 1)

    def test_throughput_does_not_invent_a_larger_gpu_speed(self):
        from .throughput import forecast
        with tempfile.TemporaryDirectory() as t:
            out = Path(t); (out/'rank-000').mkdir()
            (out/'steps.jsonl').write_text('\n'.join(json.dumps({'optimizer_step': s, 'time': s*30}) for s in range(1,29)))
            (out/'rank-000/metrics.jsonl').write_text(json.dumps({'eval_runtime': 100, 'time': 150}))
            result = forecast(out, {'monitoring': {'every_rollout_batches': 7}},
                {'created_at': 0, 'gpu': 'H100', 'gpu_count': 4, 'budget_rate': 20})
            self.assertEqual(result['measured_rollouts'], 5)
            self.assertEqual(result['gpu_count'], 4)
            self.assertFalse(result['fits_eight_hours_estimate'])

    def test_failed_startup_update_prevents_expensive_baseline(self):
        from . import start_scaled
        from .prepare import sha
        import time
        for success in (True, False):
            with self.subTest(success=success), tempfile.TemporaryDirectory() as t:
                root = Path(t); launch = root/'experiments/shorterbetter/launch'; launch.mkdir(parents=True)
                protocol = {'status': 'locked', 'budget': {'cap_authorized': True, 'proposed_total_pilot_cap_usd': 35},
                    'execution': {'scaled': True, 'phase': 'benchmark', 'world_size': 2, 'gpu': 'test'}, 'seeds': [17]}
                (launch/'protocol.json').write_text(json.dumps(protocol))
                evidence = {k: True for k in ('data_audit','reward_parity','backend_cpu_parity','distributed_cpu_check',
                    'stopping_tests','backup_release_integration')}
                evidence['protocol_sha256'] = sha(launch/'protocol.json')
                (launch/'readiness.json').write_text(json.dumps(evidence))
                (root/'worker-lease.json').write_text(json.dumps({'gpu_count':2,'gpu':'test','max_cost_usd':35,
                    'hard_deadline':time.time()+3600,'job_name':'fixture'}))
                calls = []
                def run(command, **kwargs):
                    calls.append(command)
                    if '--startup-check' in command:
                        for rank in range(2):
                            path = root/f'output/startup-check/rank-{rank:03d}'; path.mkdir(parents=True)
                            (path/'parameter-update-check.json').write_text(json.dumps({'status':'passed' if success else 'failed'}))
                with patch.object(start_scaled, 'ROOT', root), patch.object(start_scaled.subprocess, 'run', side_effect=run):
                    if success: start_scaled.main()
                    else:
                        with self.assertRaisesRegex(RuntimeError, 'startup update'): start_scaled.main()
                self.assertEqual(len(calls), 2 if success else 1)
                if success:
                    self.assertNotIn('--resume', calls[1])
                    self.assertTrue(calls[1][-1].endswith('/output/training'))


if __name__ == '__main__': unittest.main()
