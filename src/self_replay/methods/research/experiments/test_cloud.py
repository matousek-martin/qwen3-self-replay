"""No-network regression tests for bounded cloud lifecycle and archive races."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import os
import signal
import time
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('cloud', Path(__file__).with_name('cloud.py'))
cloud = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cloud)


class CloudTests(unittest.TestCase):
    def test_startup_resume_never_allocates_and_rejects_already_launched_worker(self):
        root=self.state_dir if hasattr(self,'state_dir') else cloud.STATE
        (root/'experiments').mkdir(exist_ok=True)
        (root/'experiments/eval.sh').write_text('# fixture')
        data=root/'lab/experiments/data';data.mkdir(parents=True,exist_ok=True)
        (data/'pilot.jsonl').write_text('{}')
        self.job(state='starting',entry='experiments/eval.sh',hardware_scope='replay-evaluation-v1',
                 max_cost_usd=3.8,worker_launch_attempted_at=time.time())
        args=SimpleNamespace(name='a',entry='experiments/eval.sh',hardware_scope='replay-evaluation-v1',
                             max_cost_usd=3.8,resume_startup=True)
        with patch.object(cloud,'ROOT',root),patch.object(cloud.subprocess,'run',return_value=SimpleNamespace(returncode=0)), \
             patch.object(cloud,'allocate') as allocate,patch.object(cloud,'remote') as remote:
            with self.assertRaisesRegex(RuntimeError,'never launched'):cloud.run(args)
        allocate.assert_not_called();remote.assert_not_called()

    def test_vast_startup_failure_quarantines_machine_not_gpu_class(self):
        job={'gpu':'NVIDIA GeForce RTX 4090','provider':'vast','created_at':time.time()-100,
             'released_at':time.time(),'error':'Pod startup exceeded ten minutes',
             'provider_quote':{'offer_id':10,'offer':{'machine_id':5}}}
        jobs={'failed':job}
        self.assertFalse(cloud.unhealthy_placement(jobs,job['gpu'],'vast'))
        self.assertTrue(cloud.unhealthy_placement(jobs,job['gpu'],'vast',{'offer_id':11,'offer':{'machine_id':5}}))
        self.assertFalse(cloud.unhealthy_placement(jobs,job['gpu'],'vast',{'offer_id':12,'offer':{'machine_id':6}}))
        self.assertFalse(cloud.unhealthy_placement(jobs,job['gpu'],'runpod'))
        job['provider']='runpod'
        self.assertTrue(cloud.unhealthy_placement(jobs,job['gpu'],'runpod'))

    def test_named_completion_authorization_launches_and_renews_without_time_or_money_cap(self):
        name = 'shorterbetter-scaled-completion-test'
        self.job('historical', released_at=time.time(), charged_estimate_usd=199)
        authorization = {'authorized': True, 'job_name': name, 'scope': 'finish_registered_1200_updates',
            'gpu': 'NVIDIA A100-SXM4-80GB', 'gpu_count': 4, 'optimizer_updates': 1200,
            'max_cost_usd': None, 'max_runtime_hours': None, 'max_gpu_rate': 1.79}
        cloud.write(cloud.STATE/'authorizations'/f'{name}.json', authorization)
        args = SimpleNamespace(name=name, entry='experiments/shorterbetter/run-scaled.sh',
            gpu=authorization['gpu'], gpu_count=4, hardware_scope='shorterbetter-scaled-v1',
            completion_budget_scope=authorization['scope'], hours=2, max_cost_usd=None, max_runtime_hours=None)
        def api(path, method='GET', payload=None):
            if path.startswith('catalog/gpus/'):
                return {'availability': 'HIGH', 'price': {'secure': 1.79}}
            return {'id': 'authorized-pod', 'name': payload['name']}
        with patch.object(cloud, 'api', side_effect=api): cloud.allocate(args)
        job = self.state(name)
        self.assertNotIn('hard_deadline', job)
        self.assertNotIn('max_cost_usd', job)
        self.assertAlmostEqual(job['budget_rate'], 9.1)
        cloud.update(name, deadline=time.time()-1)
        self.assertTrue(cloud.renew_budget(name))
        self.assertGreater(self.state(name)['deadline'], time.time()+3500)
        args.name = 'shorterbetter-scaled-not-authorized'
        with patch.object(cloud, 'api') as api:
            with self.assertRaisesRegex(ValueError, 'named user authorization'): cloud.allocate(args)
            api.assert_not_called()

    def test_scaled_hardware_count_and_total_rate_are_reserved(self):
        args = SimpleNamespace(name='shorterbetter-scaled-reservation-test',
            entry='experiments/shorterbetter/run-scaled.sh', gpu='NVIDIA A100-SXM4-80GB',
            gpu_count=4, hardware_scope='shorterbetter-scaled-v1',
            hours=.5, max_cost_usd=35., max_runtime_hours=2.)
        requests = []
        def api(path, method='GET', payload=None):
            requests.append((path, method, payload))
            if path.startswith('catalog/gpus/'):
                return {'availability': 'HIGH', 'price': {'secure': 1.79}}
            self.assertEqual((path, method), ('pods', 'POST'))
            return {'id': 'scaled-test-pod', 'name': payload['name']}
        with patch.object(cloud, 'api', side_effect=api):
            cloud.allocate(args)
        created = next(p for _, m, p in requests if m == 'POST')
        self.assertEqual(created['gpu']['count'], 4)
        self.assertIn('count=4', requests[0][0])
        with cloud.ledger() as state:
            job = state['jobs'][args.name]
        self.assertAlmostEqual(job['rate'], 7.16)
        self.assertAlmostEqual(job['budget_rate'], 11.15)
        self.assertEqual(job['reserved_usd'], 35.)
        self.assertEqual(job['hardware_scope'], args.hardware_scope)
        self.assertLessEqual(job['hard_deadline']-job['created_at'], 7200)

    def test_accelerated_hardware_cannot_escape_scope(self):
        args = SimpleNamespace(name='unrelated', entry='experiments/shorterbetter/run-scaled.sh',
            gpu='NVIDIA H100 80GB HBM3', gpu_count=4, hardware_scope='shorterbetter-scaled-v1',
            hours=.5, max_cost_usd=35.)
        with patch.object(cloud, 'api') as api:
            with self.assertRaises(ValueError): cloud.allocate(args)
            api.assert_not_called()

    def test_watch_resumes_until_release_without_allocating_or_launching(self):
        row = self.job(completion_driven=True)
        pod = {'id': row['pod_id'], 'name': row['pod_name'], 'status': 'RUNNING'}
        observations = []
        def observe(actual, name):
            observations.append((actual, name))
            if len(observations) == 2:
                cloud.update(name, released_at=time.time())
                return True
            return False
        with patch.object(cloud, 'api', return_value=pod) as api, \
             patch.object(cloud, 'observe_completion', side_effect=observe), \
             patch.object(cloud, 'allocate') as allocate, \
             patch.object(cloud, 'remote') as remote, patch.object(cloud.time, 'sleep') as sleep:
            cloud.watch('a')
        self.assertEqual(len(observations), 2)
        self.assertTrue(all(call.args == ('pods/pod-a',) for call in api.call_args_list))
        allocate.assert_not_called()
        remote.assert_not_called()
        sleep.assert_called_once_with(45)

    def test_watch_released_job_is_noop(self):
        self.job(released_at=time.time())
        with patch.object(cloud, 'api') as api:
            cloud.watch('a')
        api.assert_not_called()

    def test_watch_rejects_wrong_provider_identity(self):
        self.job(completion_driven=True)
        with patch.object(cloud, 'api', return_value={'id': 'other', 'name': 'other'}), \
             patch.object(cloud, 'observe_completion') as observe:
            with self.assertRaisesRegex(RuntimeError, 'identity'):
                cloud.watch('a')
        observe.assert_not_called()

    def test_watch_controller_error_does_not_stop_worker(self):
        row = self.job(completion_driven=True)
        with patch.object(cloud, 'api', return_value={'id': row['pod_id'], 'name': row['pod_name']}), \
             patch.object(cloud, 'observe_completion', side_effect=RuntimeError('temporary failure')), \
             patch.object(cloud, 'release') as release:
            with self.assertRaisesRegex(RuntimeError, 'temporary failure'):
                cloud.watch('a')
        release.assert_not_called()

    def test_active_worker_renews_budget_instead_of_timing_out(self):
        self.job(completion_driven=True)
        before = self.state()['deadline']
        self.assertTrue(cloud.renew_budget('a'))
        self.assertGreater(self.state()['deadline'], before + 3500)
        self.assertEqual(len(self.state()['budget_extensions']), 1)
        self.assertTrue(cloud.renew_budget('a'))
        self.assertEqual(len(self.state()['budget_extensions']), 1)

    def test_budget_extension_cannot_exceed_global_allowance(self):
        self.job(completion_driven=True)
        self.job('other', reserved_usd=cloud.USABLE, deadline=time.time() + 3600)
        before = self.state()['deadline']
        self.assertFalse(cloud.renew_budget('a'))
        self.assertEqual(self.state()['deadline'], before)

    def test_explicit_completion_override_renews_only_its_existing_pod(self):
        self.job(completion_driven=True, completion_budget_override={
            'authorized': True, 'job_name': 'a', 'pod_id': 'pod-a',
            'scope': 'finish_registered_80_updates'})
        self.job('other', reserved_usd=cloud.USABLE, deadline=time.time() + 3600)
        self.assertTrue(cloud.renew_budget('a'))
        cloud.update('a', deadline=time.time() + 60, completion_budget_override={
            'authorized': True, 'job_name': 'a', 'pod_id': 'wrong-pod',
            'scope': 'finish_registered_80_updates'})
        self.assertFalse(cloud.renew_budget('a'))

    def test_completion_handoff_preserves_archive_and_runs_once(self):
        from ops.shorterbetter_completion import maybe_continue
        self.job(completion_driven=True, completion_budget_override={
            'authorized': True, 'job_name': 'a', 'pod_id': 'pod-a',
            'scope': 'finish_registered_80_updates'}, completion_continuation={'phase': 'armed'})
        base = cloud.STATE / 'a/output/training'
        (base / 'checkpoint-20').mkdir(parents=True)
        (base / 'checkpoint-20/trainer_state.json').write_text('{"global_step":20}')
        (base / 'training-finished.json').write_text('{"training_completed":false,"optimizer_steps":21}')
        with patch.object(cloud, 'remote', return_value='{"phase":"launched"}') as remote:
            self.assertTrue(maybe_continue(cloud, {'id': 'pod-a'}, 'a', {'exit_code': 0}))
            self.assertFalse(maybe_continue(cloud, {'id': 'pod-a'}, 'a', {'exit_code': 0}))
            remote.assert_called_once()
        self.assertTrue((cloud.STATE / 'a/output-segment-0/training/training-finished.json').exists())
        self.assertFalse((cloud.STATE / 'a/output/training/training-finished.json').exists())
        self.assertEqual(self.state()['completion_continuation']['phase'], 'resumed')

    def test_fixed_job_envelope_never_renews_past_hard_deadline(self):
        now = time.time()
        self.job(completion_driven=True, created_at=now - 100,
                 deadline=now + 600, hard_deadline=now + 1600,
                 max_cost_usd=20, reserved_usd=20)
        # This isolated fixture explicitly supplies a $20 campaign allowance.
        with patch.object(cloud, 'USABLE', 20):
            self.assertTrue(cloud.renew_budget('a'))
        self.assertEqual(self.state()['deadline'], now + 1600)
        self.assertEqual(self.state()['reserved_usd'], 20)
        with patch.object(cloud.time, 'time', return_value=now + 1350):
            self.assertFalse(cloud.renew_budget('a'))
        self.assertEqual(self.state()['deadline'], now + 1600)

    def test_invalid_fixed_cap_cannot_allocate(self):
        args = SimpleNamespace(name='invalid-cap', hours=4, entry='entry.sh', max_cost_usd=2)
        with patch.object(cloud, 'api') as api:
            with self.assertRaises(ValueError):
                cloud.allocate(args)
        api.assert_not_called()

    def test_stale_controller_does_not_kill_live_completion_worker(self):
        self.job(completion_driven=True, heartbeat=time.time() - 900, deadline=time.time() + 3600)
        with patch.object(cloud, 'api', return_value={'status': 'RUNNING', 'ssh': {'direct': {}}}), \
             patch.object(cloud, 'backup'), patch.object(cloud, 'worker_probe', return_value={'worker_status': 'alive'}), \
             patch.object(cloud, 'release') as release:
            # A nonempty direct endpoint is normally returned by the provider.
            cloud.api.return_value['ssh']['direct'] = {'host': 'example'}
            cloud.guard()
        release.assert_not_called()
        self.assertEqual(self.state()['worker_observation']['worker_status'], 'alive')

    def test_verified_completion_releases_without_waiting_for_reserved_hours(self):
        self.job(completion_driven=True, deadline=time.time() + 10000)
        base = self.archive()
        (base / 'worker-state.json').write_text(json.dumps({'state': 'finished', 'exit_code': 0}))
        with patch.object(cloud, 'backup'), patch.object(cloud, 'release') as release:
            self.assertTrue(cloud.observe_completion({}, 'a'))
        release.assert_called_once_with('a', 'complete', verified=True)

    def test_authorized_technical_repair_hold_expires_and_never_holds_success(self):
        base=self.archive()
        (base/'worker-state.json').write_text('{}')
        for exit_code, active, held in [(1, True, True), (1, False, False), (0, True, False)]:
            with self.subTest(exit_code=exit_code, active=active):
                self.job(completion_driven=True, deadline=time.time()+3600,
                         technical_repair_authorized=True,
                         technical_repair_hold_until=time.time()+(300 if active else -1))
                with patch.object(cloud, 'backup'), patch.object(cloud, 'release') as release, \
                     patch.object(cloud, 'verify_backup', return_value={'exit_code':exit_code}), \
                     patch('experiments.shorterbetter.budget.ticket', return_value={'scope':'finish_registered_1200_updates'}):
                    self.assertEqual(cloud.observe_completion({},'a'), not held)
                if held: release.assert_not_called()
                else: release.assert_called_once_with('a','complete' if exit_code==0 else 'failed',verified=True)

    def test_completion_marker_without_supervisor_state_is_not_success(self):
        self.job(completion_driven=True, deadline=time.time() + 10000)
        self.archive()
        with patch.object(cloud, 'backup'), patch.object(cloud, 'release') as release, \
                patch.object(cloud, 'worker_probe', return_value={'worker_status': 'alive'}):
            self.assertFalse(cloud.observe_completion({}, 'a'))
        release.assert_not_called()

    def test_unreachable_expired_budget_does_not_wait_for_backup(self):
        self.job(completion_driven=True)
        with patch.object(cloud, 'worker_probe', return_value={'worker_status': 'unknown'}), \
             patch.object(cloud, 'backup') as backup, patch.object(cloud, 'release') as release:
            self.assertTrue(cloud.observe_completion({}, 'a'))
        backup.assert_not_called()
        release.assert_called_once_with('a', 'budget_lease_expired_unreachable')

    def test_dead_worker_gets_a_final_archive_copy_before_failure_cleanup(self):
        self.job(completion_driven=True, deadline=time.time() + 3600)
        with patch.object(cloud, 'backup'), patch.object(cloud, 'worker_probe', return_value={'worker_status': 'dead'}), \
             patch.object(cloud, 'release') as release:
            self.assertFalse(cloud.observe_completion({}, 'a'))
        release.assert_not_called()
        self.assertIn('worker_dead_observed_at', self.state())

    def test_transient_missing_ssh_does_not_end_live_worker_watch(self):
        self.job(completion_driven=True, deadline=time.time()+3600)
        with patch.object(cloud,'backup',side_effect=RuntimeError('Direct SSH is not ready')), \
             patch.object(cloud,'worker_probe',return_value={'worker_status':'alive'}), \
             patch.object(cloud,'release') as release:
            self.assertFalse(cloud.observe_completion({},'a'))
        release.assert_not_called()
        self.assertEqual(self.state()['backup_error'],'Direct SSH is not ready')

    def test_named_completion_archive_survives_expired_lease_and_slow_copy(self):
        self.job(completion_driven=True, deadline=time.time()-10,
                 worker_dead_observed_at=time.time()-600)
        probe = {'worker_status': 'dead', 'exit_marker': {'status': 'valid', 'exit_code': 0}}
        with patch.object(cloud, 'backup', side_effect=subprocess.TimeoutExpired('rsync', 90)), \
             patch.object(cloud, 'worker_probe', return_value=probe), \
             patch('experiments.shorterbetter.budget.ticket', return_value={'scope': 'finish_registered_1200_updates'}), \
             patch.object(cloud, 'release') as release:
            self.assertFalse(cloud.observe_completion({}, 'a'))
        release.assert_not_called()
        self.assertTrue(self.state()['final_archive_pending'])
        self.assertGreater(self.state()['deadline'], time.time())

    def test_named_cheap_archive_waits_for_copy_but_never_crosses_hard_cap(self):
        for remaining, held in ((1800, True), (200, False)):
            with self.subTest(remaining=remaining):
                self.job(completion_driven=True, deadline=time.time()+3600,
                         hard_deadline=time.time()+remaining,
                         worker_dead_observed_at=time.time()-600)
                probe={'worker_status':'dead','exit_marker':{'status':'valid','exit_code':0}}
                with patch.object(cloud,'backup',side_effect=subprocess.TimeoutExpired('rsync',90)), \
                     patch.object(cloud,'worker_probe',return_value=probe), \
                     patch('experiments.shorterbetter.budget.ticket',return_value={'scope':'shorterbetter_4090_verification'}), \
                     patch.object(cloud,'release') as release:
                    self.assertEqual(cloud.observe_completion({},'a'),not held)
                if held:release.assert_not_called()
                else:release.assert_called_once_with('a','worker_exited_without_verified_result')

    def test_uncapped_archive_exception_requires_valid_remote_completion(self):
        self.job(completion_driven=True, deadline=time.time()+3600,
                 worker_dead_observed_at=time.time()-600)
        probe = {'worker_status': 'dead', 'exit_marker': {'status': 'absent'}}
        with patch.object(cloud, 'backup'), patch.object(cloud, 'worker_probe', return_value=probe), \
             patch('experiments.shorterbetter.budget.ticket', return_value={'scope': 'finish_registered_1200_updates'}), \
             patch.object(cloud, 'release') as release:
            self.assertTrue(cloud.observe_completion({}, 'a'))
        release.assert_called_once_with('a', 'worker_exited_without_verified_result')

    def test_missing_orphan_is_confirmed_released(self):
        self.job(completion_driven=True, heartbeat=time.time() - 900)
        with patch.object(cloud, 'api', side_effect=RuntimeError('Runpod HTTP 404: absent')):
            cloud.guard()
        self.assertIn('released_at', self.state())
        self.assertEqual(self.state()['state'], 'provider_ended')

    def test_historical_non4090_placements_do_not_change_required_gpu(self):
        self.job('failed-placement', gpu='NVIDIA L40', released_at=time.time(),
                 charged_estimate_usd=.1, error='Pod startup exceeded ten minutes; no training launched')
        self.job('clean-prior', gpu='NVIDIA L4', released_at=time.time(),
                 charged_estimate_usd=.1, error=None)
        quoted = []
        def api(path, method='GET', payload=None):
            if path.startswith('catalog/'):
                quoted.append(path)
                return {'availability': 'HIGH', 'price': {'secure': .59}}
            if method == 'POST':
                return {'id': 'new', 'name': payload['name'], 'cost': .59}
            self.fail('Unexpected call')
        with patch.object(cloud, 'api', side_effect=api):
            cloud.allocate(SimpleNamespace(name='replacement', hours=5, entry='entry.sh'), wait_seconds=120)
        self.assertEqual(len(quoted), 1)
        self.assertIn('NVIDIA%20GeForce%20RTX%204090', quoted[0])

    def test_non4090_override_is_rejected_before_api_or_reservation(self):
        for gpu in ('NVIDIA L4', 'NVIDIA L40', 'NVIDIA RTX A6000', 'NVIDIA RTX A5000'):
            with patch.object(cloud, 'api') as api:
                with self.assertRaises(ValueError):
                    cloud.allocate(SimpleNamespace(name='forbidden', hours=4, entry='entry.sh', gpu=gpu))
            api.assert_not_called()
        self.assertFalse((cloud.STATE / 'ledger.json').exists())

    def test_actual_provider_capacity_rejection_wording(self):
        error = RuntimeError('Runpod HTTP 400: ' + json.dumps({
            'detail': 'There are no longer any instances available with the requested specifications. Please refresh and try again.',
            'status': 400, 'title': 'Bad Request'}))
        self.assertTrue(cloud.capacity_rejected(error))
        self.assertFalse(cloud.capacity_rejected(RuntimeError('Runpod HTTP 400: invalid configuration')))

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.patch = patch.object(cloud, 'STATE', Path(self.temp.name))
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def job(self, name='a', **kwargs):
        row = dict(pod_id='pod-' + name, pod_name='thinkingcap-pilot-' + name + '-nonce',
                   identity_confirmed=True, state='running', created_at=time.time() - 60,
                   heartbeat=time.time(), deadline=time.time() - 1,
                   budget_rate=1, reserved_usd=2)
        row.update(kwargs)
        with cloud.ledger() as data:
            data['jobs'][name] = row
        return row

    def state(self, name='a'):
        return json.loads((cloud.STATE / 'ledger.json').read_text())['jobs'][name]

    def archive(self):
        base = cloud.STATE / 'a' / 'output'
        base.mkdir(parents=True)
        result = b'{"exit_code":0,"finished_at":"now"}\n'
        (base / 'result.json').write_bytes(result)
        (base / 'exit.json').write_bytes(result)
        (base / 'weights.bin').write_bytes(b'checkpoint')
        (base / 'SHA256SUMS').write_text(''.join(
            hashlib.sha256((base / name).read_bytes()).hexdigest() + '  ./' + name + '\n'
            for name in ['result.json', 'weights.bin']))
        return base

    def test_empty_manifest_is_not_success(self):
        base = self.archive()
        (base / 'SHA256SUMS').write_text('')
        with self.assertRaises(RuntimeError):
            cloud.verify_backup('a')

    def test_partial_snapshot_rejected(self):
        base = self.archive()
        (base / 'weights.bin').write_bytes(b'partial')
        with self.assertRaises(RuntimeError):
            cloud.verify_backup('a')

    def test_marker_must_match_hashed_result(self):
        base = self.archive()
        (base / 'exit.json').write_text('{"exit_code":1}')
        with self.assertRaises(RuntimeError):
            cloud.verify_backup('a')

    def test_valid_snapshot_verified(self):
        self.archive()
        self.assertEqual(cloud.verify_backup('a')['exit_code'], 0)

    def test_supervisor_must_confirm_matching_archive_exit_status(self):
        base = self.archive()
        state = base / 'worker-state.json'
        for value in ({'state': 'running'}, {'state': 'finished', 'exit_code': 143}):
            state.write_text(json.dumps(value))
            with self.assertRaises(RuntimeError):
                cloud.verify_backup('a')
        state.write_text(json.dumps({'state': 'finished', 'exit_code': 0}))
        self.assertEqual(cloud.verify_backup('a')['exit_code'], 0)

    def test_deadline_cleanup_never_waits_for_backup_or_ssh(self):
        self.job()
        def api(path, method='GET', payload=None):
            if method == 'DELETE':
                return {}
            return {'id': 'pod-a', 'status': 'TERMINATED', 'ssh': None}
        with patch.object(cloud, 'api', side_effect=api), patch.object(cloud, 'backup') as backup:
            cloud.release('a', 'deadline')
        backup.assert_not_called()
        self.assertIn('released_at', self.state())

    def test_delete_is_not_marked_released_until_confirmed(self):
        self.job()
        with patch.object(cloud, 'api', return_value={'status': 'RUNNING'}):
            cloud.release('a', 'deadline')
        state = self.state()
        self.assertNotIn('released_at', state)
        self.assertEqual(state['state'], 'delete_requested')
        self.assertGreaterEqual(cloud.liability({'a': state}), state['reserved_usd'])

    def test_ambiguous_delete_reconciles_404(self):
        self.job(delete_requested_at=time.time())
        with patch.object(cloud, 'api', side_effect=RuntimeError('Runpod HTTP 404: missing')):
            cloud.guard()
        self.assertIn('released_at', self.state())

    def test_one_api_failure_does_not_block_other_cleanup(self):
        self.job('a')
        self.job('b')
        calls = []
        def api(path, method='GET', payload=None):
            calls.append((path, method))
            if path.endswith('pod-a'):
                raise RuntimeError('unavailable a')
            return {'status': 'TERMINATED'}
        with patch.object(cloud, 'api', side_effect=api):
            cloud.guard()
        self.assertIn(('pods/pod-b', 'DELETE'), calls)
        self.assertIn('released_at', self.state('b'))
        self.assertIn('guard_error', self.state('a'))

    def test_ambiguous_create_reconciles_without_second_post(self):
        row = self.job(pod_id=None, identity_confirmed=False, state='creating')
        calls = []
        def api(path, method='GET', payload=None):
            calls.append((path, method))
            if path == 'pods':
                return {'pods': [{'id': 'adopted', 'name': row['pod_name']}], 'pagination': {'hasNextPage': False}}
            return {'status': 'TERMINATED'}
        with patch.object(cloud, 'api', side_effect=api):
            cloud.guard()
        self.assertEqual(self.state()['pod_id'], 'adopted')
        self.assertFalse(any(method == 'POST' for _, method in calls))
        self.assertIn(('pods/adopted', 'DELETE'), calls)

    def test_running_liability_grows_past_reservation(self):
        row = self.job(created_at=time.time() - 4 * 3600, reserved_usd=1)
        self.assertGreaterEqual(cloud.liability({'a': row}), 4)

    def test_sync_has_wall_clock_timeout(self):
        with patch.object(cloud, 'ssh_args', return_value=['ssh', 'host']), \
             patch.object(cloud.subprocess, 'run') as run:
            cloud.sync({}, 'source', 'dest', timeout=7)
        self.assertEqual(run.call_args.kwargs['timeout'], 7)
        self.assertIn('--partial-dir=.rsync-partial', run.call_args.args[0])

    def test_large_checkpoint_transfer_is_bounded_by_lease(self):
        self.job(completion_driven=True, deadline=time.time() + 180,
                 backup_timeout_seconds=300)
        with patch.object(cloud, 'backup') as backup, \
             patch.object(cloud, 'worker_probe', return_value={'worker_status': 'alive'}):
            cloud.observe_completion({}, 'a')
        self.assertGreater(backup.call_args.kwargs['timeout'], 110)
        self.assertLessEqual(backup.call_args.kwargs['timeout'], 120)

    def test_large_checkpoint_transfer_gets_requested_bounded_window(self):
        self.job(completion_driven=True, deadline=time.time() + 10000,
                 backup_timeout_seconds=300)
        with patch.object(cloud, 'backup') as backup, \
             patch.object(cloud, 'worker_probe', return_value={'worker_status': 'alive'}):
            cloud.observe_completion({}, 'a')
        self.assertEqual(backup.call_args.kwargs['timeout'], 300)

    def test_actual_provider_capacity_json_is_recognized(self):
        error = RuntimeError('Runpod HTTP 400: {"error":{"code":"BAD_REQUEST","message":"There are no longer any instances available with the requested specifications. Please refresh and try again."}}')
        self.assertTrue(cloud.capacity_rejected(error))
        self.job(pod_id=None, state='creating', error=str(error))
        with patch.object(cloud, 'api', return_value={'pods': [], 'pagination': {'hasNextPage': False}}):
            cloud.guard()
        self.assertEqual(self.state()['charged_estimate_usd'], 0)

    def test_capacity_rejection_reconciles_then_retries_only_4090(self):
        attempts = []
        def api(path, method='GET', payload=None):
            if path.startswith('catalog/'):
                return {'availability': 'LOW', 'price': {'secure': 0.82}}
            if method == 'POST':
                attempts.append(payload)
                if len(attempts) == 1:
                    raise RuntimeError('Runpod HTTP 400: selected GPU is no longer available')
                return {'id': 'new-pod', 'name': payload['name'], 'cost': 0.82}
            if path == 'pods':
                return {'pods': [], 'pagination': {'hasNextPage': False}}
            self.fail('Unexpected API call ' + path)
        args = SimpleNamespace(name='retry', hours=6, entry='experiments/run.sh')
        with patch.object(cloud, 'api', side_effect=api), patch.object(cloud.time, 'sleep'):
            result = cloud.allocate(args, wait_seconds=120)
        self.assertEqual(result['id'], 'new-pod')
        self.assertEqual(len(attempts), 2)
        self.assertNotEqual(attempts[0]['name'], attempts[1]['name'])
        self.assertEqual(attempts[0]['gpu']['id'], 'NVIDIA GeForce RTX 4090')
        self.assertEqual(attempts[1]['gpu']['id'], 'NVIDIA GeForce RTX 4090')
        job = self.state('retry')
        self.assertEqual(job['allocation_attempts'][0]['outcome'], 'capacity_rejected_absent')
        self.assertEqual(job['pod_id'], 'new-pod')
        self.assertIsNone(job['error'])

    def test_ambiguous_create_never_retries_or_refunds(self):
        posts = []
        def api(path, method='GET', payload=None):
            if path.startswith('catalog/'):
                return {'availability': 'LOW', 'price': {'secure': 0.5}}
            if method == 'POST':
                posts.append(payload)
                raise TimeoutError('Network response was lost')
            self.fail('Ambiguous create must not trigger another call')
        args = SimpleNamespace(name='uncertain', hours=6, entry='experiments/run.sh')
        with patch.object(cloud, 'api', side_effect=api):
            with self.assertRaises(TimeoutError):
                cloud.allocate(args, wait_seconds=120)
        job = self.state('uncertain')
        self.assertEqual(len(posts), 1)
        self.assertFalse(job['safely_no_pod'])
        self.assertNotIn('released_at', job)
        self.assertEqual(job['state'], 'creating')

    def test_no_capacity_wait_is_bounded_and_reservation_voided(self):
        clock = [0.0]
        def sleep(seconds):
            clock[0] += seconds
        args = SimpleNamespace(name='none', hours=6, entry='experiments/run.sh')
        with patch.object(cloud, 'api', return_value={'availability': 'NONE', 'price': {'secure': 0.5}}), \
             patch.object(cloud.time, 'monotonic', side_effect=lambda: clock[0]), \
             patch.object(cloud.time, 'sleep', side_effect=sleep):
            with self.assertRaisesRegex(RuntimeError, 'bounded allocation'):
                cloud.allocate(args, wait_seconds=120)
        job = self.state('none')
        self.assertLessEqual(clock[0], 120)
        self.assertEqual(job['reserved_usd'], 0)
        self.assertEqual(job['charged_estimate_usd'], 0)
        self.assertIn('released_at', job)

    def test_explicit_gpu_has_no_fallback_when_capacity_is_unavailable(self):
        clock = [0.0]
        def sleep(seconds):
            clock[0] += seconds
        args = SimpleNamespace(name='pinned', hours=1, entry='probe.sh',
                               gpu='NVIDIA GeForce RTX 4090')
        with patch.object(cloud, 'api', return_value={'availability': 'NONE', 'price': {'secure': .89}}) as api, \
             patch.object(cloud.time, 'monotonic', side_effect=lambda: clock[0]), \
             patch.object(cloud.time, 'sleep', side_effect=sleep):
            with self.assertRaisesRegex(RuntimeError, 'bounded allocation'):
                cloud.allocate(args, wait_seconds=120)
        self.assertTrue(api.call_args_list)
        for call in api.call_args_list:
            self.assertIn('NVIDIA%20GeForce%20RTX%204090', call.args[0])
        self.assertEqual(self.state('pinned')['reserved_usd'], 0)

    def test_explicit_gpu_still_enforces_price_ceiling(self):
        clock = [0.0]
        def sleep(seconds):
            clock[0] += seconds
        args = SimpleNamespace(name='expensive', hours=1, entry='probe.sh',
                               gpu='NVIDIA GeForce RTX 4090')
        with patch.object(cloud, 'api', return_value={'availability': 'HIGH', 'price': {'secure': 1.01}}) as api, \
             patch.object(cloud.time, 'monotonic', side_effect=lambda: clock[0]), \
             patch.object(cloud.time, 'sleep', side_effect=sleep):
            with self.assertRaisesRegex(RuntimeError, 'bounded allocation'):
                cloud.allocate(args, wait_seconds=120)
        self.assertTrue(all(len(call.args) == 1 for call in api.call_args_list))
        self.assertEqual(self.state('expensive')['charged_estimate_usd'], 0)

    def test_capacity_rejection_with_reconciliation_failure_stays_reserved(self):
        def api(path, method='GET', payload=None):
            if path.startswith('catalog/'):
                return {'availability': 'LOW', 'price': {'secure': 0.5}}
            if method == 'POST':
                raise RuntimeError('Runpod HTTP 400: no longer available')
            raise TimeoutError('List unavailable')
        args = SimpleNamespace(name='list-failed', hours=6, entry='experiments/run.sh')
        with patch.object(cloud, 'api', side_effect=api):
            with self.assertRaises(TimeoutError):
                cloud.allocate(args, wait_seconds=120)
        self.assertNotIn('released_at', self.state('list-failed'))

    def test_legacy_explicit_capacity_rejection_is_reconciled_and_refunded(self):
        self.job(pod_id=None, state='creating', error='Runpod HTTP 400: GPU is no longer available')
        with patch.object(cloud, 'api', return_value={'pods': [], 'pagination': {'hasNextPage': False}}):
            cloud.guard()
        self.assertEqual(self.state()['charged_estimate_usd'], 0)
        self.assertEqual(self.state()['reserved_usd'], 0)

    def test_rejection_evidence_survives_failed_list_reconciliation(self):
        job = self.job(pod_id=None, state='creating', deadline=time.time() + 3600,
                       heartbeat=time.time() - 601, error='List request timed out')
        cloud.update('a', allocation_attempts=[{'pod_name': job['pod_name'],
            'outcome': 'capacity_rejected', 'error': 'Runpod HTTP 400: no longer available'}])
        with patch.object(cloud, 'api', return_value={'pods': [], 'pagination': {'hasNextPage': False}}):
            cloud.guard()
        self.assertEqual(self.state()['charged_estimate_usd'], 0)

    def test_guard_does_not_refund_active_rejection_retry(self):
        self.job(pod_id=None, state='creating', deadline=time.time() + 3600,
                 error='Runpod HTTP 400: GPU is no longer available', allocation_attempts=[{}])
        with patch.object(cloud, 'api', return_value={'pods': [], 'pagination': {'hasNextPage': False}}):
            cloud.guard()
        self.assertNotIn('released_at', self.state())

    def test_delayed_delete_prevents_exceeding_capacity_without_killing_existing(self):
        for i in range(cloud.MAX_LIVE_PODS-1):
            self.job(f'running-{i}')
        self.job('deleting', state='delete_requested', delete_requested_at=time.time())
        calls = []
        clock = [0.0]
        def api(path, method='GET', payload=None):
            calls.append((path, method))
            self.assertEqual((path, method), ('pods/pod-deleting', 'GET'))
            return {'status': 'RUNNING'}
        def sleep(seconds):
            clock[0] += seconds
        args = SimpleNamespace(name='fourth', hours=6, entry='experiments/run.sh')
        with patch.object(cloud, 'api', side_effect=api), \
             patch.object(cloud.time, 'monotonic', side_effect=lambda: clock[0]), \
             patch.object(cloud.time, 'sleep', side_effect=sleep):
            with self.assertRaisesRegex(RuntimeError, 'bounded allocation'):
                cloud.allocate(args, wait_seconds=120)
        self.assertTrue(calls)
        self.assertEqual(self.state('fourth')['allocation_wait_reason'], 'concurrency_limit')
        self.assertNotIn('released_at', self.state('deleting'))
        self.assertEqual(self.state('deleting')['reserved_usd'], 2)
        self.assertEqual(self.state('fourth')['charged_estimate_usd'], 0)

    def test_confirmed_deletion_frees_exactly_one_slot(self):
        for i in range(cloud.MAX_LIVE_PODS-1):
            self.job(f'running-{i}')
        self.job('deleting', state='delete_requested', delete_requested_at=time.time())
        calls = []
        def api(path, method='GET', payload=None):
            calls.append((path, method))
            if path == 'pods/pod-deleting':
                return {'status': 'TERMINATED'}
            if path.startswith('catalog/'):
                return {'availability': 'LOW', 'price': {'secure': .82}}
            if (path, method) == ('pods', 'POST'):
                return {'id': 'replacement', 'name': payload['name'], 'cost': .82}
            self.fail('Unexpected API call: ' + str((path, method)))
        args = SimpleNamespace(name='replacement', hours=6, entry='experiments/run.sh')
        with patch.object(cloud, 'api', side_effect=api):
            pod = cloud.allocate(args, wait_seconds=120)
        self.assertEqual(pod['id'], 'replacement')
        self.assertIn('released_at', self.state('deleting'))
        with cloud.ledger() as state:
            self.assertEqual(cloud.potentially_live_pods(state['jobs']), cloud.MAX_LIVE_PODS)
        self.assertFalse(any(method == 'DELETE' for _, method in calls))

    def test_ambiguous_create_occupies_slot_waiting_does_not(self):
        jobs = {
            'one': {'pod_id': 'p1'}, 'two': {'pod_id': 'p2'},
            'ambiguous': {'state': 'creating', 'pod_name': 'unique', 'safely_no_pod': False},
            'waiting': {'state': 'allocation_waiting', 'safely_no_pod': True},
            'released': {'pod_id': 'gone', 'released_at': 1},
        }
        self.assertEqual(cloud.potentially_live_pods(jobs), 3)
        self.assertEqual(cloud.potentially_live_pods(jobs, exclude='ambiguous'), 2)
        jobs['ambiguous']['duplicate_pod_ids'] = ['uncertain1', 'uncertain2']
        self.assertEqual(cloud.potentially_live_pods(jobs), 4)

    def ambiguous_job(self):
        return self.job(pod_id=None, state='creating', created_at=900, heartbeat=900,
                        deadline=99999, error='Runpod HTTP 504: upstream gateway timeout',
                        safely_no_pod=False)

    def full_list(self, pods=None):
        return {'pods': pods or [], 'pagination': {'hasNextPage': False}}

    def test_one_full_absence_does_not_refund_uncertain_create(self):
        self.ambiguous_job()
        with patch.object(cloud, 'api', return_value=self.full_list()), patch.object(cloud.time, 'time', return_value=1000):
            cloud.guard_job('a', self.state())
        job = self.state()
        self.assertEqual(job['absence_count'], 1)
        self.assertEqual(job['first_absence_at'], 1000)
        self.assertNotIn('released_at', job)
        self.assertEqual(job['reserved_usd'], 2)

    def test_twelve_complete_absences_over_five_minutes_void_without_retry(self):
        self.ambiguous_job()
        with patch.object(cloud, 'api', return_value=self.full_list()) as api:
            for index in range(12):
                with patch.object(cloud.time, 'time', return_value=1000 + index * 30):
                    cloud.guard_job('a', self.state())
        job = self.state()
        self.assertEqual(job['state'], 'create_absent_reconciled')
        self.assertEqual(job['absence_count'], 12)
        self.assertEqual(job['absence_reconciliation']['elapsed_seconds'], 330)
        self.assertEqual(job['reserved_usd'], 0)
        self.assertEqual(job['charged_estimate_usd'], 0)
        self.assertEqual(job['late_arrival_watch_until'], 1330 + 86400)
        self.assertEqual(len(job['absence_observations']), 12)
        self.assertTrue(all(call.args == ('pods',) for call in api.call_args_list))

    def test_absence_requires_both_count_and_elapsed_time(self):
        self.ambiguous_job()
        with patch.object(cloud, 'api', return_value=self.full_list()):
            for index in range(12):
                with patch.object(cloud.time, 'time', return_value=1000 + index):
                    cloud.guard_job('a', self.state())
            self.assertNotIn('released_at', self.state())
            with patch.object(cloud.time, 'time', return_value=1300):
                cloud.guard_job('a', self.state())
        self.assertEqual(self.state()['state'], 'create_absent_reconciled')

    def test_read_error_and_partial_page_reset_absence_evidence(self):
        self.ambiguous_job()
        with patch.object(cloud, 'api', return_value=self.full_list()), patch.object(cloud.time, 'time', return_value=1000):
            cloud.guard_job('a', self.state())
        with patch.object(cloud, 'api', side_effect=TimeoutError('read failed')):
            cloud.guard_job('a', self.state())
        self.assertEqual(self.state()['absence_count'], 0)
        self.assertIsNone(self.state()['first_absence_at'])
        with patch.object(cloud, 'api', return_value=self.full_list()), patch.object(cloud.time, 'time', return_value=2000):
            cloud.guard_job('a', self.state())
        self.assertEqual(self.state()['first_absence_at'], 2000)
        with patch.object(cloud, 'api', return_value={'pods': [], 'pagination': {'hasNextPage': True}}):
            cloud.guard_job('a', self.state())
        self.assertEqual(self.state()['absence_count'], 0)
        self.assertNotIn('released_at', self.state())
        self.assertEqual(self.state()['reserved_usd'], 2)

    def test_listing_without_authoritative_pagination_is_not_absence(self):
        for response in [[], {'pods': []}, {'pods': [], 'pagination': {'hasNextPage': True}}]:
            with patch.object(cloud, 'api', return_value=response):
                with self.assertRaisesRegex(RuntimeError, 'pagination|completeness|partial'):
                    cloud.list_pods()

    def test_late_duplicate_arrivals_all_cleaned_before_financial_release(self):
        job = self.job(pod_id=None, state='create_absent_reconciled', created_at=900,
                       released_at=1000, charged_estimate_usd=0, reserved_usd=0,
                       absence_reserved_usd=2, late_arrival_watch_until=90000)
        listed = [{'id': value, 'name': job['pod_name'], 'status': 'RUNNING'} for value in ['one', 'two']]
        listed.append({'id': 'unrelated', 'name': job['pod_name'] + '-different', 'status': 'RUNNING'})
        deleted = set()
        fail_two = [True]
        calls = []
        def api(path, method='GET', payload=None):
            calls.append((path, method))
            if path == 'pods':
                return self.full_list(listed)
            identifier = path.split('/')[-1]
            if method == 'DELETE':
                if identifier == 'two' and fail_two[0]:
                    raise RuntimeError('delete temporarily unavailable')
                deleted.add(identifier)
                return {}
            return {'id': identifier, 'name': job['pod_name'],
                    'status': 'TERMINATED' if identifier in deleted else 'RUNNING'}
        with patch.object(cloud, 'api', side_effect=api), patch.object(cloud.time, 'time', return_value=2000):
            cloud.guard_job('a', self.state())
        pending = self.state()
        self.assertNotIn('released_at', pending)
        self.assertEqual(pending['state'], 'late_arrival_cleanup')
        self.assertGreater(pending['reserved_usd'], 0)
        self.assertIn('confirmed_released_at', pending['late_arrival_pods']['one'])
        self.assertNotIn('confirmed_released_at', pending['late_arrival_pods']['two'])
        fail_two[0] = False
        with patch.object(cloud, 'api', side_effect=api), patch.object(cloud.time, 'time', return_value=2060):
            cloud.guard_job('a', self.state())
        released = self.state()
        self.assertEqual(released['state'], 'late_arrival_cleaned')
        self.assertGreater(released['charged_estimate_usd'], 0)
        self.assertTrue(all(item.get('confirmed_released_at') for item in released['late_arrival_pods'].values()))
        self.assertFalse(any('unrelated' in path or method == 'POST' for path, method in calls))

    def test_late_arrival_identity_mismatch_refuses_delete(self):
        job = self.job(pod_id=None, state='create_absent_reconciled', released_at=1000,
                       charged_estimate_usd=0, reserved_usd=0, absence_reserved_usd=2,
                       late_arrival_watch_until=90000)
        def api(path, method='GET', payload=None):
            if path == 'pods':
                return self.full_list([{'id': 'late', 'name': job['pod_name']}])
            self.assertEqual(method, 'GET')
            return {'id': 'late', 'name': 'foreign-name', 'status': 'RUNNING'}
        with patch.object(cloud, 'api', side_effect=api), patch.object(cloud.time, 'time', return_value=2000):
            cloud.guard_job('a', self.state())
        self.assertNotIn('released_at', self.state())
        self.assertIn('identity mismatch', self.state()['late_arrival_pods']['late']['last_error'])

    def test_expired_late_watch_without_pending_resources_makes_no_api_calls(self):
        self.job(pod_id=None, state='create_absent_reconciled', released_at=1000,
                 charged_estimate_usd=0, reserved_usd=0, late_arrival_watch_until=2000)
        with patch.object(cloud, 'api') as api, patch.object(cloud.time, 'time', return_value=2001):
            cloud.guard_job('a', self.state())
        api.assert_not_called()

    def test_background_launcher_releases_capture_pipes_immediately(self):
        base = Path(self.temp.name)
        (base / 'output').mkdir()
        (base / 'worker.sh').write_text('sleep 10\n')
        started = time.monotonic()
        result = subprocess.run(['bash', '-c', cloud.launch_command('worker.sh', 20, str(base))],
                                capture_output=True, text=True, timeout=2)
        elapsed = time.monotonic() - started
        pid = int((base / 'output/worker.pid').read_text())
        try:
            self.assertEqual(result.returncode, 0)
            self.assertLess(elapsed, 2)
        finally:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    def test_finish_trap_preserves_failure_and_publishes_final_snapshot(self):
        script = Path(__file__).with_name('setup_gpu.sh').read_text()
        finish = script[script.index('finish() {'):script.index('nvidia-smi')]
        base = cloud.STATE / 'a'
        (base / 'output').mkdir(parents=True)
        result = subprocess.run(['bash', '-c', 'set -euo pipefail\n' + finish + '\nprintf checkpoint > output/weights.bin\nexit 7\n'], cwd=base)
        self.assertEqual(result.returncode, 7)
        self.assertEqual(cloud.verify_backup('a')['exit_code'], 7)

    def test_finish_trap_records_term_as_failure(self):
        script = Path(__file__).with_name('setup_gpu.sh').read_text()
        finish = script[script.index('finish() {'):script.index('nvidia-smi')]
        base = cloud.STATE / 'a'
        (base / 'output').mkdir(parents=True)
        result = subprocess.run(['bash', '-c', 'set -euo pipefail\n' + finish + '\nkill -TERM $$\n'], cwd=base)
        self.assertEqual(result.returncode, 143)
        self.assertEqual(cloud.verify_backup('a')['exit_code'], 143)


if __name__ == '__main__':
    unittest.main()
