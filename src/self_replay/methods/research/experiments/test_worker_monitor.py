"""No-cloud tests for completion-driven monitoring and verified signal delivery."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

PATH = Path(__file__).with_name('worker_monitor.py').resolve()
spec = importlib.util.spec_from_file_location('worker_monitor', PATH)
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)


class MonitorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / 'output'
        self.output.mkdir()

    def wait_state(self, predicate, timeout=5):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            path = self.output / 'worker-state.json'
            if path.exists():
                state = json.loads(path.read_text())
                if predicate(state):
                    return state
            time.sleep(.02)
        self.fail('Supervisor did not reach expected state')

    def fake_proc(self, pid=123, ticks=456, state='S'):
        proc = self.root / 'proc'
        (proc / str(pid)).mkdir(parents=True)
        (proc / 'sys/kernel/random').mkdir(parents=True)
        (proc / 'sys/kernel/random/boot_id').write_text('boot-test')
        fields = [state] + ['0'] * 18 + [str(ticks)] + ['0'] * 5
        (proc / str(pid) / 'stat').write_text(f'{pid} (name with ) spaces) ' + ' '.join(fields))
        return proc

    def test_process_identity_handles_comm_parentheses_and_pid_reuse(self):
        proc = self.fake_proc()
        expected = {'pid': 123, 'start_ticks': 456, 'boot_id': 'boot-test'}
        with patch.object(monitor, 'PROC', proc):
            self.assertEqual(monitor.process_observation(123, expected)['status'], 'alive')
            self.assertEqual(monitor.process_observation(123, dict(expected, start_ticks=457))['reason'], 'pid_identity_changed')
            self.assertEqual(monitor.process_observation(124, expected)['status'], 'dead')

    def test_process_read_errors_mean_unknown_not_dead(self):
        with patch.object(Path, 'read_text', side_effect=PermissionError('denied')):
            self.assertEqual(monitor.process_observation(123)['status'], 'unknown')
        proc = self.fake_proc()
        (proc / 'sys/kernel/random/boot_id').unlink()
        with patch.object(monitor, 'PROC', proc):
            self.assertEqual(monitor.process_observation(123)['status'], 'unknown')

    def test_progress_ignores_housekeeping_and_heartbeats(self):
        (self.output / 'training').mkdir()
        meaningful = self.output / 'training/metrics.jsonl'
        meaningful.write_text('{"step":1}\n')
        os.utime(meaningful, (100, 100))
        for name in ['worker-state.json', 'worker.pid', 'SHA256SUMS', 'heartbeat.json', 'exit.json', 'result.json', 'monitor.log']:
            (self.output / name).write_text('housekeeping')
        progress = monitor.meaningful_progress(self.output)
        self.assertEqual(progress['latest']['path'], 'training/metrics.jsonl')
        self.assertEqual(progress['latest']['modified_unix'], 100)

    def test_legacy_pid_is_observed_but_never_signaled_without_identity(self):
        (self.output / 'worker.pid').write_text(str(os.getpid()))
        probe = monitor.probe_worker(self.output, include_gpu=False)
        self.assertEqual(probe['mode'], 'legacy')
        self.assertEqual(probe['worker_status'], 'alive')
        self.assertFalse(probe['worker']['identity_verified'])
        self.assertFalse(monitor.stop_worker(self.output)['ok'])

    def test_missing_or_corrupt_state_is_unknown(self):
        self.assertEqual(monitor.probe_worker(self.output, include_gpu=False)['worker_status'], 'unknown')
        (self.output / 'worker-state.json').write_text('{partial')
        (self.output / 'worker.pid').write_text(str(os.getpid()))
        # A corrupt authoritative identity must not fall back to an unverified PID.
        self.assertEqual(monitor.probe_worker(self.output, include_gpu=False)['worker_status'], 'unknown')

    def sample_probe(self, observed=2000, gpu_busy=False):
        return {'observed_unix': observed, 'worker': {'status': 'alive', 'pid': 1, 'start_ticks': 2, 'boot_id': 'b'},
                'supervisor': {'status': 'alive'}, 'supervisor_state': 'running', 'mode': 'supervised',
                'exit_marker': {'status': 'absent'}, 'started_unix': 0,
                'progress': {'latest': {'path': 'runner.log', 'modified_unix': 0, 'bytes': 12}},
                'gpu': {'devices': [{'utilization_percent': 90 if gpu_busy else 0}]}}

    def test_idle_needs_second_probe_and_never_authorizes_kill(self):
        first = self.sample_probe()
        first['health'] = monitor.assess(first)
        self.assertEqual(first['health']['status'], 'idle_suspect')
        second = self.sample_probe(observed=2070)
        result = monitor.assess(second, first)
        self.assertEqual(result['status'], 'idle_suspect_confirmed')
        self.assertFalse(result['idle_authorizes_termination'])
        self.assertFalse(result['worker_exit_confirmed'])
        second['gpu']['devices'][0]['utilization_percent'] = 90
        self.assertEqual(monitor.assess(second, first)['status'], 'busy_without_recent_progress')
        second['worker']['start_ticks'] = 3
        self.assertFalse(monitor.assess(second, first)['repeated_idle_suspect'])

    def test_dead_worker_without_marker_identified(self):
        probe = self.sample_probe()
        probe['worker']['status'] = 'dead'
        health = monitor.assess(probe)
        self.assertEqual(health['status'], 'worker_exited_without_exit_marker')
        self.assertTrue(health['worker_exit_confirmed'])

    def test_supervisor_records_completion_and_reaps_child(self):
        entry = self.root / 'worker.sh'
        entry.write_text('echo progress\nprintf \'{"exit_code":7}\\n\' > output/exit.json\nexit 7\n')
        result = subprocess.run([sys.executable, PATH, 'run', '--entry', entry, '--output', self.output,
                                 '--cwd', self.root], capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 7, result.stderr)
        state = json.loads((self.output / 'worker-state.json').read_text())
        self.assertEqual(state['state'], 'finished')
        self.assertEqual(state['exit_code'], 7)
        self.assertIsNone(state['runtime_deadline'])
        self.assertFalse(Path('/proc', str(state['worker']['pid'])).exists())
        probe = monitor.probe_worker(self.output, include_gpu=False)
        self.assertEqual(probe['health']['status'], 'finished')

    @unittest.skipUnless(hasattr(os, 'pidfd_open') and hasattr(signal, 'pidfd_send_signal'), 'Linux pidfd required')
    def test_verified_term_is_forwarded_and_child_reaped(self):
        entry = self.root / 'worker.sh'
        entry.write_text("trap 'exit 0' TERM\nwhile :; do sleep 0.1; done\n")
        process = subprocess.Popen([sys.executable, PATH, 'run', '--entry', entry, '--output', self.output,
                                    '--cwd', self.root], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            state = self.wait_state(lambda value: value['state'] == 'running')
            self.assertEqual(monitor.probe_worker(self.output, include_gpu=False)['worker_status'], 'alive')
            result = monitor.stop_worker(self.output)
            self.assertTrue(result['ok'], result)
            self.assertEqual(result['status'], 'term_forward_requested')
            process.communicate(timeout=5)
            final = json.loads((self.output / 'worker-state.json').read_text())
            self.assertEqual(final['state'], 'finished')
            self.assertIn(signal.SIGTERM, final['forwarded_signals'])
            self.assertFalse(Path('/proc', str(state['worker']['pid'])).exists())
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=5)
                if 'state' in locals():
                    try:
                        os.killpg(state['worker']['pid'], signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    def test_stop_rechecks_identity_after_pidfd_open(self):
        state = {'schema_version': 1, 'state': 'running', 'supervisor': {'pid': 123, 'start_ticks': 4, 'boot_id': 'b'}}
        (self.output / 'worker-state.json').write_text(json.dumps(state))
        observations = [{'status': 'alive', 'identity_verified': True}, {'status': 'dead', 'identity_verified': True}]
        with patch.object(monitor, 'process_observation', side_effect=observations), \
             patch.object(os, 'pidfd_open', return_value=99), patch.object(os, 'close'), \
             patch.object(signal, 'pidfd_send_signal') as send:
            result = monitor.stop_worker(self.output)
        self.assertFalse(result['ok'])
        send.assert_not_called()

    def test_probe_directory_cli_is_json_and_read_only(self):
        result = subprocess.run([sys.executable, PATH, 'probe', '--directory', self.output, '--no-gpu'],
                                capture_output=True, text=True, check=True, timeout=5)
        self.assertEqual(json.loads(result.stdout)['worker_status'], 'unknown')
        self.assertEqual(list(self.output.iterdir()), [])


if __name__ == '__main__':
    unittest.main()
