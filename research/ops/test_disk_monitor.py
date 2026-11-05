import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import disk_monitor
import worker_disk_check as worker


class DiskTests(unittest.TestCase):
    def test_thresholds_cover_bytes_percent_and_inodes(self):
        healthy = {'used_percent': 62, 'available_bytes': 12 * worker.GIB, 'inode_used_percent': 10}
        self.assertEqual(worker.pressure(healthy), 'healthy')
        for change in [{'used_percent': 85}, {'available_bytes': 3 * worker.GIB}, {'inode_used_percent': 86}]:
            self.assertEqual(worker.pressure({**healthy, **change}), 'high')
        for change in [{'used_percent': 95}, {'available_bytes': worker.GIB - 1}, {'inode_used_percent': 99}]:
            self.assertEqual(worker.pressure({**healthy, **change}), 'critical')

    def test_only_owned_identified_unreleased_workers_selected(self):
        live = {'pod_id': 'p', 'identity_confirmed': True}
        rows = {'opd-isolated-v2-seed17-a': live,
                'unrelated-experiment': live,
                'replay-selection-seed29-a': {**live, 'released_at': 1},
                'adaptive-isolated-v2-seed17-a': {'pod_id': 'p'},
                'teacher-matched-isolated-v1-seed17-a': live}
        self.assertEqual(set(disk_monitor.selected(rows)), {'opd-isolated-v2-seed17-a', 'teacher-matched-isolated-v1-seed17-a'})

    def test_cache_links_mappings_and_installs_block_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);cache = root/'cache';venv = root/'venv';proc = root/'proc'
            for p in [cache, venv, proc/'123'/'fd']:p.mkdir(parents=True)
            (proc/'123'/'cmdline').write_bytes(b'python\0training.py\0')
            (proc/'123'/'maps').write_text('')
            self.assertIsNone(worker.cache_blocker(cache, venv, proc))
            (venv/'package').symlink_to(cache, target_is_directory=True)
            self.assertEqual(worker.cache_blocker(cache, venv, proc), 'installed_environment_links_to_cache')
            (venv/'package').unlink()
            (proc/'123'/'maps').write_text('mapped '+str(cache)+'/lib.so')
            self.assertEqual(worker.cache_blocker(cache, venv, proc), 'cache_memory_mapped')
            (proc/'123'/'maps').write_text('')
            (proc/'123'/'cmdline').write_bytes(b'/workspace/project/uv\0pip\0install\0')
            self.assertEqual(worker.cache_blocker(cache, venv, proc), 'package_manager_active')

    def test_wrong_provider_identity_never_runs_remote_command(self):
        with patch.object(disk_monitor.cloud, 'api', return_value={'id':'p','name':'someone-else'}), \
                patch.object(disk_monitor.cloud, 'remote') as remote:
            with self.assertRaises(ValueError):
                disk_monitor.inspect('job', {'pod_id':'p','pod_name':'owned'})
            remote.assert_not_called()

    def test_healthy_disk_does_not_clean(self):
        healthy = {'used_percent': 62, 'available_bytes': 12 * worker.GIB, 'inode_used_percent': 10}
        with patch.object(worker, 'usage', return_value=healthy), \
                patch.object(worker, 'cache_blocker') as blocker, \
                patch.object(worker.subprocess, 'run') as run, patch('builtins.print'):
            worker.main()
            blocker.assert_not_called();run.assert_not_called()

    def test_monitor_stops_only_after_all_campaigns_finish(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(disk_monitor, 'ROOT', Path(directory)):
            root = Path(directory)/'lab/experiments'
            for name in ['replay-selection-v1','adaptive-replay-isolated-v2','on-policy-distillation-isolated-v2']:
                p = root/name/'campaign.json';p.parent.mkdir(parents=True)
                p.write_text(json.dumps({'state':'development_complete'}))
            teacher=root/'teacher-generation/isolated-pipeline-20261025/pipeline.json'
            teacher.parent.mkdir(parents=True)
            for phase in ['queued_gpu','needs_inspection','development_complete','complete_insufficient_matched_yield']:
                teacher.write_text(json.dumps({'phase':phase}))
                self.assertEqual(disk_monitor.campaigns_complete(),phase in ['development_complete','complete_insufficient_matched_yield'])

    def test_completed_pilots_do_not_stop_pending_full_evaluation_monitor(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(disk_monitor, 'ROOT', Path(directory)):
            root = Path(directory)/'lab/experiments'
            summary = root/'completion-20261026/summary.json'
            summary.parent.mkdir(parents=True)
            summary.write_text(json.dumps({'complete': True}))
            full = root/'teacher-full-20261026/campaign.json'
            full.parent.mkdir(parents=True)
            full.write_text(json.dumps({'state': 'running'}))
            self.assertFalse(disk_monitor.campaigns_complete())
            full.write_text(json.dumps({'state': 'development_complete'}))
            self.assertTrue(disk_monitor.campaigns_complete())
            live = {'pod_id': 'p', 'identity_confirmed': True}
            self.assertEqual(set(disk_monitor.selected({'teacher-full-v1-seed17-a': live})),
                             {'teacher-full-v1-seed17-a'})


if __name__ == '__main__':
    unittest.main()
