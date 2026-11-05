import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from experiments import replay_selection
from experiments.replay_selection import ROOT, load_protocol, training_command
from experiments.rl_training.train import make_parser
from experiments.replay_round2 import training_command as historical_command


class ProtocolTests(unittest.TestCase):
    def test_all_six_commands_dry_run_and_only_admission_differs(self):
        protocol = load_protocol()
        for seed in protocol['training_seeds']:
            configs = []
            for arm in protocol['arms']:
                command = training_command(protocol, arm, seed, ROOT / 'lab/experiments/data', '/tmp/replay-selection-dry-run')
                result = subprocess.run(command + ['--dry-run'], capture_output=True, text=True, check=True)
                config = json.loads(result.stdout)
                self.assertEqual(config['replay_admission'], protocol['arms'][arm])
                self.assertEqual(config['revision'], protocol['revision'])
                self.assertEqual(config['max_steps'], 128)
                self.assertFalse(config['scale_rewards_resolved'])
                for key in ('out', 'replay_admission'): config.pop(key)
                configs.append(config)
            self.assertEqual(*configs)

    def test_round_two_control_recipe_preserved(self):
        protocol = load_protocol()
        for seed in protocol['training_seeds']:
            old = historical_command({'seed':seed, 'length_coefficient':.1,'replay_weight':.1}, Path('data'), Path('out/control-seed'+str(seed)))
            new = training_command(protocol, 'control', seed, 'data', 'out')
            old_args = vars(make_parser().parse_args(old[3:]))
            new_args = vars(make_parser().parse_args(new[3:]))
            self.assertEqual(old_args.pop('scale_rewards'), 'auto')
            self.assertEqual(new_args.pop('scale_rewards'), 'no')
            self.assertEqual(old_args, new_args)

    def test_unregistered_trial_rejected(self):
        protocol = load_protocol()
        for arm, seed in [('control', 1), ('adaptive', 17)]:
            with self.assertRaises(ValueError): training_command(protocol, arm, seed, 'data', 'out')

    def test_registered_protocol_cannot_be_silently_changed(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'protocol.json'
            target.write_bytes(replay_selection.PROTOCOL.read_bytes() + b'\n')
            target.with_name('registration.json').write_bytes(
                replay_selection.PROTOCOL.with_name('registration.json').read_bytes())
            with patch.object(replay_selection, 'PROTOCOL', target):
                with self.assertRaisesRegex(ValueError, 'changed after registration'):
                    load_protocol()

    def test_nonreplay_cli_cannot_silently_ignore_new_admission(self):
        command = training_command(load_protocol(), 'mixed', 17, ROOT / 'lab/experiments/data', '/tmp/unused')
        command[command.index('--variant') + 1] = 'fixed'
        result = subprocess.run(command + ['--dry-run'], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('--replay-admission requires --variant replay', result.stderr)


if __name__ == '__main__':
    unittest.main()
