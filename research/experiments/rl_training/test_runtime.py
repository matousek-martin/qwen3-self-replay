import json
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

from experiments.rl_training.runtime import RuntimeLimit
from experiments.rl_training.train import make_parser
from experiments.offline_training import common
from experiments import study_train


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class RuntimeTests(unittest.TestCase):
    def test_zero_has_no_elapsed_deadline_but_positive_expiry_is_unchanged(self):
        clock = Clock()
        complete = RuntimeLimit(0, clock=clock)
        bounded = RuntimeLimit(10, clock=clock)
        clock.now = 109.99
        self.assertFalse(bounded.expired)
        clock.now = 110
        self.assertTrue(bounded.expired)
        clock.now = 10**9
        self.assertFalse(complete.expired)

    def test_sigterm_still_stops_completion_driven_rl(self):
        previous = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
        try:
            runtime = RuntimeLimit(0)
            runtime.install_signal_handlers()
            signal.raise_signal(signal.SIGTERM)
            self.assertTrue(runtime.expired)
            self.assertEqual(runtime.reason, "signal")
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)

    def test_rollout_deadline_zero_and_real_sigterm(self):
        previous = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
        clock = Clock()
        try:
            with patch.object(common.time, "monotonic", clock):
                deadline = common.Deadline(0)
                clock.now += 10**9
                self.assertFalse(deadline.expired)
                signal.raise_signal(signal.SIGTERM)
                self.assertTrue(deadline.expired)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)

    def test_positive_rollout_deadline(self):
        previous = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
        clock = Clock()
        try:
            with patch.object(common.time, "monotonic", clock):
                deadline = common.Deadline(10)
                clock.now += 9
                self.assertFalse(deadline.expired)
                clock.now += 1
                self.assertTrue(deadline.expired)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)

    def test_control_file_disables_then_reenables_elapsed_limit(self):
        clock = Clock()
        with tempfile.TemporaryDirectory() as directory:
            control = Path(directory) / "control.json"
            runtime = RuntimeLimit(10, control, clock=clock)
            clock.now += 9
            control.write_text(json.dumps({"max_seconds": 0}))
            self.assertFalse(runtime.expired)
            clock.now += 100
            self.assertFalse(runtime.expired)
            control.write_text(json.dumps({"max_seconds": 200}))
            clock.now += 2
            self.assertFalse(runtime.expired)
            clock.now += 100
            self.assertTrue(runtime.expired)
            self.assertEqual(runtime.summary()["accepted_control_updates"], 2)

    def test_malformed_update_retains_last_policy_and_stop_is_sticky(self):
        clock = Clock()
        with tempfile.TemporaryDirectory() as directory:
            control = Path(directory) / "control.json"
            runtime = RuntimeLimit(10, control, clock=clock)
            control.write_text('{"max_seconds":0,"stop":"false"}')
            self.assertFalse(runtime.expired)
            self.assertEqual(runtime.seconds, 10)
            self.assertIsNotNone(runtime.last_error)
            control.write_text('{"max_seconds":0,"stop":true}')
            clock.now += 2
            self.assertTrue(runtime.expired)
            self.assertEqual(runtime.reason, "runtime_control")
            control.write_text('{"stop":false}')
            clock.now += 2
            self.assertTrue(runtime.expired)
            self.assertEqual(runtime.seconds, 0)

    def test_no_file_keeps_default_and_invalid_limits_rejected(self):
        runtime = RuntimeLimit(0, "/nonexistent-thinkingcap-control.json")
        self.assertFalse(runtime.expired)
        for value in (-1, float("nan"), float("inf"), True, "0"):
            with self.assertRaises(ValueError):
                RuntimeLimit(value)
        args = make_parser().parse_args(["--data", "x", "--out", "y"])
        self.assertEqual(args.max_seconds, 0)

    def test_study_arm_requests_completion_driven_training(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            target = out / "training/grpo"
            target.mkdir(parents=True)
            (target / "status.json").write_text(json.dumps({"status": "completed", "completed_groups": 128, "optimizer_updates": 2}))
            with patch.object(study_train, "OUT", out), patch.object(study_train, "run") as run, \
                 patch.object(study_train, "validate"), patch.object(study_train, "write"), \
                 patch("sys.argv", ["study_train", "grpo"]):
                study_train.main()
            command = run.call_args.args
            self.assertEqual(command[command.index("--max-seconds") + 1], 0)
            self.assertEqual(command[command.index("--max-steps") + 1], 128)


if __name__ == "__main__":
    unittest.main()
