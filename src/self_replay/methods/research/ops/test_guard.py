import unittest
from guard_run import should_stop


class GuardDecisions(unittest.TestCase):
    def test_active_controller_owns_cleanup(self):
        self.assertFalse(should_stop(True, "RUNNING", {"exit_code": 0, "finished_at": "now"}))

    def test_running_job_is_never_timed_out(self):
        self.assertFalse(should_stop(False, "RUNNING", None))

    def test_finished_job_releases_compute(self):
        for code in (0, 1):
            self.assertTrue(should_stop(False, "RUNNING", {"exit_code": code, "finished_at": "now"}))

    def test_incomplete_exit_marker_does_not_stop_job(self):
        self.assertFalse(should_stop(False, "RUNNING", {"phase": "baseline"}))

    def test_stopped_pod_is_left_alone(self):
        self.assertFalse(should_stop(False, "EXITED", {"exit_code": 0, "finished_at": "now"}))


if __name__ == "__main__":
    unittest.main()
