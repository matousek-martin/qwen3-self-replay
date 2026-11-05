#!/usr/bin/env python3
"""Run existing scientific and lifecycle regression suites in a fresh workspace."""

import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from self_replay.artifacts import write_json
from self_replay.methods.runtime import environment, initialize
from self_replay.paths import repository

modules = [
    "experiments.rl_training.test_objectives",
    "experiments.rl_training.test_runtime",
    "experiments.rl_training.test_replay_selection",
    "experiments.rl_training.test_adaptive_replay",
    "experiments.data_pipeline.test_pipeline",
    "experiments.replay_scaled.test_core",
    "experiments.replay_scaled.test_generation",
    "experiments.test_baseline_cache",
    "experiments.shorterbetter.test_objective",
    "experiments.coding_abc.test_abc",
    "experiments.coding_curriculum.test_curriculum",
    "experiments.test_gpu_policy",
    "ops.test_vastai",
]
cloud_cases = [
    "watch_resumes_until_release_without_allocating_or_launching",
    "watch_released_job_is_noop",
    "watch_rejects_wrong_provider_identity",
    "watch_controller_error_does_not_stop_worker",
    "fixed_job_envelope_never_renews_past_hard_deadline",
    "empty_manifest_is_not_success",
    "partial_snapshot_rejected",
    "marker_must_match_hashed_result",
    "valid_snapshot_verified",
    "delete_is_not_marked_released_until_confirmed",
    "ambiguous_create_occupies_slot_waiting_does_not",
    "one_full_absence_does_not_refund_uncertain_create",
    "twelve_complete_absences_over_five_minutes_void_without_retry",
    "absence_requires_both_count_and_elapsed_time",
    "read_error_and_partial_page_reset_absence_evidence",
]
modules += ["experiments.test_cloud.CloudTests.test_" + name for name in cloud_cases]
with tempfile.TemporaryDirectory(prefix="qsr-science-") as directory:
    workspace = initialize(Path(directory) / "workspace", "rl-replay")
    result = subprocess.run(
        [sys.executable, "-m", "unittest", *modules],
        cwd=workspace,
        env=environment(workspace),
        text=True,
        capture_output=True,
    )
    out = repository() / "artifacts/verification"
    out.mkdir(parents=True, exist_ok=True)
    (out / "science-tests.log").write_text(result.stdout + result.stderr)
    write_json(
        out / "science-tests.json",
        {
            "command": ["python", "-m", "unittest", *modules],
            "exit_code": result.returncode,
            "paid_computation": False,
        },
    )
    print(result.stderr[-14000:])
    raise SystemExit(result.returncode)
