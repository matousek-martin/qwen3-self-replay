#!/usr/bin/env python3
"""Synthetic CPU loss/gradient/distributed checks; no submitted-model training."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from self_replay.artifacts import write_json
from self_replay.methods.runtime import environment, initialize
from self_replay.paths import repository

modules = [
    "experiments.rl_training.test_torch",
    "experiments.rl_training.test_replay_integration",
    "experiments.replay_scaled.test_distributed",
    "experiments.on_policy_distillation.test_loss",
    "experiments.offline_training.test_offline",
]
with tempfile.TemporaryDirectory(prefix="qsr-numerics-") as directory:
    workspace = initialize(Path(directory) / "workspace", "rl-replay")
    # Historical CLI test needs a row even in --dry-run; use an explicit synthetic fixture.
    (workspace / "lab/experiments/data/pilot.jsonl").write_text(
        json.dumps(
            {
                "id": "synthetic-numerical-test",
                "source": "gsm8k",
                "split": "train",
                "task": "math",
                "prompt": "What is 1 + 1?",
                "answer": "2",
            }
        )
        + "\n"
    )
    result = subprocess.run(
        [sys.executable, "-m", "unittest", *modules],
        cwd=workspace,
        env=environment(workspace),
        text=True,
        capture_output=True,
    )
    output = repository() / "artifacts/verification"
    output.mkdir(parents=True, exist_ok=True)
    (output / "numerics-tests.log").write_text(result.stdout + result.stderr)
    write_json(
        output / "numerics-tests.json",
        {
            "modules": modules,
            "exit_code": result.returncode,
            "device": "cpu",
            "submitted_model_training": False,
        },
    )
    print(result.stderr[-14000:])
    raise SystemExit(result.returncode)
