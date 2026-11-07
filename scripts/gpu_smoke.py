#!/usr/bin/env python3
"""Exercise the public training and evaluation interfaces on a leased GPU worker.

Technical checks only: four synthetic training questions and the official smoke
cohort. This script never allocates a GPU and never runs a full baseline.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from self_replay.artifacts import sha256, write_json
from self_replay.data.prepare import stage_frozen
from self_replay.methods.runtime import environment, initialize
from self_replay.paths import repository


def check_evaluation(directory):
    directory = Path(directory)
    result = json.loads((directory / "results.json").read_text())
    records = [json.loads(line) for line in (directory / "records.jsonl").read_text().splitlines()]
    if result.get("smoke") is not True or result["samples"] != 8:
        raise ValueError("Expected the official smoke cohort with eight samples")
    if result["failed_benchmarks"] or len(records) != result["cohort_total"] * 8:
        raise ValueError("Incomplete generation or failed benchmark grading")
    return {
        "passed": True,
        "questions": result["cohort_total"],
        "responses": len(records),
        "sandbox": result["sandbox"],
        "cohort_sha256": result["cohort_sha256"],
        "results_sha256": sha256(directory / "results.json"),
        "technical_only": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--adapters", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--lease", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise ValueError("Use a fresh smoke output directory")
    out.mkdir(parents=True)
    receipt = {
        "technical_only": True,
        "passed": False,
        "phases": [],
        "full_baseline_generated": False,
        "started_at_unix": time.time(),
    }
    write_json(out / "report.json", receipt)
    workspace = initialize(out / "runtime", "self-replay")
    stage_frozen(args.data.resolve(), workspace / "data/replay-scaled")
    shutil.copy2(args.lease, workspace / "worker-lease.json")
    env = environment(workspace)
    # The source checkout is staged separately from the materialized worker.
    env["SELF_REPLAY_REPO"] = str(repository())
    env["PYTHONPATH"] += os.pathsep + str(repository() / "src")

    def run(name, command, *, inference=False):
        lease = json.loads(args.lease.read_text())
        if time.time() >= min(lease["deadline"], lease["hard_deadline"]) - 600:
            raise RuntimeError("Not enough leased time remains for another smoke phase")
        phase_env = dict(env)
        if inference:
            phase_env["CUDA_VISIBLE_DEVICES"] = "0"
        phase = {"name": name, "command": list(map(str, command)), "started_at_unix": time.time()}
        receipt["phases"].append(phase)
        receipt["current_phase"] = name
        write_json(out / "report.json", receipt)
        print("SMOKE PHASE:", name, flush=True)
        try:
            subprocess.run(phase["command"], cwd=workspace, env=phase_env, check=True)
            phase["passed"] = True
        except BaseException:
            phase["passed"] = False
            raise
        finally:
            phase["seconds"] = time.time() - phase["started_at_unix"]
            write_json(out / "report.json", receipt)

    cli = [sys.executable, "-m", "self_replay.cli"]
    run("training-command", [*cli, "train", "self-replay", "--workspace", workspace, "--dry-run"])
    reference, resumed = out / "reference", out / "resumed"
    run(
        "four-group-training",
        [
            *cli,
            "train",
            "self-replay",
            "--workspace",
            workspace,
            "--out",
            reference,
            "--",
            "--pilot",
        ],
    )
    run(
        "complete-state-resume",
        [
            *cli,
            "train",
            "self-replay",
            "--workspace",
            workspace,
            "--out",
            resumed,
            "--",
            "--pilot",
            "--resume",
            reference / "checkpoints/group-000002",
        ],
    )
    run(
        "bitwise-resume-parity",
        [
            sys.executable,
            "-m",
            "experiments.replay_scaled.check",
            "--reference",
            reference,
            "--resumed",
            resumed,
            "--world",
            "2",
            "--out",
            out / "resume-parity.json",
        ],
    )
    run(
        "vllm-routing-compatibility",
        [sys.executable, "-m", "experiments.vllm_compat", "--out", out / "vllm-compatibility.json"],
    )
    evidence = json.loads((repository() / "provenance/evidence.json").read_text())
    receipt["evaluations"] = {}
    for checkpoint in (256, 850):
        name = f"checkpoint-{checkpoint}"
        adapter = args.adapters.resolve() / name
        for filename in ("adapter_model.safetensors", "adapter_config.json"):
            expected = evidence[f"artifacts/models/{name}/{filename}"]["sha256"]
            if sha256(adapter / filename) != expected:
                raise ValueError("Released adapter checksum differs: " + name)
        routing = out / ("routing-" + name)
        (routing / "adapters").mkdir(parents=True)
        (routing / "adapters" / name).symlink_to(adapter, target_is_directory=True)
        write_json(routing / "config.json", {"candidates": ["base", name]})
        run(
            "routing-" + name,
            [
                sys.executable,
                "-m",
                "experiments.vllm_lora_smoke",
                "--config",
                routing / "config.json",
                "--out",
                routing / "result.json",
            ],
            inference=True,
        )
        evaluation = out / ("evaluation-" + name)
        run(
            "official-smoke-" + name,
            [
                *cli,
                "evaluate",
                "--",
                "--smoke",
                "--questions",
                args.questions.resolve(),
                "--adapter",
                adapter,
                "--out",
                evaluation,
                "--workers",
                "2",
            ],
            inference=True,
        )
        receipt["evaluations"][name] = check_evaluation(evaluation)
        write_json(out / "report.json", receipt)
    receipt.update(
        passed=True, current_phase="complete", seconds=time.time() - receipt["started_at_unix"]
    )
    write_json(out / "report.json", receipt)
    print("GPU smoke suite passed; no efficacy score is claimed.", flush=True)


if __name__ == "__main__":
    main()
