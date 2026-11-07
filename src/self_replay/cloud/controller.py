"""Portable entrypoints around the single, battle-tested cloud controller."""

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from self_replay.artifacts import write_json
from self_replay.cloud.jobs import load, validate_authorization
from self_replay.methods.runtime import verify


def canonical(workspace):
    workspace = Path(workspace).resolve()
    # One workspace per process avoids module globals pointing at another ledger.
    already = sys.modules.get("experiments.cloud")
    if already is not None and Path(already.ROOT).resolve() != workspace:
        raise RuntimeError("Use a separate process for a different cloud workspace")
    sys.path[:0] = [str(workspace), str(workspace / "ops")]
    return importlib.import_module("experiments.cloud")


def launch(workspace, spec_path, approval_path):
    verify(workspace)
    workspace = Path(workspace).resolve()
    spec = load(spec_path)
    approval = validate_authorization(spec, spec_path, json.loads(Path(approval_path).read_text()))
    if not (workspace / spec.entry).is_file():
        raise ValueError("Worker entrypoint is missing")
    # Persist the fresh approved contract into this workspace only.
    state = workspace / "lab/experiments/cloud"
    auth_path = state / "authorizations" / (spec.name + ".json")
    if auth_path.exists():
        raise ValueError("Job already registered; inspect/watch it instead of relaunching")
    if subprocess.run(["systemctl", "is-active", "--quiet", spec.guard_unit]).returncode:
        raise RuntimeError("Independent guard must be active before launch")
    cloud = canonical(workspace)
    cloud.resolve_uv()
    write_json(auth_path, approval)
    os.environ["SELF_REPLAY_GUARD_UNIT"] = spec.guard_unit
    args = SimpleNamespace(
        name=spec.name,
        provider=spec.provider,
        entry=spec.entry,
        gpu=spec.gpu,
        gpu_count=spec.gpu_count,
        hardware_scope=spec.hardware_scope,
        max_cost_usd=spec.max_cost_usd,
        max_runtime_hours=spec.max_runtime_hours,
        hours=spec.hours,
        vast_offer_id=spec.offer_id,
        vast_disk_gb=spec.disk_gb,
        completion_budget_scope=None,
        public_authorization=approval,
    )
    return cloud.run(args)


def inspect(workspace):
    path = Path(workspace) / "lab/experiments/cloud/ledger.json"
    return (
        json.loads(path.read_text())
        if path.exists()
        else {"jobs": {}, "note": "No jobs registered"}
    )


def watch(workspace, name):
    return canonical(workspace).watch(name)


def guard(workspace):
    return canonical(workspace).guard()


def quote(workspace, provider, gpu, count=1):
    cloud = canonical(workspace)
    if provider == "vast":
        return cloud.vastai.quote(gpu, count)
    if count != 1:
        raise ValueError("Use the RunPod contract-specific quote for multiple GPUs")
    runpod = importlib.import_module("runpod")
    return runpod.quote("SECURE")
