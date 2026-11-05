"""Materialize portable scientific workers without rewriting historical imports.

Workers deliberately keep their proven `experiments.*` module names. Each workspace
has its own source, data and output tree; nothing refers back to the author's lab.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from self_replay.artifacts import sha256, verify_files, write_json
from self_replay.paths import repository, research


def recipes():
    return json.loads((repository() / "configs/recipes.json").read_text())


def initialize(destination, recipe="self-replay"):
    destination = Path(destination).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Workspace must be empty; existing experiments are never overwritten")
    spec = recipes()[recipe]
    shutil.copytree(
        research(),
        destination,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    overlay = research().parent / "overlays" / (spec.get("overlay") or "")
    if spec.get("overlay") and overlay.exists():
        shutil.copytree(
            overlay,
            destination,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    # Reapply portable host transport/policy after scientific source overlays.
    for relative in (
        "experiments/cloud.py",
        "experiments/public_policy.py",
        "experiments/shorterbetter/hardware.py",
        "experiments/shorterbetter/budget.py",
        "ops/runpod.py",
        "ops/vastai.py",
    ):
        shutil.copy2(research() / relative, destination / relative)
    shutil.copytree(
        repository() / "vendor/thinkingcap",
        destination / "ThinkingCap-Test",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copytree(
        repository() / "vendor/shorterbetter",
        destination / "lab/experiments/shorterbetter-preparation-20261026/sources",
        dirs_exist_ok=True,
    )
    (destination / "lab").mkdir(exist_ok=True)
    shutil.copy2(repository() / "requirements/gpu.lock", destination / "lab/gpu-requirements.lock")
    for folder in ["data", "output", "lab/experiments"]:
        (destination / folder).mkdir(parents=True, exist_ok=True)
    # Preparers historically use lab/experiments/data; workers use data.
    # One directory serves both interfaces, avoiding divergent copies.
    (destination / "lab/experiments/data").symlink_to("../../data", target_is_directory=True)
    # Input/source locks are scientific records, never cloud authorizations.
    inputs = repository() / "configs/inputs" / (spec.get("overlay") or recipe)
    if inputs.exists():
        shutil.copytree(inputs, destination / "data", dirs_exist_ok=True)
    source_lock = destination / "data/isolated/sources.json"
    if source_lock.exists():
        original = json.loads(source_lock.read_text())
        missing = [name for name in original if not (destination / name).is_file()]
        # Missing historical-only files are an explicit compatibility issue, never silently ignored.
        if any(not name.startswith("ThinkingCap-Test/questions/") for name in missing):
            raise ValueError("Source snapshot missing: " + ", ".join(missing))
        # Frozen benchmark downloads remain required, with their original hashes.
        # They are evaluation-only assets and are not imported into the code repository.
        write_json(source_lock.with_name("historical-sources.json"), original)
        current = {
            name: sha256(destination / name) if name not in missing else digest
            for name, digest in original.items()
        }
        write_json(source_lock, current)
        write_json(source_lock.with_name("pending-evaluation-assets.json"), missing)
        write_json(
            source_lock.with_name("refactor-source-changes.json"),
            {
                name: {"historical": digest, "current": current[name]}
                for name, digest in original.items()
                if current[name] != digest
            },
        )
    files = {
        str(p.relative_to(destination)): sha256(p)
        for p in sorted(destination.rglob("*"))
        if p.is_file()
    }
    write_json(
        destination / "workspace.json", {"schema_version": 1, "recipe": recipe, "files": files}
    )
    return destination


def verify(workspace):
    workspace = Path(workspace).resolve()
    manifest = json.loads((workspace / "workspace.json").read_text())
    verify_files(workspace, manifest["files"])
    return manifest


def environment(workspace):
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(Path(workspace).resolve()),
            str(Path(workspace).resolve() / "ops"),
            str(repository() / "src"),
        ]
    )
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    env.setdefault("OMP_NUM_THREADS", "1")
    return env


def command(recipe, workspace, data=None, output=None, extra=()):
    spec = recipes()[recipe]
    workspace = Path(workspace).resolve()
    manifest = verify(workspace)
    if recipes()[manifest["recipe"]].get("overlay") != spec.get("overlay"):
        raise ValueError("Recipe needs a workspace with its own source overlay")
    values = {
        "python": sys.executable,
        "workspace": str(workspace),
        "data": str(Path(data).resolve()) if data else str(workspace / spec.get("data", "data")),
        "output": str(Path(output).resolve()) if output else str(workspace / "output/training"),
    }
    return [part.format(**values) for part in spec["command"]] + list(extra)


def train(recipe, workspace, data=None, output=None, extra=(), dry_run=False):
    args = command(recipe, workspace, data, output, extra)
    if dry_run:
        return args
    spec = recipes()[recipe]
    for relative in spec.get("required_inputs", []):
        if not (Path(workspace) / relative).exists():
            raise ValueError("Prepare required input first: " + relative)
    if recipe in ("coding-a", "coding-b", "coding-c"):
        arm = json.loads((Path(workspace) / "data/coding-abc/plan.json").read_text())["arm"]
        if arm.lower() != recipe[-1]:
            raise ValueError("Coding plan arm does not match selected recipe")
    if spec["kind"] == "training":
        subprocess.run(
            [sys.executable, "-m", "experiments.gpu_policy"],
            cwd=workspace,
            env=environment(workspace),
            check=True,
        )
    subprocess.run(args, cwd=workspace, env=environment(workspace), check=True)
    return args
