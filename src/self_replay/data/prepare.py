"""Portable preparation and verification for frozen scientific datasets."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

from self_replay.artifacts import verify_files, write_json
from self_replay.methods.runtime import environment, verify
from self_replay.paths import repository


def stage_frozen(source, destination, plan=None):
    """Copy only the exact registered train/development files; never infer a split."""
    source, destination = Path(source), Path(destination)
    plan = Path(plan) if plan else repository() / "configs/self-replay-plan.json"
    config = json.loads(plan.read_text())
    verify_files(source, config["files_sha256"])
    ids = {}
    for split in ("train", "development"):
        rows = [json.loads(line) for line in (source / (split + ".jsonl")).read_text().splitlines()]
        ids[split] = {row["id"] for row in rows}
        if len(ids[split]) != len(rows) or any(row["split"] != split for row in rows):
            raise ValueError("Invalid or duplicate split rows")
        if any(
            row["source"].lower() in {"svamp", "mbpp", "mbpp+", "bfcl", "ifeval"} for row in rows
        ):
            raise ValueError("Evaluation benchmark in training/development inputs")
    if ids["train"] & ids["development"]:
        raise ValueError("Training/development overlap")
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Refusing to overwrite a prepared dataset")
    destination.mkdir(parents=True, exist_ok=True)
    for relative in config["files_sha256"]:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, target)
    shutil.copy2(plan, destination / "plan.json")
    write_json(
        destination / "preparation-receipt.json",
        {
            "files": config["files_sha256"],
            "counts": {k: len(v) for k, v in ids.items()},
            "semantic_overlap_guarantee": False,
            "note": "Original screening is recorded in the source manifest; SHA verification is not new screening.",
        },
    )
    return destination


def prepare(workspace, recipe, arguments):
    verify(workspace)
    commands = {
        "initial": ["experiments/data_pipeline/pipeline.py", "prepare"],
        "select": ["experiments/data_pipeline/pipeline.py", "select"],
        "four-domain": ["-m", "experiments.shorterbetter.multidomain.prepare_full"],
        "self-replay": ["-m", "experiments.replay_scaled.prepare"],
        "opencoder": ["-m", "experiments.opencoder_sft.prepare"],
        "coding-frontier": ["-m", "experiments.coding_frontier.prepare"],
        "coding-curriculum": ["-m", "experiments.coding_curriculum.prepare"],
    }
    subprocess.run(
        [sys.executable, *commands[recipe], *arguments],
        cwd=workspace,
        env=environment(workspace),
        check=True,
    )
