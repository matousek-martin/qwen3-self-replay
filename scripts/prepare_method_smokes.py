#!/usr/bin/env python3
"""Materialize and inspect other-method smoke proposals locally; never execute them."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from self_replay.artifacts import sha256, write_json
from self_replay.methods.runtime import command, initialize, recipes, verify
from self_replay.paths import repository


def prepare(plan_path, output):
    plan_path, output = Path(plan_path).resolve(), Path(output).resolve()
    plan = json.loads(plan_path.read_text())
    registry = recipes()
    names = [case["recipe"] for case in plan["cases"]]
    if len(names) != len(set(names)) or set(names) != set(registry) - {"self-replay"}:
        raise ValueError("Plan must cover every other recipe exactly once")
    # This command only materializes source; plan status never enables execution.
    if output.exists():
        raise ValueError("Use a fresh output directory; preserve previous preparation evidence")
    output.mkdir(parents=True)
    receipt = {
        "status": "prepared_locally_not_executed",
        "gpu_execution": False,
        "provider_calls": False,
        "plan_sha256": sha256(plan_path),
        "cases": [],
    }
    for case in plan["cases"]:
        workspace = initialize(output / "workspaces" / case["recipe"], case["recipe"])
        verify(workspace)
        bounded = None
        if case["family"] == "offline":
            bounded = [
                "--steps",
                "2",
                "--grad-accum",
                "1",
                "--checkpoint-every",
                "1",
                "--max-seconds",
                "600",
            ]
        elif case["family"] == "online-rl":
            bounded = ["--max-steps", "4", "--save-every", "1", "--max-seconds", "900"]
        missing = [
            p for p in registry[case["recipe"]]["required_inputs"] if not (workspace / p).exists()
        ]
        default_data = workspace / registry[case["recipe"]]["data"]
        if not default_data.exists():
            missing.append(str(default_data.relative_to(workspace)))
        receipt["cases"].append(
            {
                "recipe": case["recipe"],
                "source_verified": True,
                "workspace_manifest_sha256": sha256(workspace / "workspace.json"),
                "public_command_for_inspection_only": command(case["recipe"], workspace),
                "bounded_training_command_needs_inputs_and_lease": (
                    command(case["recipe"], workspace, extra=bounded) if bounded else None
                ),
                "missing_inputs": sorted(set(missing)),
                "preparation_still_required": case["preparation"],
                "launch_ready": False,
            }
        )
        write_json(output / "preparation.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan", type=Path, default=repository() / "configs/smoke/other-methods.json"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    receipt = prepare(args.plan, args.out)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "source_verified_recipes": len(receipt["cases"]),
                "bounded_native_commands": sum(
                    c["bounded_training_command_needs_inputs_and_lease"] is not None
                    for c in receipt["cases"]
                ),
                "gpu_execution": False,
                "receipt": str(args.out.resolve() / "preparation.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
