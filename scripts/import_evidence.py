#!/usr/bin/env python3
"""Export an explicit allowlist from the private lab; never copy runtime secrets."""

import argparse
import gzip
import json
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from self_replay.artifacts import sha256, verify_files, write_json
from self_replay.paths import repository


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lab", type=Path, required=True)
    a = p.parse_args()
    root, lab = repository(), a.lab.resolve()
    exported = {}

    def copy(source, target):
        source, target = Path(source), root / target
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        exported[str(target.relative_to(root))] = {
            "source": str(source.relative_to(lab)),
            "sha256": sha256(target),
        }
        return target

    history = []
    for registration in sorted(
        (lab / "lab/experiments/replay-scaled-evaluation").glob("*/campaign.json")
    ):
        campaign = json.loads(registration.read_text())
        if campaign.get("state") != "complete":
            continue
        if not campaign.get("backup_verified") or not campaign.get("released"):
            raise ValueError("Incomplete archive: " + str(registration))
        n = campaign["checkpoint_group"]
        output = Path(campaign["output"])
        comparison = json.loads((output / "comparison.json").read_text())
        if comparison["samples"] != 8 or any(comparison["full_evaluation"].values()):
            raise ValueError("Not an official complete evaluation")
        prefix = f"studies/self-replay/results/checkpoint-{n}"
        copy(output / "comparison.json", prefix + "/comparison.json")
        copy(output / f"runs/replay-{n}/results.json", prefix + "/results.json")
        copy(output / "completion.json", prefix + "/completion.json")
        copy(output / "questions-manifest.json", prefix + "/questions-manifest.json")
        if not (root / "studies/self-replay/results/base/results.json").exists():
            copy(output / "runs/base/results.json", "studies/self-replay/results/base/results.json")
        else:
            assert sha256(output / "runs/base/results.json") == sha256(
                root / "studies/self-replay/results/base/results.json"
            )
            exported["studies/self-replay/results/base/results.json"] = {
                "source": str((output / "runs/base/results.json").relative_to(lab)),
                "sha256": sha256(output / "runs/base/results.json"),
            }
        records = output / f"runs/replay-{n}/records.jsonl"
        history.append(
            {
                "checkpoint": n,
                "job": campaign["job_name"],
                "comparison": prefix + "/comparison.json",
                "records_sha256": sha256(records),
                "cohort_sha256": comparison["cohort_sha256"],
                "baseline_cache_key": campaign["models"]["base"].get("baseline_cache_key"),
                "baseline_source": campaign["models"]["base"].get("baseline_source"),
                "average": comparison["average"],
            }
        )
        # Response evidence is a release asset, never ordinary Git history.
        if n in (256, 850):
            for kind, source in [
                (f"checkpoint-{n}", records),
                ("base", output / "runs/base/records.jsonl"),
            ]:
                target = root / f"artifacts/records/{kind}/records.jsonl.gz"
                if target.exists():
                    with gzip.open(target, "rb") as existing:
                        import hashlib

                        if hashlib.file_digest(existing, "sha256").hexdigest() != sha256(source):
                            raise ValueError("Existing response asset differs: " + str(target))
                    exported[str(target.relative_to(root))] = {
                        "source": str(source.relative_to(lab)),
                        "sha256": sha256(target),
                        "uncompressed_sha256": sha256(source),
                    }
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with (
                    source.open("rb") as inp,
                    target.open("wb") as raw,
                    gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as out,
                ):
                    shutil.copyfileobj(inp, out)
                exported[str(target.relative_to(root))] = {
                    "source": str(source.relative_to(lab)),
                    "sha256": sha256(target),
                    "uncompressed_sha256": sha256(source),
                }
    history.sort(key=lambda row: row["checkpoint"])
    assert len({x["cohort_sha256"] for x in history}) == 1
    write_json(root / "studies/self-replay/history.json", history)
    training = (
        lab / "lab/experiments/cloud/replay-scaled-vast-resume1024-20261105t075310z/output/training"
    )
    for name in ("validation-curve.json", "training-series.json", "references.json"):
        copy(training / name, "studies/self-replay/training/" + name)
    for n in (256, 850):
        checkpoint = (
            lab
            / f"lab/experiments/cloud/replay-scaled-training-20261101t122440z/output/training/checkpoints/group-{n:06d}"
        )
        manifest = json.loads((checkpoint / "complete.json").read_text())
        verify_files(checkpoint, manifest["files"])
        for file in sorted((checkpoint / "adapter").iterdir()):
            if file.is_file():
                copy(file, f"artifacts/models/checkpoint-{n}/" + file.name)
        copy(checkpoint / "complete.json", f"provenance/checkpoint-{n}-complete.json")
    selected = {
        "replay-selection": "lab/experiments/completion-20261026/replay-selection-results.json",
        "adaptive-replay": "lab/experiments/completion-20261026/adaptive-replay-results.json",
        "on-policy-distillation": "lab/experiments/completion-20261026/on-policy-distillation-results.json",
        "teacher-pilot": "lab/experiments/completion-20261026/teacher-pilot-results.json",
        "coding-abc": "lab/experiments/coding-abc/20261104t192501z/official-vs-parent768-summary.json",
        "coding-curriculum": "lab/experiments/coding-curriculum/20261104t231104z/results-analysis.json",
        "opencoder-diagnostic": "lab/experiments/opencoder-mode-probe/20261103t231900z/analysis.json",
        "early-study": "lab/experiments/study/development-report.json",
        "1280-vs-850": "lab/analysis/replay-1280-review-20261105/1280-vs-850.json",
    }
    for name, relative in selected.items():
        copy(lab / relative, f"studies/experiments/{name}.json")
    # Only comparisons are exported from ShorterBetter, never ledger/account data.
    pointer = json.loads((lab / "lab/experiments/shorterbetter-paired/current.json").read_text())
    paired = json.loads((lab / pointer["path"]).read_text())
    for job in paired["parallel_evaluations"]:
        name = job["job_name"] if isinstance(job, dict) else job
        for f in sorted(
            (lab / "lab/experiments/cloud" / name / "output/paired-evaluation/comparisons").glob(
                "*.json"
            )
        ):
            copy(f, "studies/experiments/shorterbetter/" + f.name)
    data = lab / "lab/experiments/replay-scaled/20261101-checkpoints-v5"
    target = root / "artifacts/data/self-replay-frozen-data.zip"
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ("plan.json", "train.jsonl", "development.jsonl"):
            archive.write(data / name, name)
        archive.write(root / "provenance/training-data.json", "source-manifest.json")
        archive.write(root / "THIRD_PARTY_NOTICES.md", "THIRD_PARTY_NOTICES.md")
    exported[str(target.relative_to(root))] = {
        "source": str(data.relative_to(lab)),
        "sha256": sha256(target),
    }
    # No prompts in Git: a row-selection manifest can reconstruct/audit the frozen sample.
    selection = {}
    for split in ("train", "development"):
        rows = [json.loads(line) for line in (data / (split + ".jsonl")).read_text().splitlines()]
        selection[split] = [
            {
                k: row[k]
                for k in ("id", "source", "source_index", "source_revision", "split", "task")
                if k in row
            }
            for row in rows
        ]
    write_json(root / "provenance/data-selection.json", selection)
    write_json(root / "provenance/evidence.json", exported)
    print(f"Exported {len(exported)} verified artifacts and {len(history)} official checkpoints")


if __name__ == "__main__":
    main()
