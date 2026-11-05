#!/usr/bin/env python3
"""Recompute 256/850 official comparisons from released responses, without a GPU."""

import argparse
import gzip
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from self_replay.artifacts import sha256, write_json
from self_replay.evaluation.official import compare
from self_replay.paths import repository

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--records", type=Path, default=repository() / "artifacts/records")
p.add_argument(
    "--out", type=Path, default=repository() / "artifacts/verification/recomputed-results.json"
)
a = p.parse_args()
root = repository()
proof = {}
evidence = json.loads((root / "provenance/evidence.json").read_text())
with tempfile.TemporaryDirectory(prefix="self-replay-compare-") as directory:
    tmp = Path(directory)
    for name in ("base", "checkpoint-256", "checkpoint-850"):
        target = tmp / name
        target.mkdir()
        with (
            gzip.open(a.records / name / "records.jsonl.gz", "rb") as source,
            (target / "records.jsonl").open("wb") as out,
        ):
            shutil.copyfileobj(source, out)
        expected = evidence[f"artifacts/records/{name}/records.jsonl.gz"]["uncompressed_sha256"]
        if sha256(target / "records.jsonl") != expected:
            raise ValueError("Corrupt response archive: " + name)
        shutil.copy2(
            root / f"studies/self-replay/results/{name}/results.json", target / "results.json"
        )
    for n in (256, 850):
        output = tmp / f"{n}.json"
        compare(tmp / "base", tmp / f"checkpoint-{n}", output)
        actual = json.loads(output.read_text())
        expected = json.loads(
            (root / f"studies/self-replay/results/checkpoint-{n}/comparison.json").read_text()
        )
        for key in ("average", "results", "samples", "cohort_sha256", "full_evaluation"):
            if actual[key] != expected[key]:
                raise ValueError(f"Official recomputation differs for {n}: {key}")
        proof[str(n)] = {
            "exact_official_metrics_match": True,
            "responses": 22072,
            "comparison_sha256": sha256(output),
        }
write_json(a.out, proof)
print("Exact official comparisons reproduced for 256 and 850; no inference")
