"""Recheck downloaded, revision-pinned tokenizer snapshots; no model/API calls."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from transformers import AutoTokenizer

from .loss import assert_tokenizer_alignment

REVISIONS = {
    "Qwen3-0.6B": "c1899de289a04d12100db370d81485cdf75e47ca",
    "Qwen3-4B": "1cfa9a7208912126459214e8b04321603b3df60c",
    "Qwen3-8B": "b968826d9c46dd6066d109eabc6255188de91218",
}


def audit(root):
    snapshots = {name: root / name / rev for name, rev in REVISIONS.items()}
    tokenizers = {name: AutoTokenizer.from_pretrained(str(path), local_files_only=True)
                  for name, path in snapshots.items()}
    configs = {name: json.loads((path / "config.json").read_text())
               for name, path in snapshots.items()}
    student = tokenizers["Qwen3-0.6B"]
    results = {}
    for name, teacher in tokenizers.items():
        assert_tokenizer_alignment(student, teacher, configs["Qwen3-0.6B"]["vocab_size"],
                                   configs[name]["vocab_size"])
        path = snapshots[name]
        results[name] = {
            "revision": REVISIONS[name], "aligned": True,
            "file_sha256": {file.name: hashlib.sha256(file.read_bytes()).hexdigest()
                            for file in sorted(path.iterdir()) if file.is_file()},
        }
    return {"checked_at": datetime.now(timezone.utc).isoformat(), "models": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    # Refuse accidental replacement of evidence from an earlier check.
    result = audit(args.snapshots)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print("Aligned pinned tokenizers:", ", ".join(result["models"]))
