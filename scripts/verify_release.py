#!/usr/bin/env python3
"""Verify code history, evidence hashes, adapter files and public artifact hygiene."""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from self_replay.artifacts import sha256, write_json
from self_replay.paths import repository

SECRET_PATTERNS = [
    r"gh[pousr]_[A-Za-z0-9]{25,}",
    r"hf_[A-Za-z0-9]{25,}",
    r"sk-[A-Za-z0-9_-]{30,}",
    r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----",
    r"\b\d{8,12}:[A-Za-z0-9_-]{30,}\b",
]


def scan(text, label):
    if any(re.search(pattern, text) for pattern in SECRET_PATTERNS):
        raise ValueError("Possible secret in " + label)  # Never print the value.


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--code-only", action="store_true")
    args = p.parse_args()
    root = repository()
    files = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
    for name in filter(None, files):
        path = root / name
        if not path.exists():
            continue  # An unstaged deletion is caught by packaging's clean-tree gate.
        if path.suffix in (
            ".py",
            ".sh",
            ".md",
            ".json",
            ".toml",
            ".txt",
            ".yml",
            ".yaml",
            ".patch",
            ".lock",
        ):
            scan(path.read_text(), name)
        if path.name.startswith(".env") or path.suffix in (".pt", ".safetensors"):
            raise ValueError("Private/large artifact tracked in Git: " + name)
    # Scan all reachable text blobs, including files removed by later cleanup commits.
    objects = subprocess.check_output(
        ["git", "rev-list", "--objects", "--all"], cwd=root, text=True
    )
    scanned = 0
    for line in objects.splitlines():
        oid, _, name = line.partition(" ")
        if not name or Path(name).suffix not in (
            ".py",
            ".sh",
            ".md",
            ".json",
            ".toml",
            ".txt",
            ".yml",
            ".yaml",
            ".patch",
            ".lock",
        ):
            continue
        blob = subprocess.check_output(["git", "cat-file", "blob", oid], cwd=root)
        scan(blob.decode(errors="replace"), name)
        scanned += 1
    evidence = json.loads((root / "provenance/evidence.json").read_text())
    checked = 0
    for name, record in evidence.items():
        if args.code_only and name.startswith("artifacts/"):
            continue
        if sha256(root / name) != record["sha256"]:
            raise ValueError("Evidence hash mismatch: " + name)
        checked += 1
    if not args.code_only:
        from safetensors import safe_open

        for n in (256, 850):
            directory = root / f"artifacts/models/checkpoint-{n}"
            config = json.loads((directory / "adapter_config.json").read_text())
            if config["r"] != 16 or config["lora_alpha"] != 32 or config["peft_type"] != "LORA":
                raise ValueError("Unexpected adapter architecture")
            with safe_open(directory / "adapter_model.safetensors", framework="numpy") as tensors:
                if not tensors.keys() or any(
                    not key.endswith(("lora_A.weight", "lora_B.weight")) for key in tensors.keys()
                ):
                    raise ValueError("Invalid LoRA tensors")
            complete = json.loads((root / f"provenance/checkpoint-{n}-complete.json").read_text())
            for path in directory.iterdir():
                if sha256(path) != complete["files"]["adapter/" + path.name]:
                    raise ValueError("Checkpoint adapter changed: " + path.name)
    result = {
        "evidence_files_verified": checked,
        "historical_text_blobs_scanned": scanned,
        "code_only": args.code_only,
        "passed": True,
        "secret_scan_scope": "Known credential patterns, all reachable Git text blobs; not a guarantee against every secret format",
    }
    write_json(root / "artifacts/verification/release-check.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
