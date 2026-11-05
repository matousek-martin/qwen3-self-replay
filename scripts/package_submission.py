#!/usr/bin/env python3
"""Build the official ZIP and separately licensed release assets from verified files."""

import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from self_replay.artifacts import sha256, write_json
from self_replay.paths import repository


def archive(source, destination):
    with zipfile.ZipFile(
        destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as out:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                info = zipfile.ZipInfo(
                    str(path.relative_to(source)), date_time=(2026, 11, 5, 0, 0, 0)
                )
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                out.writestr(info, path.read_bytes())
    with zipfile.ZipFile(destination) as check:
        if check.testzip() is not None:
            raise ValueError("Corrupt ZIP: " + str(destination))


def main():
    root = repository()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip():
        raise ValueError("Commit the reviewed source before building the submission")
    subprocess.run([sys.executable, str(root / "scripts/verify_release.py")], cwd=root, check=True)
    subprocess.run(
        [sys.executable, str(root / "scripts/prepare_model_cards.py")], cwd=root, check=True
    )
    out = root / "dist"
    out.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="thinkingcap-submission-") as temp:
        stage = Path(temp) / "submission"
        stage.mkdir()
        for n in (256, 850):
            destination = stage if n == 256 else stage / "other-checkpoints/checkpoint-850"
            shutil.copytree(
                root / f"artifacts/huggingface/Qwen3-0.6B-Self-Replay-{n}",
                destination,
                dirs_exist_ok=True,
            )
        for name in ("IDEA.md", "AI_USAGE.md"):
            shutil.copy2(root / name, stage / name)
        shutil.copy2(
            root / "studies/self-replay/results/base/results.json", stage / "baseline-results.json"
        )
        shutil.copytree(root / "artifacts/verification", stage / "verification")
        bundle = stage / "martin-matousek.bundle"
        subprocess.run(["git", "bundle", "create", str(bundle), "--all"], cwd=root, check=True)
        subprocess.run(["git", "bundle", "verify", str(bundle)], cwd=root, check=True)
        restored = Path(temp) / "restored"
        subprocess.run(["git", "clone", "--quiet", str(bundle), str(restored)], check=True)
        expected = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        actual = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=restored, text=True
        ).strip()
        if actual != expected:
            raise ValueError("Bundle does not restore the source HEAD")
        manifest = {
            str(p.relative_to(stage)): sha256(p) for p in sorted(stage.rglob("*")) if p.is_file()
        }
        write_json(stage / "SHA256SUMS.json", manifest)
        archive(stage, out / "martin-matousek.zip")
    if (out / "martin-matousek.zip").stat().st_size >= 500_000_000:
        raise ValueError("Submission exceeds the contest 500 MB upload limit")
    archive(root / "artifacts/records", out / "qwen3-self-replay-responses.zip")
    shutil.copy2(
        root / "artifacts/data/self-replay-frozen-data.zip", out / "self-replay-frozen-data.zip"
    )
    for n in (256, 850):
        archive(
            root / f"artifacts/huggingface/Qwen3-0.6B-Self-Replay-{n}",
            out / f"Qwen3-0.6B-Self-Replay-{n}.zip",
        )
    assets = {
        p.name: {"sha256": sha256(p), "bytes": p.stat().st_size} for p in sorted(out.glob("*.zip"))
    }
    write_json(
        out / "release-manifest.json",
        {
            "git_commit": expected,
            "assets": assets,
            "bundle_restored": True,
            "primary_checkpoint": 256,
            "alternative_checkpoint": 850,
        },
    )
    print(json.dumps(assets, indent=2))


if __name__ == "__main__":
    main()
