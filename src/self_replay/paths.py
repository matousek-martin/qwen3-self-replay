"""Resolve the source checkout explicitly; never fall back to a private lab."""

import os
from pathlib import Path


def repository():
    candidates = (
        [Path(os.environ["SELF_REPLAY_REPO"])]
        if "SELF_REPLAY_REPO" in os.environ
        else Path(__file__).resolve().parents
    )
    for path in candidates:
        if (path / "vendor/thinkingcap/evaluate.py").is_file() and (
            path / "provenance/import.json"
        ).is_file():
            return path
    raise RuntimeError("Use a source checkout (pip install -e .), or set SELF_REPLAY_REPO to it.")


def research():
    return Path(__file__).parent / "methods/research"
