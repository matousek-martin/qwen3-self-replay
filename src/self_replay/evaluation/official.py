"""Invoke the pinned official evaluator without changing its scientific settings."""

import subprocess
import sys

from self_replay.paths import repository


def evaluate(arguments):
    vendor = repository() / "vendor/thinkingcap"
    return subprocess.run(
        [sys.executable, str(vendor / "evaluate.py"), *arguments], cwd=vendor, check=True
    )


def compare(base, candidate, output):
    vendor = repository() / "vendor/thinkingcap"
    return subprocess.run(
        [
            sys.executable,
            str(vendor / "compare.py"),
            str(base),
            str(candidate),
            "--out",
            str(output),
        ],
        cwd=vendor,
        check=True,
    )
