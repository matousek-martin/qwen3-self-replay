"""Finite, explicit job specifications for the existing canonical controller.

This module describes work; it never treats historical funding as authorization.
The canonical controller still owns allocation locking, reservations, reconciliation,
monitoring, verified backup and release for both providers.
"""

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from self_replay.artifacts import sha256, write_json

GPU_4090 = "NVIDIA GeForce RTX 4090"
GPU_A100 = "NVIDIA A100-SXM4-80GB"
SCOPE = "public-reproduction-v1"


@dataclass(frozen=True)
class JobSpec:
    name: str
    provider: str
    entry: str
    max_cost_usd: float
    max_runtime_hours: float
    max_gpu_rate: float
    gpu: str = GPU_4090
    gpu_count: int = 1
    hardware_scope: str | None = None
    workload: str = "training"
    disk_gb: int = 80
    offer_id: int | None = None
    guard_unit: str = "self-replay-guard.timer"

    def validate(self):
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,100}", self.name):
            raise ValueError("Job name must be a unique lowercase slug")
        if self.provider not in ("runpod", "vast") or self.workload not in (
            "training",
            "evaluation",
        ):
            raise ValueError("Unknown provider or workload")
        if (
            Path(self.entry).is_absolute()
            or ".." in Path(self.entry).parts
            or not self.entry.endswith(".sh")
        ):
            raise ValueError("Entry must be a workspace-relative shell script")
        for value in (self.max_cost_usd, self.max_runtime_hours, self.max_gpu_rate):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError("Explicit positive finite cost, runtime and rate required")
        if self.max_cost_usd <= 1 or self.max_runtime_hours > 72:
            raise ValueError("Keep at least $1 cleanup reserve and at most 72 hours per job")
        if (
            type(self.gpu_count) is not int
            or self.gpu_count < 1
            or type(self.disk_gb) is not int
            or not 50 <= self.disk_gb <= 500
        ):
            raise ValueError("Invalid GPU count or storage")
        if self.workload == "training" and self.gpu != GPU_4090:
            replay = (
                self.hardware_scope == "replay-scaled-a100-v1"
                and self.name.startswith("replay-scaled-")
                and self.gpu_count in (2, 4)
            )
            curriculum = (
                self.hardware_scope == "coding-curriculum-256-v1"
                and self.name.startswith("coding-curriculum-256-")
                and self.gpu_count == 2
            )
            shorter = (
                self.hardware_scope == "shorterbetter-scaled-v1"
                and self.name.startswith("shorterbetter-scaled-")
                and self.gpu_count in (2, 4)
            )
            if self.gpu != GPU_A100 or not (replay or curriculum or shorter):
                raise ValueError("Training requires RTX 4090 or an explicit named A100 recipe")
        if self.provider == "vast" and (type(self.offer_id) is not int or self.offer_id <= 0):
            raise ValueError("Vast launch needs a current, explicitly selected offer ID")
        return self

    @property
    def budget_rate(self):
        return self.max_gpu_rate * self.gpu_count * 1.25 + 0.15

    @property
    def hours(self):
        return min(self.max_runtime_hours, (self.max_cost_usd - 1) / self.budget_rate)

    def authorization(self, spec_path, approved=False):
        self.validate()
        return {
            **asdict(self),
            "job_name": self.name,
            "scope": SCOPE,
            "authorized": approved,
            "spec_sha256": sha256(spec_path),
            "source_user_instruction": "",
            "note": "A human must approve this NEW finite job; no old budget is inherited.",
        }


def load(path):
    return JobSpec(**json.loads(Path(path).read_text())).validate()


def plan(path, output):
    spec = load(path)
    write_json(output, spec.authorization(path))
    return {
        "authorization": str(output),
        "initial_hours": spec.hours,
        "budget_rate_usd_per_hour": spec.budget_rate,
        "paid_execution": False,
    }


def validate_authorization(spec, spec_path, authorization):
    spec.validate()
    if authorization.get("authorized") is not True or not authorization.get(
        "source_user_instruction"
    ):
        raise ValueError("A new explicit approval with its source is required")
    if authorization.get("scope") != SCOPE or authorization.get("spec_sha256") != sha256(spec_path):
        raise ValueError("Authorization does not match this job specification")
    if any(authorization.get(k) != v for k, v in asdict(spec).items()):
        raise ValueError("Job differs from the approved cost/hardware/provider contract")
    return authorization
