# Other-method GPU smoke coverage

**GPU validation in progress (2026-11-07). I authorized up to five concurrent
single-RTX4090 workers within a separate $10 combined ceiling. Results are
reported only after actual execution; this page does not claim all methods passed.**

The main self-replay recipe already has separate [GPU evidence](GPU_SMOKE.md).
For the other public entrypoints, I am checking the coverage below. Every
single-GPU test must see exactly one NVIDIA GeForce RTX 4090 at worker startup.

| Family | Recipes | Bounded technical check |
|---|---:|---|
| SFT selection variants and DPO | 5 | Two finite optimizer steps each; saved checkpoint and adapter reload; preserve selector provenance and DPO-matched prompts |
| GRPO, Dr.GRPO, length-reward/replay variants | 8 | Four groups × four responses each, original token limits; actual update, replay activity where applicable and checkpoint reload |
| Adaptive replay | 1 | Four groups for each fixed/adaptive arm; check reward and replay paths |
| On-policy distillation | 1 | Two updates for offline-hard, offline-soft and on-policy-soft; frozen teacher/cache provenance and finite loss |
| Teacher replay | 1 | Four groups each for archived student/teacher replay schedules; no new API generation |
| ShorterBetter | 1 | Existing two-rollout/eight-update technical check in its separate pinned TRL/vLLM environment |
| OpenCoder SFT | 1 | Existing two-update technical mode with the longest scheduled examples and verified parent768 |
| Coding A/B/C | 3 | Two questions per arm, native-context backward check, objective-specific updates and optimizer roundtrip |
| Curriculum replay | 1 | Deferred from the single4090 proposal: exact worker/resume contract requires two A100-SXM4-80GB GPUs |
| Teacher generation | 1 | CPU-only planning/cached-response checks; this is an API tool, not another training recipe |

That is **21 single4090 training recipes**, one deferred curriculum recipe and
one API tool. Some recipes contain multiple arms. A no-update rollout, missing
cache, out-of-memory failure or unexercised replay path is not a pass.

## Hardware distinction

Curriculum is the only remaining recipe explicitly locked to **two A100s**. It
inherits a complete checkpoint256 with optimizer/replay state and per-rank RNG;
the worker verifies both hardware and world size. This is the implementation's
execution/resume contract, not evidence that curriculum learning inherently needs
160GB of GPU memory. A single-rank adaptation would be a different technical
configuration and would not verify the original two-rank resume.

ShorterBetter has a single4090 implementation despite its larger historical runs
using multiple A100s. Other recipes have single4090 paths, but passing a hardware
check does not prove that every sequence/model combination fits 24GB; memory
failures must remain visible. In particular, do not reduce the coding native
context or substitute a smaller distillation teacher to claim a pass.

## Reproducible checks

`python scripts/prepare_method_smokes.py --out artifacts/method-smokes-review`
materializes all 23 source workspaces, verifies their hashes and records commands
without executing them. It has no cloud launcher or execution option. The
machine-readable proposal is [other-methods.json](../configs/smoke/other-methods.json).
All 23 materializations passed locally. The GPU harness is
`scripts/gpu_method_smoke.py`; it runs on an already leased worker and cannot rent
compute. Each batch uses immutable source and SHA-verified training inputs.

Adaptive, teacher and distillation arms call their native trainers with bounded
training-only configurations. Coding A/B/C use an explicit technical cohort while
retaining the native 40,960-token backward check. OpenCoder tests its longest
64 scheduled examples. ShorterBetter uses fresh data/reward checks and its separate
pinned environment. These engineering fixtures do not reproduce historical scores.

The pass criteria are actual finite updates, saved/reloaded adapters, method-specific
replay/cache/grading activity and retained source/data/environment hashes. Exact
resume is claimed only where complete optimizer/RNG state is saved and compared;
loading an adapter alone is not resume. Backup verification and confirmed provider
release are required for every rental. No full official baseline is regenerated.

## Bounded execution

Use **at most five smoke workers concurrently**, exactly one4090 each, with
sequential cases and separate outputs on each worker. Failed attempts count
against the same combined ceiling.

Read-only quotes on2026-11-07: RunPod Secure4090 **$0.89/GPU-hour**, low availability;
Vast **$0.393/hour including the quoted80GB disk**, plus transfer. Quotes are not
reservations. Working estimate: **3–6 GPU-hours**, with substantial uncertainty
until the first families are measured. Shared ceiling: **$10 and6hours**,
including setup, retries and cleanup; no automatic top-ups or extra budget for a
failed test. Curriculum2A100testing is excluded from this estimate.

Existing campaign reservations must remain protected. Recheck available credit,
provider identity/rate and the canonical ledger before every allocation. The CPU preparer cannot launch or spend; GPU execution
requires its own current, named lease.
