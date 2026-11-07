# Other-method RTX 4090 smoke checks

I verified all **21 single-GPU training recipes** on NVIDIA GeForce RTX 4090 workers.
I ran independent batches in parallel, with at most five smoke workers allocated
at once and exactly one RTX 4090 visible to each test. Every passing run has a verified
backup and confirmed provider release. The conservative recorded rental estimate,
including failed attempts, is **$1.7944**, within the separate $10 ceiling;
it is an operational estimate, not an invoice.

[Machine-readable results](results.json) contain native completion markers,
optimizer/replay/distillation activity, checkpoint reload results, input hashes
and source revisions. [Attempt history](attempts.json) retains failures and cleanup;
[SHA-256 manifest](SHA256.json) covers the exported evidence. Raw harness reports
and installed-package lists are under `runs/`.

| Recipe | Arms checked | Optimizer updates | Test minutes | Result |
|---|---:|---:|---:|---|
| `adaptive-replay` | 2 | 8 | 4.8 | passed |
| `coding-a` | 1 | 2 | 5.1 | passed |
| `coding-b` | 1 | 2 | 1.7 | passed |
| `coding-c` | 1 | 2 | 5.1 | passed |
| `dpo` | 1 | 2 | 0.3 | passed |
| `dr-grpo` | 1 | 3 | 3.7 | passed |
| `grpo` | 1 | 1 | 4.6 | passed |
| `on-policy-distillation` | 3 | 6 | 5.4 | passed |
| `opencoder-sft` | 1 | 2 | 0.4 | passed |
| `replay-selection` | 1 | 4 | 2.1 | passed |
| `rl-adaptive` | 1 | 4 | 3.7 | passed |
| `rl-difficulty` | 1 | 4 | 3.7 | passed |
| `rl-fixed` | 1 | 4 | 5.2 | passed |
| `rl-progressive` | 1 | 2 | 4.8 | passed |
| `rl-replay` | 1 | 4 | 3.2 | passed |
| `sft-compatible` | 1 | 2 | 0.3 | passed |
| `sft-dpo-matched` | 1 | 2 | 0.3 | passed |
| `sft-random` | 1 | 2 | 0.3 | passed |
| `sft-short` | 1 | 2 | 0.6 | passed |
| `shorterbetter` | 1 | 8 | 1.9 | passed |
| `teacher-replay` | 2 | 8 | 4.2 | passed |

Times include native commands and fresh model reloads on their recorded worker,
but exclude rental startup and package installation. They are engineering timings,
not controlled performance comparisons between methods.

## What each pass means

- SFT/DPO: two native updates, finite loss, saved adapter and fresh inference.
- GRPO/reward/replay: four groups of four responses with original token limits,
  an actual optimizer update and replay tokens where replay is part of the method.
- Adaptive replay: both fixed and adaptive arms run independently.
- Distillation: offline-hard, offline-soft and on-policy-soft all execute their
  distillation loss with the original pinned frozen 4B teacher; teacher replay tests
  both archived student and teacher targets, without new teacher API calls.
- OpenCoder: its native two-update technical mode uses the longest 64 scheduled
  examples, effective batch 32 and the verified checkpoint 768 parent.
- Coding A/B/C: each uses two original training questions and four independent
  development questions × two responses; native 40,960-token backward, bitwise
  optimizer roundtrip, objective-specific updates and saved-adapter reload pass.
- ShorterBetter: its native single-4090 math technical fixture runs eight full-weight
  optimizer updates in the separate pinned TRL/vLLM runtime, with a 256-token cap,
  precision checks, saved full-model reload and measurable model effect. It does
  not test the full 6,000-token, four-domain scientific training configuration.

The ShorterBetter fixture deliberately varies short completion caps to exercise
its SOL reward; the coding fixtures select short previously verified own answers
to bound runtime. The native objectives remain unchanged. These checks establish
execution, checkpoint and cleanup behavior on small engineering fixtures. They do
not reproduce historical scores, prove convergence or cover every long-run branch.
Adapter reload alone is not exact optimizer/RNG resume; the stronger coding
optimizer proof and [main self-replay resume test](../2026-11-07/README.md) are
identified separately.

## Failures retained and fixed

The first batch launch encountered an orchestration concurrency limit; both scoped
admission checks were corrected. One host did not start within its bounded window,
and two rejected allocations required complete-list/time reconciliation before
replacement; each reconciliation registered a 24-hour late-arrival watch. An unused
startup attempt was also released after its controller was interrupted during an
offer correction. All charges remain in the same combined ceiling.

The first coding batch completed OpenCoder's native training/checkpoints, but its
fresh reload timed out on blocked Hugging Face metadata requests. I retained that
failure, archived and released the worker, then repeated the batch with a
SHA-verified pinned base snapshot and explicit offline loading. The bootstrap
command path was corrected, subprocess interruption now reaps child processes,
and technical development labels report their actual cohort size.

The next run exposed two additional defects: pytest was missing from the main GPU
lock even though the coding sandbox imports it, and ShorterBetter passed an
implicit CUDA index to RNG isolation in a single-process run. I pinned the
historically used grading dependencies, resolved the current CUDA device explicitly,
added an RNG regression test and reran the affected methods on separate workers.
All 47 public CPU tests, lint and release artifact verification also passed.

Disposable ShorterBetter smoke weights are removed only after successful fresh
reload, retaining their exact hashes and a retention receipt. Scientific
checkpoints and existing campaigns are preserved.

## Scope and reproduction

The main submission and released adapters 256/850 have [separate GPU evidence](../2026-11-07/README.md).
Curriculum replay remains excluded: its original exact-resume contract uses
two A100s and two-rank state. Teacher generation is an API tool, excluded from
this GPU count. No full competition evaluation or official baseline was generated
by this campaign, and no smoke example is added to a submitted checkpoint.

See the [coverage and worker instructions](../../../docs/OTHER_METHOD_SMOKE_PLAN.md)
and the [GPU runner](../../../scripts/gpu_method_smoke.py). Every runtime/data/source snapshot is recorded
independently; earlier failures are not relabeled as successful runs.
