# ThinkingCap parallel pilots

> Navigation: [project start](../README.md) · [study index](../studies/INDEX.md) · [dated status](../docs/STATUS.md).
> This file contains several generations of campaigns. Historical plans and commands are not current launch instructions. For a specific experiment read its study README/HANDOFF first.
>
> Current hardware policy: **RTX 4090 for model training only**. Generation/validation/evaluation may use other hardware or hosted APIs. Historical L4 training remains L4 provenance. The current shared cloud controller still restricts its candidate list to RTX 4090; that implementation is narrower than the allowed evaluation policy. [AGENTS.md](../AGENTS.md) is authoritative.
>
> Current campaigns are completion-driven under the existing total spending cap, verified backups and GPU-release controls. Do not interpret legacy six-hour/four-hour limits below as current campaign policy. Inspect the selected campaign, controller and ledger before resuming.

Original pilot authorization (23 October 2026): prepare data on this server and run parallel RunPod experiments within **US$200 total**. No Telegram subscription was requested for that original launch; later notification authorization is task-specific and must be read from the relevant task/session records. Cloud credentials remain in the existing project `.env` and never enter GPU pods or artifacts.

## Full replay evaluation (25 October 2026)

The requested full evaluation uses the original locked `rl-replay` adapter and
a freshly generated baseline on one physical worker: all 2,759 official questions,
eight samples each, 22,072 responses per model. It preserves the official prompts,
decoding, grading, and paired comparison. This is a full benchmark measurement on
previously examined questions, not a new untouched holdout. It performs no training;
the adapter's historical NVIDIA L4 training provenance is retained.

`full_replay_eval.py` verifies the candidate lock, frozen question manifest, GPU
identity, complete response coverage, and official full-evaluation flags.
`full_replay_campaign.py` uses the existing budget, backup, verification, and GPU
release controller and logs progress and final results to the existing W&B
`thinkingcap` project. W&B credentials stay on this control server. Final artifacts
include both sets of graded responses, comparison intervals, source hashes,
software versions, and GPU provenance. Thinking-token savings are descriptive;
the official total-token statistic retains its geometric aggregation.

The durable service is `thinkingcap-full-replay-eval.service`. Campaign records
live under `../lab/experiments/full-replay/`; GPU outputs remain under the existing
`../lab/experiments/cloud/<job>/output/full-evaluation/` archive. If W&B publication
fails after the GPU is released, resume only publication without another rental:

```bash
.venv-lab/bin/python -m experiments.full_replay_campaign --name <existing-job> --publish-only
```

## Follow-up study: all approaches and accuracy preservation

The user subsequently requested completing every approach, retaining promising
methods with approximately one percentage point of regression for refinement,
and targeting zero regression in the final result. The registered criteria are
in `../lab/experiments/study/protocol.json`; the $200 ceiling includes the original
pilots. One percentage point is an exploratory point-estimate screening tolerance,
not the final accuracy claim. A confidence interval crossing zero is inconclusive.

`study_campaign.py` completes seven independent 128-group RL arms. Following the
user's request, new workers finish by completed work rather than elapsed time:
training/validation wall-time limits are disabled, and five-hour initial budget
reservations extend while the worker is alive and the total budget allows it.
Two training rentals leave a
third slot for `study_evaluation_campaign.py`. Cloud admission also counts pending
deletions and ambiguous creates toward the three-pod maximum. Replay now has a
fixed-reward control so it tests preservation while applying length pressure.

The evaluator uses unchanged official prompts, generation settings and graders
on a frozen development split of 512 questions (128 per domain), two samples
each. Existing SFT/DPO controls and three earlier DPO checkpoints join all seven
RL arms. Each evaluation batch generates a matched base-model result.
`confirmation.py` computes per-domain and equal-domain paired comparisons.
These are explicitly study subsets, not full official contest evaluations.
Reports distinguish total output-token savings from measured thinking-token
savings. The latter includes incorrect and truncated responses and reports the
ratio of equal-domain mean reasoning-token counts, with no confidence interval
claimed for that descriptive metric. The registered total-token ranking and
accuracy criteria remain unchanged.

The pinned vLLM 0.11.0 V1 output processor omits LoRA request metadata, which
caused the official evaluator's adapter-routing check to reject a completed
generation batch. Evaluation startup now applies a source-hash-pinned metadata
patch (`vllm_compat.py`) that carries the accepted request's actual adapter
identity into its output. The evaluator's check and all numerical generation
settings remain unchanged. A small real-GPU preflight verifies base/adapter/base
routing, a measurable adapter effect, and restoration of base inference before
the full evaluation starts. Patch hashes and preflight results are saved with
each job. The failed attempt remains archived, and its replacement regenerates
a matched baseline. Archive completion must also agree with the worker
supervisor's exit status; an interrupted shell cannot report a successful job.

The remaining 2,247 questions are reserved for eight-sample confirmation after
one primary candidate is selected and its files are locked. The development
controller never opens that confirmation set. Further tuning must use only the
development/training-validation evidence. Strict preservation requires the lower
paired confidence bound to reach zero, with domain-specific regressions reported;
the study may legitimately conclude that preservation remains inconclusive.

Current study state: `../lab/experiments/study/training-campaign.json`,
`evaluation-campaign.json`, and `work-status.json`. Exact uploaded source snapshots
are retained under each rental's `source/` directory. The sections below describe
the original pilot campaign and its unchanged historical outputs.

The server polls new workers every 45 seconds, verifies their final artifact
archive, and then terminates the rental. `worker_monitor.py` records process
identity and completion without a fixed timeout. Quiet logs trigger inspection,
not termination. If the controller stops, the independent guard observes the
worker instead of assuming it died. A reservation that cannot be renewed under
the budget ceiling requests graceful shutdown before its cost allowance expires.
Already-running legacy trainers were left uninterrupted and retain their loaded
four-hour limits; their ordinary completion is monitored. Controller service
lifetime limits have been removed. Unusable startup placements are still cleaned
up before any training is launched.

The CPU-prepared shared data lives in `../lab/experiments/data/`: 44,077 training questions, 800 held-out validation questions, a balanced 512-question training pilot, and a frozen 64-question validation pilot. Revisions, seeds, exclusions and hashes are recorded beside the data. Sources cover math and multiple-choice QA; they do not establish transfer to all four contest domains.

## Original pilot campaign (historical)

`campaign.py` schedules at most three GPUs concurrently:

1. **Bootstrap:** four base-model responses for each pilot question; independent verification; short/random SFT targets; short/long correct DPO pairs; frozen-student compatibility scores; matched base validation; one-step CUDA SFT/DPO and saved-adapter reload checks.
2. **RL controls, alongside bootstrap:** bounded CUDA smoke requiring a real update; 128 groups each of correctness-only GRPO and Dr. GRPO.
3. **After bootstrap:** 64-update short/random/compatible SFT comparisons, plus DPO and its SFT control restricted to identical paired prompts.
4. **After bootstrap and successful RL controls:** fixed, adaptive and difficulty-dependent length rewards; progressive training caps and solved-solution replay. Each has a 128-group pilot budget.

All candidates use the same 64 held-out questions, two samples per question and identical validation generation settings. These are **training-source validation results**, not official contest scores. Cross-method comparisons are exploratory: offline candidate generation has a 4,096-token cap and RL training a 2,048-token cap. Within each controlled comparison the settings match. Student compatibility is a frozen-model answer-likelihood proxy, not a full ReDiST reproduction; replay is not EMA self-distillation. Partial runs keep their checkpoints and validation but fail the completion gate.

The campaign does not automatically launch larger training or full contest evaluations. Those require reviewing the pilot metrics and remaining budget. There is no claim of accuracy preservation from this small validation set.

## Historical pilot runtime and budget controls (superseded)

The following records the original pilot policy for provenance. Its GPU fallback list and fixed six-hour limit are **not current instructions**; use the current policy above and the selected campaign configuration. Monetary examples are historical estimates, not a current balance.

`cloud.py` selects temporary secure-cloud GPUs in this order: L40, L4, RTX A6000, RTX A5000, and RTX 4090, at no more than $1.00 per GPU-hour. A40 is excluded because its initial placements failed to create containers; previous startup costs remain in the ledger. Allocation waits at most ten minutes, serializes local quote/create operations, and uses a new unique name for each retry. Only an explicit capacity rejection reconciled to no created pod permits another attempt; ambiguous network failures retain their reservation and are never retried automatically.

Each campaign job is limited to six hours including allocation/setup, with a separate ten-minute container startup limit. Before allocation, each job reserves the maximum eligible rate plus a margin: `$1.00 * 1.25 + $0.05 = $1.30/hour`. Six fresh six-hour windows therefore reserve up to $46.80, in addition to prior charges or outstanding commitments. The ledger permits at most $180 of estimated commitments and retains $20 for billing uncertainty. Estimates remain conservative and are not invoices.

A local independent systemd timer checks deadlines and controller heartbeats every minute. Artifacts are synchronized throughout each job, final backups are checksum-verified, and pods are terminated to release compute and disks. A failed or uncertain create is reconciled by its unique name, never blindly retried. Reservations remain until termination is confirmed. Service/API failures can delay cleanup; the substantial budget margin reduces exposure.

State and outputs:

- `../lab/experiments/campaign.json`: campaign dependency status.
- `../lab/experiments/cloud/ledger.json`: pod identities, deadlines, reservations and cleanup state.
- `../lab/experiments/cloud/<job>/output/`: checkpoints, logs, validation and provenance.

Inspect with `python3 experiments/cloud.py status` and `systemctl status thinkingcap-pilot-campaign.service`. The independent cleanup timer is `thinkingcap-pilot-guard.timer`. Do not disable the guard while any campaign pod remains active. Stopping the campaign causes orphan cleanup after the heartbeat timeout; use the provider's termination path for immediate cleanup, preserving current artifacts first when practical.

## Verification

Pure data and cloud lifecycle tests run without model dependencies. `offline_training/.venv-test/bin/python` is a local CPU test environment used to compare chunked losses and adapter gradients against dense tiny-Qwen calculations. Real CUDA checks run before their dependent pilots. See each subdirectory README for exact schemas and CLI options.
