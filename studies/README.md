# Experiment findings

The primary submission is **self-replay checkpoint 256**. Checkpoint 850 is an explicit compression/accuracy tradeoff, not a second claim of unchanged accuracy. [Every official checkpoint](self-replay/history.json), [per-domain plots](self-replay/figures/domains.png) and [development monitoring](self-replay/figures/development.png) remain visible.

## What we tried and learned

The rows below deliberately distinguish development screens from full official evaluation. Different cohorts, baselines and objectives must not be ranked as one leaderboard. A negative result is evidence about that tested configuration, not a proof that a method cannot work.

| Experiment | Evidence and outcome | Decision |
|---|---|---|
| Early SFT, DPO, GRPO/Dr.GRPO and length-reward variants | [Development report](experiments/early-study.json); includes matched-data SFT controls and multiple reward formulations | Keep the implementations and controls; do not promote development estimates to full benchmark claims |
| Replay selection | [Three-seed screen](experiments/replay-selection.json): +0.03 pp accuracy, −6.35% token savings (outputs grew) | More permissive admission did not deliver the desired compression |
| Adaptive replay | [Three-seed screen](experiments/adaptive-replay.json): +0.59 pp accuracy, −4.07% savings | Inconclusive accuracy and longer answers; retain fixed correctness-gated reward for the main run |
| On-policy distillation | [One-seed screen](experiments/on-policy-distillation.json): on-policy soft target vs offline soft target +1.95 pp, −9.74% savings | Did not pass the compression screen; comparison is not against the official base |
| Teacher replay pilot | [Three-seed training-source validation](experiments/teacher-pilot.json): −12.42 pp accuracy, 47.83% savings | Reject the tested configuration; keep teacher provenance separate from self-replay |
| ShorterBetter α2/α5 | [Twelve full evaluations](experiments/shorterbetter/); compression increased while accuracy regressed | Motivated examining early checkpoints and a different reward/replay mechanism |
| Main four-domain self-replay | [Seven full evaluations](self-replay/history.json): 256 saves 31.23% at +0.21 pp; 850 saves 41.60% at −1.17 pp | Select 256 as primary and show 850 as the stronger-compression alternative |
| OpenCoder SFT diagnostic | [Format audit](experiments/opencoder-diagnostic.json): an apparent formatting gain did not survive the first-code-block audit | Retain the parent checkpoint; do not advertise the apparent gain |
| Coding A/B/C from 768 | [Official parent comparison](experiments/coding-abc.json): B/C changed accuracy by −0.30/−0.23 pp vs parent; A stopped without a full result | No demonstrated improvement over the parent; A is incomplete, not a scored failure |
| Coding curriculum from 256 | [Development-only analysis](experiments/coding-curriculum.json): both branches +3.13 pp, intervals include zero; coding cohort is small | Interesting but inconclusive; no full official result claimed |
| Main continuation to 1280 | [Direct comparison to 850](experiments/1280-vs-850.json): −1.18 pp, 95% interval [−1.84, −0.53] pp | More compression did not justify replacing 256/850 |

## ShorterBetter checkpoints

Both branches use seed 17. Iteration is a rollout group; optimizer steps are four times that count. These are the original full-evaluation aggregates, rounded for display.

| Iteration | α2 tokens saved | α2 accuracy Δ | α5 tokens saved | α5 accuracy Δ |
|---:|---:|---:|---:|---:|
| 5 | 10.25% | −0.36 pp | 10.86% | −0.66 pp |
| 10 | 24.12% | −0.89 pp | 23.47% | −1.09 pp |
| 15 | 35.88% | −1.58 pp | 36.51% | −1.76 pp |
| 20 | 46.16% | −2.64 pp | 48.66% | −2.89 pp |
| 25 | 56.23% | −3.84 pp | 56.86% | −4.20 pp |
| 30 | 63.69% | −4.70 pp | 62.39% | −5.22 pp |

## Interpreting the main result

Checkpoint numbers count processed training questions, not model size or independent seeds. The main run planned 3,840 questions; the reported trajectory reached 1,280. We publish all seven evaluated checkpoints to expose selection effects. The 256 interval includes zero and does not establish equivalence. The 850 aggregate regression is supported by its interval. Both checkpoints regress on the BFCL no-call subset.

The repeatedly inspected official benchmarks are selection evidence, not an untouched holdout. Development validation used independent training-source data. Historical subset screens and main full evaluations have separate contracts; source JSON takes precedence over this rounded summary.

## Reproduce or extend

[Method recipes](../docs/METHODS.md) expose all method implementations, including negative experiments. [Reproduction](../docs/REPRODUCING.md) documents required data, teacher caches, parent checkpoints and scientific environments. The public packaging checks do not certify a new end-to-end GPU rerun of every method.

Source SHA manifests, immutable evaluator JSON and [export provenance](../provenance/evidence.json) connect the presentation to the original archives. Complete optimizer/RNG states remain in the research archive; the public checkpoint release contains inference adapters.
