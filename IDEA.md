# Learning shorter reasoning from the model's own verified solutions

**Primary checkpoint: 256. Alternative: 850.** This is a Qwen3-0.6B LoRA submission with thinking enabled and unchanged contest evaluation settings.

## Motivation

A correct long answer is not necessarily useful supervision for a small model. Conversely, rewarding short outputs without protecting correctness can shorten failure. I explored whether the student could reuse solutions it already solves reliably, keeping the learning signal close to its own capabilities.

This combines established group-relative policy optimization and supervised replay; it is not a claim to have invented either. The contribution here is the concrete verified-solution selection rule, its implementation, the experimental comparison and the observed compression/accuracy trajectory on the contest model.

## Method

For each training question, sample four responses from the current student and verify them with the task-specific checker. Incorrect or incomplete responses receive no correctness reward. For a correct response of length L, the fixed reward is `1 − 0.1 × min(L / reference, 1)`, where the reference is a frozen base-model calibration on training prompts. Use mean-centered, unscaled group advantages with the registered Dr. GRPO normalization.

Admit a question to replay only when all four responses are verified correct. Keep its shortest correct response; replace an existing response only with a shorter one. The bounded buffer holds 128 questions. Sample one buffer entry per group and add completion-only supervised loss with weight 0.1. The replay random stream is separate from the prompt/rollout schedule. No external teacher supplies training answers in this main lineage.

The main configuration uses rank-16 LoRA, alpha 32, learning rate 1e-5, four answers per question and a 2,048-token training generation cap. Domains are balanced: math, coding, tools and instruction following. Full state includes adapter, Adam state, replay contents and rank RNG, with complete saves every 32 questions and separate development validation. The frozen configuration is [here](configs/self-replay-plan.json).

Training ran on **2× NVIDIA A100-SXM4-80GB**. Official evaluations ran on **RTX 4090**, reusing one verified compatible baseline. Reuse is reported as reuse, not a fresh baseline on each rental. The original plan contained 3,840 questions; the observed trajectory reaches 1,280, not the full planned count. Checkpoint numbers count processed training questions, not necessarily optimizer steps.

## Results

| Checkpoint | Total tokens saved | Accuracy Δ | 95% interval for accuracy Δ |
|---|---:|---:|---:|
| 128 | 27.10% | +0.03 pp | [−0.59, +0.69] pp |
| **256** | **31.23%** | **+0.21 pp** | **[−0.48, +0.88] pp** |
| 576 | 36.85% | −1.35 pp | [−2.09, −0.57] pp |
| 768 | 39.54% | −1.16 pp | [−1.93, −0.36] pp |
| **850** | **41.60%** | **−1.17 pp** | **[−1.96, −0.35] pp** |
| 1024 | 43.87% | −1.77 pp | [−2.57, −0.91] pp |
| 1280 | 46.58% | −2.35 pp | [−3.19, −1.44] pp |

Each point uses the full 2,759-question × 8-response official protocol. Accuracy is the arithmetic mean of four domains; token savings use the geometric mean of domain token ratios, pooling BFCL subsets. Intervals use the official paired-question bootstrap. Primary result files are preserved byte-for-byte, including `results.json` rather than only derived summaries.

256 is the primary checkpoint because it offers substantial savings without a detectable aggregate accuracy shift in this evaluation. This does **not** establish equivalence. 850 is a useful second point on the tradeoff curve, not an accuracy-preserving release. The direct paired 850→1280 comparison estimates an additional −1.18 pp accuracy change, with interval [−1.84, −0.53] pp.

## What did not work

- More training kept reducing length but eventually reduced aggregate accuracy.
- Relaxing replay admission and adaptive length targets did not establish a better compression tradeoff in the earlier matched experiments.
- Teacher replay shortened outputs substantially but harmed accuracy on its separate training-source validation panel.
- On-policy distillation did not pass its initial screening: its paired control comparison produced longer outputs.
- ShorterBetter showed a sharp early compression/accuracy tradeoff; later training and the higher-alpha variant did not establish a superior result.
- Coding SFT/RL branches did not improve the parent on full official evaluation; easy-to-hard curricula did not reliably solve the all-four-correct admission bottleneck.

These experiments use different protocols and controls. Their results are documented separately in [the experiment report](studies/README.md), not pooled into a fabricated leaderboard.

## Limitations and next questions

The most important hidden regression is BFCL no-call: −4.17 pp at 256 and −6.88 pp at 850. The two no-call questions in the development panel were insufficient to detect this reliably. Coding also suffered heavy truncation at the training cap. Correctness checks provide task-specific signals rather than proof of broad semantic correctness.

The main trajectory has one training seed. Repeated benchmark inspection influenced model selection; the confidence intervals do not correct for that selection or capture between-seed variation. Screening rejects exact and lexical overlap with evaluation prompts, but is not proof of zero semantic contamination. Evaluation prompts/answers were not used as training examples or generation seeds.

Next work would test the method across seeds and independent tasks, improve training-only no-call coverage, and examine coding admission separately. None of these future results is claimed here.

## Implementation, provenance and AI assistance

Historical scientific snapshots and artifact hashes are retained; the public refactor is a later engineering step. The MBPP worker memory fix and the pinned vLLM adapter-routing compatibility patch are disclosed in [EVALUATION.md](docs/EVALUATION.md); prompts, sampling, caps and grading rules were not intentionally changed.

Martin Matousek directed the project, priorities, budgets and experiment decisions. AI agents substantially assisted research suggestions, coding, operation, debugging, analysis and writing. [AI_USAGE.md](AI_USAGE.md) describes that division without claiming a manual review of every line.
