# Learning shorter answers from verified self-solutions

I wanted to test whether Qwen3-0.6B could learn brevity from answers it already produces correctly. The submitted checkpoint uses **31.23% fewer total output tokens**, with an average accuracy change of **+0.21 percentage points** (95% interval **−0.48 to +0.88 pp**). Thinking remains enabled.

## Method

For each training question, generate four answers and check them with the task's verifier. Incorrect or incomplete answers receive zero reward. Correct answers receive:

```text
reward = 1 − 0.1 × min(output_tokens / reference_length, 1)
```

The reference length is fixed from base-model generations on training prompts. Policy updates use mean-centered, unscaled group advantages with Dr. GRPO normalization.

When all four answers are correct, keep the shortest in a replay buffer. Each iteration also trains on one uniformly sampled buffer answer using completion-only supervised loss, weighted by 0.1. The buffer holds 128 questions and evicts the oldest entry. This lets replay provide an update even when the current question yields no useful policy gradient.

These are established ingredients; I do not claim to have invented group-relative optimization or replay. The question is whether this combination gives a useful compression/accuracy tradeoff. The four-correct gate is a heuristic, not an established optimum. [The method specification](docs/METHOD.md) gives the exact losses and implementation.

Training used rank-16 LoRA, alpha 32, learning rate 1e-5, four answers per question and a 2,048-token training cap. Questions cover math, coding, tool use and instruction following. No external teacher supplied training answers, and competition benchmarks were excluded from training and generation seeds.

**Hardware:** 2×A100-SXM4-80GB for training; RTX 4090 for full evaluation. Compatible cached baselines were reused with their original provenance. Checkpoint numbers count processed questions. The planned run contained 3,840 questions; the measured trajectory reached 1,280.

## Results

| Checkpoint | Total tokens saved | Accuracy change | 95% accuracy interval |
|---|---:|---:|---:|
| 128 | 27.10% | +0.03 pp | [−0.59, +0.69] pp |
| **256 — submitted primary** | **31.23%** | **+0.21 pp** | **[−0.48, +0.88] pp** |
| 850 — alternative | 41.60% | −1.17 pp | [−1.96, −0.35] pp |
| 1280 | 46.58% | −2.35 pp | [−3.19, −1.44] pp |

Each result uses the full official 2,759-question × 8-response protocol with unchanged prompts, sampling and token caps. Accuracy averages the four domains equally; total-token savings use the official geometric aggregation. [All seven measured checkpoints](studies/self-replay/history.json) and original evaluator files are retained.

I chose 256 because it gave substantial shortening with the accuracy estimate close to baseline. Its interval does not establish equivalence. The alternative checkpoint shows what additional compression costs.

## What worked, and what did not

Removing replay reduced savings from **27.10% to 1.85%** at 128 questions. Removing the explicit length reward retained **27.52% savings**, versus **31.23%** for the combined method at 256 questions. These [component comparisons](studies/self-replay/ablations/) use the original question order and seed, with controls from the earlier trajectory. They suggest replay contributes strongly to shortening, but do not establish that the combined method is always better: shortest-answer selection still favors brevity without the explicit length reward.

Longer training shortened outputs further but reduced accuracy. Teacher replay, distillation, ShorterBetter and coding follow-ups did not establish a better replacement under their respective protocols. Later replay-policy and data changes also failed to give a consistently better tradeoff. The [experiment notes](studies/README.md) distinguish full evaluations from development screens.

## Limitations

The main failure is tool abstention: BFCL no-call accuracy drops **4.17 pp at 256** and **6.88 pp at 850**. The development panel had only two no-call questions. Coding was also limited by truncation during training.

The main trajectory has one training seed, and repeated benchmark inspection influenced checkpoint selection. The intervals do not account for either selection or training-seed variability. Lexical overlap screening cannot prove zero semantic contamination, and task-specific verifiers do not guarantee general semantic correctness.

## AI assistance

I set the research direction, budgets and experiment decisions. Coding agents substantially assisted implementation, operations, debugging, analysis and writing; I remain responsible for the submission and do not claim every line was manually reviewed. The archive includes selected redacted conversations and timestamped tool-call metadata, not the complete private history. Source hashes, tests and saved evaluator outputs support the reported results.
