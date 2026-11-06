# Method: verified self-replay

This specifies the main four-domain self-replay lineage behind checkpoints 256 and 850. Other replay, distillation, SFT and curriculum experiments have different contracts; their outcomes are documented in [the experiment findings](../studies/README.md).

## Two learning signals

For each question, the current student generates four answers. A task-specific verifier supplies correctness labels. A small length penalty distinguishes successful answers without making an incorrect answer preferable in raw reward. Group-relative policy optimization uses these rewards.

Separately, when **all four answers are verified correct**, the buffer admits the shortest answer. One stored answer supplies completion-only supervised loss alongside the policy update. The intended effect is to reinforce concise behavior already within the student's capabilities. Four successes are a selection heuristic, not a statistical certificate that a question is reliably solved.

## Reward and policy update

For answer $i$, let $c_i$ be its binary verified-correct label and $L_i$ its generated token count. Let $R_d$ be the frozen length reference for its domain:

$$
r_i = c_i\left(1 - 0.1\min\left(\frac{L_i}{R_d},1\right)\right),
\qquad
A_i = r_i - \frac{1}{4}\sum_{j=1}^{4}r_j.
$$

Correct answers receive at least 0.9; incorrect answers receive zero. This is a statement about **raw rewards**: a correct answer can still have a negative centered advantage when the other correct answers are shorter. Advantages are not divided by their standard deviation. The length penalty saturates at the reference, so it does not distinguish two correct answers that both exceed that reference.

Before training, the pinned base model generates four responses on each of the first 32 training questions—eight questions per domain. Each domain's reference is the mean generated length across those responses, including unsuccessful ones. These are training prompts, separate from development and official evaluation; the reference stays fixed during the run.

The policy term in the archived worker is

$$
\mathcal L_{\mathrm{policy}}
= -\frac{1}{4\cdot 2048}\sum_{i=1}^{4}\sum_{t=1}^{L_i}
A_i\exp\!\left(\ell_{i,t}-\operatorname{stopgrad}(\ell_{i,t})\right),
$$

where $\ell_{i,t}$ is the token log-probability computed by the worker at temperature 0.6. The exponential is numerically one but carries the policy gradient through its first argument. Each group gets one update opportunity; this is not a multi-epoch PPO loop. The denominator is the fixed group size times the maximum completion length, rather than each answer's realized length. There is no additional KL penalty in this main worker.

## Buffer and supervised replay

Admission runs **before** drawing the replay sample, so an answer from the current group can be used immediately. The buffer maps question IDs to a prompt and completion:

- Admit only a group with four verified-correct answers; select its shortest completion.
- Replace a stored completion for the same ID only if the new one is shorter.
- Keep at most 128 questions; evict the oldest inserted question when full.
- Draw one question uniformly from the current buffer, using a separate replay RNG.

For a sampled stored completion $y^*$ of length $L^*$:

$$
\mathcal L_{\mathrm{replay}}
= -\frac{1}{L^*}\sum_{t=1}^{L^*}\log p_\theta(y^*_t\mid x^*,y^*_{<t}),
\qquad
\mathcal L = \mathcal L_{\mathrm{policy}} + 0.1\mathcal L_{\mathrm{replay}}.
$$

Replay log-probabilities use temperature 1.0 and supervise only the completion. An empty buffer contributes no replay term. All-wrong groups have zero policy advantages; an existing buffer can still produce an update. If there is neither policy signal nor a replay sample, the optimizer step is skipped. Existing buffer entries are not reverified under each later student checkpoint.

The [pseudocode in IDEA.md](../IDEA.md#method) summarizes this loop.

## A small worked example

**Illustrative arithmetic, not measured model outputs.** Suppose the frozen domain reference is 1,000 tokens and all four answers pass the verifier:

| Answer length | Raw reward | Centered advantage | Replay selection |
|---:|---:|---:|---|
| 200 | 0.98 | +0.03 | Admitted as the shortest answer |
| 400 | 0.96 | +0.01 | — |
| 600 | 0.94 | −0.01 | — |
| 800 | 0.92 | −0.03 | — |

Here the policy term favors the shorter successful attempts, while supervised replay provides a concrete 200-token target. If the 200-token attempt instead fails verification, its reward is zero and the group supplies **no new replay entry**; its remaining successful attempts still provide relative policy signal. Existing buffered questions remain eligible for replay.

## Data and implementation

The [frozen plan](../configs/self-replay-plan.json) contains 3,840 questions balanced across math, coding, tools and instruction following, plus 256 independent development questions. The observed trajectory reached 1,280 training questions; checkpoint IDs count processed questions, including groups without an optimizer update. No external teacher supplies targets in this main lineage.

Training uses Qwen3-0.6B at revision `c1899de289a04d12100db370d81485cdf75e47ca`, LoRA rank 16 / alpha 32, AdamW with learning rate 1e-5 and zero weight decay, gradient-norm clipping at 1.0, and seed 17. Generation uses temperature 0.6, top-p 0.95, top-k 20 and a 2,048-token cap. The original run distributes the same four responses over two A100-SXM4-80GB GPUs; it does not multiply the global group size. Rank-local policy gradients are summed, and replay contributes once, on rank zero.

Complete checkpoints preserve adapter weights, optimizer, buffer, calibration and RNG state. The public inference adapters alone cannot reproduce exact continuation. Development monitoring is separate from the full official evaluation used for the reported results.

| Source | What to inspect |
|---|---|
| [Archived worker](../src/self_replay/methods/overlays/self-replay/experiments/replay_scaled/worker.py) | Calibration, admission-before-sampling, both losses and optimizer step |
| [Archived core](../src/self_replay/methods/overlays/self-replay/experiments/replay_scaled/core.py) | Fixed reward wiring and buffer admission/eviction |
| [Reward definitions](../src/self_replay/methods/research/experiments/rl_training/objectives.py) | Correctness gating and unscaled advantages |
| [Log-probability implementation](../src/self_replay/methods/research/experiments/rl_training/train.py) | Completion masking and temperature handling |
| [Source manifest](../provenance/self-replay-source.json) | Original archived source hashes |
| [Evaluation contract](EVALUATION.md) | Official metrics, uncertainty and compatibility changes |

## Evidence boundary

The measured result belongs to the **combined procedure and selected checkpoints**. There is no matched ablation of the final four-domain run isolating replay, the admission threshold or their interaction with the length reward. Earlier experiments informed these choices but cannot supply that attribution.

The all-correct gate may concentrate replay on easier questions and starve difficult coding tasks; the 2,048-token cap may worsen that bottleneck. These are mechanisms to investigate, not demonstrated causes of the result. A controlled follow-up would hold data, rollout budget and seeds fixed while removing replay, then changing only admission. It would also improve training-only no-call coverage and evaluate on an independent selection holdout.
