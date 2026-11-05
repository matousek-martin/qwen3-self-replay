# ThinkingCap RL pilots

[Project context](../../README.md) · [Replay study](../../studies/replay/README.md). Model training requires verified RTX 4090. Read the study handoff before launching or resuming campaign work; standalone examples below do not register a new campaign.

This implements small, single-GPU, one-update-per-group LoRA experiments starting from Qwen3-0.6B revision `c1899de289a04d12100db370d81485cdf75e47ca`. It saves ordinary PEFT adapters. It does not rent GPUs, modify the contest evaluator, or submit results.

Run from `/srv/projects/llm-hub` in an environment with torch 2.8, Transformers 4.57.1, PEFT 0.17.1 and Accelerate 1.10.1:

```bash
python -m unittest experiments.rl_training.test_objectives experiments.rl_training.test_torch -v
python -m experiments.rl_training.train --data PATH/pilot-train.jsonl --out PATH/dr-control --optimizer dr_grpo --variant correctness --max-steps 32 --max-seconds 0
```

Use `--dry-run` to validate the dataset and print a configuration without importing GPU libraries. Output directories must be empty. Input rows must contain unique `id`, `prompt` chat messages, `answer`, `task`, `source`, and explicit `split: "train"`. This trainer currently supports the contest's math and multiple-choice training verifiers. It must receive the decontaminated prepared training file; it does not independently repeat benchmark decontamination.

## Controlled comparisons

| Variant | Hypothesis and comparison |
| --- | --- |
| `correctness` | Binary success only; compare `--optimizer grpo` and `--optimizer dr_grpo`. |
| `fixed` | Correctness-gated length cost, with `--length-coefficient 0`, `0.05`, `0.1` or `0.2`. |
| `adaptive` | Penalize correct responses longer than the shortest correct response in that prompt's group. |
| `difficulty` | Multiply the fixed length coefficient by that group's success rate. |
| `progressive` | Correctness-only reward, decreasing the training generation cap from `--max-new-tokens` to `--final-cap`. |
| `replay` | NLL replay of concise solutions from previously all-correct groups, added to correctness-only RL by default or fixed length shaping with `--replay-reward fixed`. |

Reward variants are explicit local hypotheses, not exact reproductions of ShorterBetter, Apriel, ThinkPrune or Length Self-Distillation. Replay does not implement an EMA teacher. Rewards require nonempty closed reasoning, a complete EOS-terminated response, and the contest training verifier's success. Wrong responses remain in the group with reward zero. No unsuccessful group is silently resampled.

To test preservation **while shortening**, use `--variant replay --replay-reward fixed` with exactly the same `--references`, `--length-coefficient`, optimizer and learning rate as the fixed-reward control. Only the replay auxiliary loss then changes. Replay sampling has an independent random stream so it cannot change prompt order in later epochs. The default `--replay-reward correctness` remains the simpler preservation control. KL metrics are null when no reference-policy KL was computed; a disabled KL term is not evidence of zero divergence.

GRPO uses per-response length normalization and group-standard-deviation reward scaling. Dr. GRPO uses the fixed initial maximum completion length and no standard-deviation scaling. To isolate these two differences, use `--scale-rewards yes` or `--scale-rewards no` to hold scaling constant. With a single on-policy update per group, the importance ratio is `exp(logp - stop_gradient(logp))`; clipping is inactive at ratio one. There are no reused/off-policy trajectories. This follows the single-update formulation and normalization distinctions documented in the [TRL GRPO reference](https://huggingface.co/docs/trl/grpo_trainer).

Generation defaults to temperature 0.6, top-p 0.95 and top-k 20. The loss uses temperature-scaled full-vocabulary log probabilities, as a conventional approximation to top-k/top-p sampling; it does not differentiate the truncated sampling support. The approximation is identical across variants. `--temperature 1 --top-p 1 --top-k 0` removes that mismatch if used for every compared training arm. These are training settings; leave contest evaluation settings unchanged.

`--references FILE` accepts a JSON map from training source to mean length measured on base-model training rollouts. Freeze and share that file across fixed and difficulty variants. The fallback `--length-reference 1024` is for smoke tests, not an empirically measured reference. Only the adaptive reward uses a moving group-derived target. Difficulty estimates from four samples are noisy; report that limitation.

## Cost, artifacts and verification

`--max-seconds 0` (the default) has no elapsed-time deadline: the trainer finishes its requested groups unless it receives a stop request. Positive values retain the optional elapsed-time limit, checked while generating and between groups. SIGTERM/SIGINT always request a checkpointed stop. Model loading, a backward pass, and checkpoint writing can exceed a positive soft limit. Infrastructure budget enforcement remains separate; the trainer cannot enforce a dollar cap.

For future processes, optional `--runtime-control /path/control.json` permits explicit policy updates. Write `{"max_seconds": 0}` to remove the trainer deadline, `{"max_seconds": 14400}` to set a total elapsed-from-start limit, or `{"stop": true}` to request a graceful checkpointed stop. The file is checked at most once per second. Invalid updates retain the last valid policy and their error is recorded in `status.json`; atomically replacing the JSON file is recommended. Stop requests are sticky. This feature cannot change an already-running older trainer or override external infrastructure/budget enforcement. Existing processes launched under the old code retain their original four-hour limit until completion or stopping.

`metrics.jsonl` records success density, all-wrong/all-correct and zero-variance groups, output lengths, truncation, invalid reasoning, loss, gradient norm, sampled entropy/KL where computed, memory and elapsed time. `rollouts.jsonl` preserves all generated attempts. It distinguishes attempted groups from actual optimizer updates: correctness-only zero-variance groups with zero KL do not update. `--max-steps` counts groups, giving variants the same attempted rollout budget; it does not promise the same number of effective updates.

`adapter/` contains the final standard adapter, tokenizer and small state summary. Periodic checkpoints use `checkpoint-NNNN/`. Saves include model-weight hashes in `status.json`. Checkpoints do not contain optimizer or RNG state and must not be represented as exact resumable runs. Training does not select checkpoints using evaluation benchmarks. Use the separately held-out validation set to choose candidates, then the untouched contest evaluator for final measurements.

The pure-Python objective tests run on the control server. `test_torch` compares the chunked output projection and adapter gradients against a dense forward pass using a tiny random Qwen3, without network downloads. Passing that test validates the tensor implementation, not the scientific performance of any experiment; a real-model GPU smoke run is still required before expanding pilots.
