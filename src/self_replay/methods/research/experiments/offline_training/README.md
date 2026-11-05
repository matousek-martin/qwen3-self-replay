# ThinkingCap offline training pilots

[Project context](../../README.md) · [DPO study](../../studies/dpo/README.md). Training requires verified RTX 4090; inference/validation may use other hardware or hosted APIs. Commands below are standalone examples, not the live registered campaign configuration.

These scripts generate a shared rollout pool, train LoRA SFT/DPO adapters, and
score separate training-source validation questions. They do not replace the
contest evaluation or launch/rent GPUs. The base is `Qwen/Qwen3-0.6B` at revision
`c1899de289a04d12100db370d81485cdf75e47ca` with thinking enabled.

Use the project's existing GPU lock (`lab/gpu-requirements.lock`): torch 2.8.0,
Transformers 4.57.1, PEFT 0.17.1, Accelerate 1.10.1, vLLM 0.11.0. Run from the
project root; model training must use RTX 4090 under the current project policy. GPU process isolation is required: vLLM
reserves most device memory, so run one generation/training process per GPU.

## Generate the shared pool

```bash
python experiments/offline_training/generate.py \
  --prompts experiments/data/pilot-train.jsonl \
  --output experiments/runs/base-rollouts.jsonl \
  --samples 4 --max-new-tokens 4096 --backend vllm --batch-size 16 \
  --max-seconds 3600
```

The prompt file contains `{id, prompt, answer, task, source, split}` rows, with
`prompt` a list of chat messages. Paths in these examples are placeholders;
point them at the data-pipeline outputs. Generation writes each sample with
its prompt ID, sample index, raw completion, token IDs/count, stop reason,
truncation flag, pinned model revision, seed and generation settings.

Re-running the exact command resumes missing prompt/sample pairs. It refuses
to mix input hashes, models, adapters, seeds or sampling settings. vLLM seeds
are per prompt for the group of samples; Transformers seeds are per sample.
Do not treat the two engines as bitwise-equivalent. `--backend transformers`
is a slower fallback. Generation does not overwrite incomplete or malformed
JSONL; investigate such an interruption before resuming.

`--max-seconds` is a **soft** wall-clock cap. vLLM checks between bounded batches;
Transformers checks during generation. An external supervisor must enforce the
financial deadline and stop the billed GPU. If that deadline ends a Transformers
response, its `finish_reason` is `deadline` and it is excluded from successful
training examples.

## Train matched SFT and DPO pilots

After the shared data pipeline selects verified examples:

```bash
python experiments/offline_training/train.py \
  --method sft --data experiments/data/sft-short.jsonl \
  --output experiments/runs/sft-short --steps 100 --grad-accum 8 \
  --max-seconds 3600

python experiments/offline_training/train.py \
  --method sft --data experiments/data/sft-random.jsonl \
  --output experiments/runs/sft-random --steps 100 --grad-accum 8 \
  --max-seconds 3600

python experiments/offline_training/train.py \
  --method dpo --data experiments/data/dpo.jsonl \
  --output experiments/runs/dpo --steps 100 --grad-accum 8 \
  --max-seconds 3600
```

SFT expects `{prompt_id, prompt, completion}`. DPO expects
`{prompt_id, prompt, chosen, rejected}`. Extra source/provenance fields survive
in the input dataset hash; `split`, if present, must be `train`. Every sequence
is encoded before optimization. Any input exceeding `--max-seq-len` fails the
run rather than silently dropping or truncating examples. Training uses
completion-only supervision including EOS, micro-batch one, gradient accumulation,
LoRA dropout zero, and constant AdamW learning rate. SFT averages loss over each
completion; DPO uses the standard **summed** sequence log probabilities with
the pinned, adapter-disabled base as the frozen reference.

The Qwen backbone produces hidden states without full sequence-wide logits.
The LM head and cross-entropy are checkpointed in chunks of 64 tokens to reduce
peak memory. Transformer layers also use non-reentrant gradient checkpointing.
The implementation is intentionally specific to the fixed Qwen3 architecture.

All SFT controls should use matching prompt order, seed, update count and
accumulation. Shorter examples process fewer tokens, which is recorded along
with wall time. If a soft deadline stops a run before its update target, its
`result.json` says `complete: false`; compare matched checkpoints or resume.
Use `--resume` with the original settings to restore the last checkpoint,
optimizer, step and RNG state. Soft time limits apply per invocation.

Outputs include a provenance manifest, dataset summary, per-update metrics,
checkpoint adapters in `.safetensors`, local optimizer state, `latest.json`,
and `result.json`. `result.json` names the exact adapter path to evaluate.
Only load local optimizer state produced by these scripts: it uses trusted
PyTorch checkpoint deserialization.

## Held-out training validation

Run both the unmodified model and each adapter on the exact same held-out
prompts, cap, engine and sample count:

```bash
python experiments/offline_training/generate.py \
  --prompts experiments/data/pilot-val.jsonl \
  --output experiments/runs/val-base.jsonl --samples 1

python experiments/offline_training/generate.py \
  --prompts experiments/data/pilot-val.jsonl \
  --output experiments/runs/val-short.jsonl --samples 1 \
  --adapter experiments/runs/sft-short/checkpoint-000100

python experiments/offline_training/score.py \
  --prompts experiments/data/pilot-val.jsonl \
  --rollouts experiments/runs/val-short.jsonl \
  --output experiments/runs/val-short-results.json --expected-samples 1
```

Scoring rejects missing or duplicate samples. It reports accuracy, total token
length, truncation, malformed reasoning, and lengths conditioned on success per
training source. Its main accuracy requires a nonempty, closed thinking segment
and a verifier-passing answer. Answer accuracy without that format gate is also
reported. These are **training-source validation figures**, not contest scores;
the contest's official evaluation remains the final arbiter across four domains.

## Exploratory student-compatibility proxy

```bash
python experiments/offline_training/score_compatibility.py \
  --prompts experiments/data/pilot-train.jsonl \
  --rollouts experiments/runs/base-rollouts.jsonl \
  --output experiments/runs/compatibility-scores.jsonl \
  --include-completion-nll --max-seconds 1800
```

The frozen base scores the verified answer conditioned on each successful
candidate's reasoning and a fixed answer header. Original final-answer text is
removed. Scores are mean gold-answer log probability and optionally mean whole
completion NLL. Preparation time and memory are logged separately. Select the
highest-scoring candidate for each prompt only after every eligible candidate is
scored, and retain the same prompt pool as the short/random SFT controls.

This is an exploratory **proxy**, not a reproduction of ReDiST: it neither
rewrites traces nor uses the paper's full construction and loss recipe. The
reasoning can already mention the answer, so high conditional probability does
not prove that a trace is understandable or useful for teaching. Its value must
be judged by downstream held-out accuracy/token tradeoffs.

## Tests and remaining validation

```bash
python -m unittest discover -s experiments/offline_training -p 'test_*.py' -v
```

Eight stdlib tests check token boundaries, EOS, truncation refusal, deterministic
seeds, validation completeness and answer-prefix leakage. Two CPU tests with
torch/Transformers/PEFT check tiny random Qwen3 chunked loss and LoRA-gradient
equivalence to full logits, and the initial DPO objective/gradient. These ten
tests passed during implementation. They do not establish CUDA memory behavior,
vLLM compatibility or end-to-end real-model quality; run a one-step GPU smoke
and adapter reload/generation before larger pilots.
