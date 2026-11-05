# Shared ThinkingCap data pipeline

`pipeline.py` prepares one shared prompt pool and creates matched SFT/DPO datasets from verified model rollouts. Preparation downloads the four contest-suggested training datasets; it does not generate model responses or train models.

## Prepare prompts

Run from the project root using the existing CPU environment:

```bash
.venv-lab/bin/python experiments/data_pipeline/pipeline.py prepare \
  --benchmarks lab/baseline-20261022T134220Z/inputs/code/questions/questions.jsonl \
  --out lab/experiments/data
```

The checked-in `sources.lock.json` pins each Hugging Face dataset to an exact commit. If absent, the first preparation resolves and records current commits before downloading. The lock records upstream license metadata; Skywork does not declare a consolidated dataset license in its pinned card, so its license is explicitly unknown, not assumed from its code repository. The dataset combines upstream sources with their own terms. No dataset redistribution is included in the experiment code.

The initial split uses the contest helper's source order, random seed 0, and 200 held-out questions per source. Filtering then removes exact and lexical near-duplicates against frozen evaluation **prompts only**, within the data, and between training and validation. Validation takes precedence over training. Filtered validation counts can therefore be below 200 per source. No removed validation question is replaced from training. The pilot selects 512 prompts balanced across available sources (128 each with the default sources), independently of validation.

Outputs:

- `train.jsonl`, `val.jsonl`: screened shared prompt sets.
- `pilot.jsonl`: 512 training prompts, intended for four initial rollouts each.
- `exclusions.jsonl`: removed prompt IDs, matching IDs, and lexical similarity; no evaluation answers or reasoning.
- `sources.lock.json`, `manifest.json`: revisions, source metadata, seeds, counts, code hashes, and content SHA-256 hashes.

Outputs refuse to overwrite a nonempty directory. Use a new output directory for a changed dataset. IDs include the source revision and row content, so a source change cannot silently reuse the same identity.

Near-duplicate screening uses normalized exact matching and five-token-shingle containment of at least 0.8 with at least three shared shingles. This catches lexical overlap, not all semantic paraphrases. It is a documented heuristic, not a claim that all contamination is eliminated. The current source mixture is math and multiple-choice QA; coding, function calling, and instruction-following training tasks need independently sourced/generated follow-up data. Their evaluation benchmarks remain excluded.

## Shared schemas

Prompt rows:

```json
{"id":"gsm8k-...","prompt":[{"role":"user","content":"..."}],"answer":"42","task":"math","source":"gsm8k","source_revision":"40-character commit","split":"train"}
```

Rollout rows supplied by a generation worker:

```json
{"prompt_id":"gsm8k-...","sample_id":0,"completion":"We calculate... </think>\nFinal answer: 42","total_tokens":34,"truncated":false,"finish_reason":"eos","model_revision":"c1899de289a04d12100db370d81485cdf75e47ca","generation":{"temperature":0.6,"top_p":0.95,"top_k":20,"max_new_tokens":2048,"seed":0}}
```

`completion` contains only newly generated text, not the prompt. An initial `<think>` may be absent when supplied by the thinking-mode chat template. A nonempty reasoning segment ending in exactly one `</think>` is required. `total_tokens` is the actual generated token count, including the final answer and emitted EOS if counted by the generator; do not infer it from characters. `sample_id` is unique within a prompt. `finish_reason` is `eos`, `stop`, or `length`. Explicit truncation status and generation provenance are required. Additional fields, such as generated token IDs, are preserved in the raw rollout artifact but are not required by the selector.

## Select verified targets

```bash
python3 experiments/data_pipeline/pipeline.py select \
  --prompts lab/experiments/data/pilot.jsonl \
  --rollouts lab/experiments/rollouts.jsonl \
  --out lab/experiments/selected
```

The selector regrades responses with the contest's **training** verifier. It does not trust a supplied correctness flag. Truncated, malformed, wrong-answer, and empty-reasoning responses are excluded. Wrong responses remain available in the original rollout pool for separate RL work.

Outputs:

- `sft-short.jsonl`: shortest correct response for each solvable prompt.
- `sft-random.jsonl`: random correct response for exactly the same prompts, using the fixed selection seed. Random selection can legitimately choose the shortest response.
- `dpo.jsonl`: shortest versus longest correct response for prompts with a genuine length difference. `chosen` and `rejected` are strings; `prompt` is the chat message list. Both answers pass the verifier.
- `selection-audit.jsonl`: acceptance/rejection per rollout.
- `manifest.json`: input and verifier hashes, output hashes/counts, and rollout-count distribution, including missing prompts.

SFT rows contain `prompt_id`, `prompt`, `completion`, `answer`, `task`, `source`, `sample_id`, `total_tokens`, and `model_revision`. DPO rows contain `prompt_id`, `prompt`, `chosen`, `rejected`, their sample IDs/token counts, and answer/task/source metadata. DPO naturally has fewer usable prompts than SFT; comparisons that claim identical training prompts must restrict SFT to the DPO prompt IDs.

Empty success pools produce empty training outputs rather than inventing successful examples. Trainers should reject empty datasets. Missing rollout groups are visible in the manifest; they are never silently resampled.

## Checks

```bash
python3 -m unittest discover -s experiments/data_pipeline -v
```

Tests cover stable/disjoint splits, balanced pilots, benchmark and validation duplicate exclusion, independent regrading, truncation and malformed reasoning, all-wrong groups, tied-length preference pairs, and provenance errors. Tests do not require model libraries or a GPU.

## Student-compatible SFT proxy

After `offline_training/score_compatibility.py` scores the complete verified pool, add `--compatibility-scores path/to/compatibility.jsonl` to `select`. The companion `.manifest.json` must match the prompt/rollout hashes and model revision. Every eligible rollout must have exactly one finite score with matching identity, token count, and revision. Partial scoring (for example after a time limit) fails instead of changing the training prompt mix.

This additionally writes `sft-compatible.jsonl`, selecting the highest verified-answer mean log probability per prompt, with ties resolved toward shorter responses. Its prompt IDs match `sft-short.jsonl` and `sft-random.jsonl`. This is a frozen-student scoring proxy; it does not rewrite explanations or reproduce ReDiST, and answer likelihood does not establish teaching utility.

## Frozen validation pilot

```bash
python3 experiments/data_pipeline/pipeline.py validation-pilot \
  --validation lab/experiments/data/val.jsonl \
  --out lab/experiments/data/val-pilot.jsonl
```

This writes 64 held-out questions (16 per source), plus `val-pilot.manifest.json` containing the parent validation hash, pilot hash, seed, and source counts. It refuses training rows and existing output paths. The original shared dataset manifest remains unchanged; this derived artifact has its own manifest.
