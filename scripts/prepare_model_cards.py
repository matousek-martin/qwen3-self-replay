#!/usr/bin/env python3
"""Prepare HF-ready folders without modifying original checkpoint evidence."""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from self_replay.artifacts import sha256, write_json
from self_replay.paths import repository


def main():
    root = repository()
    for n, saving, delta, interval in [
        (256, "31.23%", "+0.21", "[−0.48, +0.88]"),
        (850, "41.60%", "−1.17", "[−1.96, −0.35]"),
    ]:
        src = root / f"artifacts/models/checkpoint-{n}"
        dst = root / f"artifacts/huggingface/Qwen3-0.6B-Self-Replay-{n}"
        shutil.copytree(src, dst, dirs_exist_ok=True)
        shutil.copy2(root / "LICENSE", dst / "LICENSE")
        shutil.copy2(root / "THIRD_PARTY_NOTICES.md", dst / "THIRD_PARTY_NOTICES.md")
        shutil.copy2(
            root / "studies/self-replay/results" / f"checkpoint-{n}/results.json",
            dst / "results.json",
        )
        shutil.copy2(
            root / "studies/self-replay/results" / f"checkpoint-{n}/comparison.json",
            dst / "comparison.json",
        )
        description = (
            "Primary ThinkingCap submission: the accuracy interval includes zero; this is not proof of improvement or equivalence."
            if n == 256
            else "Alternative with stronger compression and a measured accuracy regression; use checkpoint 256 when accuracy preservation matters more."
        )
        card = f"""---
license: apache-2.0
base_model: Qwen/Qwen3-0.6B
base_model_relation: adapter
library_name: peft
pipeline_tag: text-generation
language:
- en
tags:
- qwen3
- lora
- reinforcement-learning
- self-replay
- thinkingcap
- reasoning
---

# Qwen3-0.6B Self-Replay {n}

**{saving} fewer total output tokens; {delta} percentage-point accuracy change** under the official ThinkingCap protocol. {description}

This is a roughly 40 MB PEFT LoRA adapter for Qwen3-0.6B, not a standalone full model. The checkpoint number means processed training questions.

## Results

| Quantity | Value |
|---|---|
| Total-output-token savings | {saving} |
| Macro accuracy change | {delta} pp |
| Paired-prompt 95% accuracy interval | {interval} pp |
| Evaluation | 2,759 questions × 8 responses; seed 9001 |
| Domains | SVAMP, MBPP+, BFCL, IFEval; official equal-domain weighting |
| Hardware | 2×A100-SXM4-80GB training; RTX 4090 evaluation |

The unmodified evaluator's `results.json` and original `comparison.json` are included. Savings use its geometric aggregation and include total output tokens, not only thinking tokens. The compatible cached baseline retains its original provenance; it was not regenerated on every physical worker.

## Method and training

Correctness-gated length reward, four self-generated responses per question, unscaled Dr.GRPO advantages, and supervised replay of the shortest correct response only when all four are correct. Replay capacity 128, auxiliary weight 0.1; LoRA rank 16, alpha 32, learning rate 1e-5, seed 17. Main training caps output at 2,048 tokens. No external teacher generated the main training targets.

The frozen plan contains 3,840 balanced math/coding/tools/instruction questions and 256 independent development questions. This checkpoint processed {n} questions. Competition benchmark examples are excluded from training; exact/lexical screening is a heuristic, not a semantic-contamination guarantee.

## Use

Install the pinned scientific dependencies from the source repository for reproducibility. For ordinary local inference:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_id = "Qwen/Qwen3-0.6B"
revision = "c1899de289a04d12100db370d81485cdf75e47ca"
tokenizer = AutoTokenizer.from_pretrained(base_id, revision=revision)
base = AutoModelForCausalLM.from_pretrained(base_id, revision=revision, torch_dtype="auto")
model = PeftModel.from_pretrained(base, "./Qwen3-0.6B-Self-Replay-{n}")
```

Keep the original chat template and thinking enabled. Contest evaluation must retain official sampling and output caps. This adapter does not include optimizer, scheduler or replay/RNG state and cannot by itself provide exact training continuation.

## Limitations

One main training seed; repeated benchmark inspection and checkpoint selection. BFCL no-call accuracy drops {("4.17" if n == 256 else "6.88")} pp. Aggregate results hide domain and subset regressions. Not validated for high-stakes or general production deployment. Other negative experiments, later checkpoints and development curves are retained in the code release.

## Attribution and AI assistance

Martin Matousek directed the research and decisions. AI assistance contributed substantially to implementation, debugging, analysis and documentation; source/evidence hashes and local checks support the reported claims. This does not imply independent human review of every line. See the source release's AI_USAGE.md and IDEA.md.

Base model: [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B), pinned above. Evaluation: [BottleCapAI/ThinkingCap-Test](https://github.com/BottleCapAI/ThinkingCap-Test). Original code and adapters are Apache-2.0; upstream dataset terms remain separate in THIRD_PARTY_NOTICES.md.
"""
        (dst / "README.md").write_text(card)
        files = {
            p.name: sha256(p)
            for p in sorted(dst.iterdir())
            if p.is_file() and p.name != "release-manifest.json"
        }
        write_json(
            dst / "release-manifest.json",
            {
                "checkpoint": n,
                "base_revision": "c1899de289a04d12100db370d81485cdf75e47ca",
                "adapter_source_sha256": sha256(src / "adapter_model.safetensors"),
                "files": files,
            },
        )
        print(dst)


if __name__ == "__main__":
    main()
