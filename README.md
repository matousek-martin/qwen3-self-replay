# Qwen3 Self-Replay

**31.23% fewer total output tokens, with a +0.21 percentage-point accuracy estimate.**

An independent submission to [BottleCapAI's ThinkingCap Test](https://github.com/BottleCapAI/ThinkingCap-Test), built on Qwen3-0.6B. The model learns from its own verified solutions: a correctness-gated length reward plus supervised replay of short solutions from consistently solved questions. Thinking remains enabled; inference uses the contest's unchanged settings.

| Release | Total tokens saved | Accuracy change | 95% accuracy interval | Intended choice |
|---|---:|---:|---:|---|
| **Self-Replay-256** | **31.23%** | **+0.21 pp** | **[−0.48, +0.88] pp** | Primary submission |
| Self-Replay-850 | 41.60% | −1.17 pp | [−1.96, −0.35] pp | More compression, measurable regression |

Full official evaluation: **2,759 questions × 8 responses**, equal weighting across SVAMP, MBPP+, BFCL and IFEval. Token savings use the official geometric aggregation. The 256 interval includes zero: this is neither proof of improvement nor a statistical equivalence claim.

![Official token–accuracy tradeoff](studies/self-replay/figures/official-tradeoff.png)

## Start here

- [IDEA.md](IDEA.md): the method, why it was tried, results and limitations.
- [Experiment findings](studies/README.md): negative results and what changed our decisions.
- [Reproduction](docs/REPRODUCING.md): installation, data, training and official evaluation.
- [Method recipes](docs/METHODS.md): all 24 runnable entrypoints and required inputs.
- [RunPod / Vast](docs/CLOUD.md): reusable job planning, monitoring, backup and release.
- [AI usage](AI_USAGE.md): human direction, substantial AI implementation and evidence-based validation.

## Quick start

Python 3.11+, from a source checkout:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,analysis]'
self-replay recipes
self-replay init workspaces/replay --recipe self-replay
self-replay train self-replay --workspace workspaces/replay --dry-run
python scripts/build_figures.py
pytest -q
```

These commands prepare local files, display the training command and rebuild plots; they do not rent a GPU. Install the pinned scientific runtime separately before actual training. CPU tests on tiny tensors are distinct from training the submitted model.

## Use the adapters

The release folders `Qwen3-0.6B-Self-Replay-256` and `Qwen3-0.6B-Self-Replay-850` contain ordinary PEFT adapters in safetensors format. Public hosting links are added only after upload; there are no placeholder model IDs to accidentally download.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_id = "Qwen/Qwen3-0.6B"
revision = "c1899de289a04d12100db370d81485cdf75e47ca"
tokenizer = AutoTokenizer.from_pretrained(base_id, revision=revision)
base = AutoModelForCausalLM.from_pretrained(base_id, revision=revision, torch_dtype="auto")
model = PeftModel.from_pretrained(base, "./Qwen3-0.6B-Self-Replay-256")
# Keep thinking enabled and apply the model's normal chat template.
```

For contest evaluation, use the pinned evaluator with `--adapter`; do not substitute a custom prompt, disable thinking or reduce the output cap.

## What the headline leaves out

- BFCL no-call accuracy falls by **4.17 pp at 256** and **6.88 pp at 850**, even where the aggregate looks favorable.
- The main trajectory has one training seed, and the benchmark was repeatedly inspected when selecting checkpoints.
- Later checkpoints reach 46.58% savings at 1280, but lose 2.35 pp accuracy; all evaluated points are retained.
- Development monitoring, historical benchmark subsets and full official evaluations are labeled separately.
- The main run uses no external teacher responses; separate teacher and SFT experiments are documented as separate lineages.

## Repository design

`src/self_replay/` provides the small public interface for data, methods, evaluation and cloud jobs. Scientific workers retain their existing module names inside an isolated runtime, with explicit historical source overlays. This keeps the numerical implementation auditable while making inputs, workspaces and cloud control reusable.

`configs/` holds recipes; `studies/` holds machine-readable results and analysis; `provenance/` binds claims to source and artifact hashes. Weights, response archives and prepared data are release assets, outside ordinary Git history. Original optimizer states are not included in the inference release.

The Git history begins with the actual BottleCapAI evaluator history, followed by a clearly dated research import and refactor. It does not pretend to be the original day-by-day experiment history.

## License and attribution

Own code and released adapters: Apache-2.0. Qwen3-0.6B and the contest evaluator retain their original attribution. Dataset, benchmark and vendored-code terms are documented in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Research data is not relicensed as software.
