# Reproducing the work

## Install

Use Python 3.11 from a source checkout:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,analysis]'
pytest -q
```

Model execution needs the pinned scientific environment in `requirements/gpu.lock`. ShorterBetter uses its separate `requirements/shorterbetter.lock`. The source-checkout interface is supported; a standalone wheel is not.

## Check the submitted results without a GPU

Extract the release's response archive into `artifacts/records`, then run:

```bash
python scripts/reproduce_results.py
```

This verifies the saved responses and recomputes the official comparisons and confidence intervals for checkpoints 256 and 850. No new answers are generated. [EVALUATION.md](EVALUATION.md) records the evaluator settings, baseline reuse and compatibility patches.

## Prepare the main training recipe

Download `self-replay-frozen-data.zip` from the release. It contains the original plan, 3,840 training questions, 256 development questions and dataset provenance.

```bash
self-replay init workspaces/replay --recipe self-replay
python -m zipfile -e self-replay-frozen-data.zip artifacts/frozen-data
self-replay stage-data artifacts/frozen-data workspaces/replay/data/replay-scaled
self-replay train self-replay --workspace workspaces/replay --dry-run
```

`init` creates an independent workspace with source, evaluator and configuration; `stage-data` verifies the frozen inputs. Neither command starts training. The default recipe processes 256 questions on 2×A100-SXM4-80GB. Actual execution requires verified hardware and a finite worker lease; [CLOUD.md](CLOUD.md) explains the managed workflow.

For continuation, append `-- --resume /absolute/group-000256 --stop-after 850` using a complete checkpoint. The released inference adapters do not include Adam, replay or per-rank RNG state and cannot provide exact training continuation.

To rebuild training data from upstream, inspect `self-replay prepare initial --workspace WORKSPACE -- --help` and the `four-domain`/`self-replay` preparers. They require parent manifests to preserve the original exclusions and splits. Overlap screening is lexical, not proof of zero semantic contamination.

## Tests and other recipes

```bash
python scripts/check_science.py
python scripts/check_numerics.py
python scripts/verify_release.py
```

Science/numerical checks require the CPU stack in `requirements/cpu-test.txt`; release verification also requires the staged model assets. The dated [GPU reports](GPU_SMOKE.md) cover bounded training, resume and inference checks, not full retraining of the submitted checkpoint.

[METHODS.md](METHODS.md) lists 24 historical entrypoints and their required inputs. Newer balanced/fresh/domain branches and component-ablation wrappers remain in the research archive; they are not supported public CLI recipes. Some historical recipes also require parent checkpoints or teacher caches that are separate from the submission.

## Source history

Git preserves the upstream evaluator and later import/refactor commits. Earlier experiment dates and implementations come from archived source manifests and result hashes in `provenance/`; the public refactor is not the historical training source. [The method specification](METHOD.md) links the implementation to those records.
