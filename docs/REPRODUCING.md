# Reproducing the work

## Install and inspect

Use Python 3.11 from a source checkout, with `pip install -e '.[dev,analysis]'`. The lightweight interface has no model dependency. Scientific runs use the recorded requirements/gpu.lock and separate method-specific constraints; ShorterBetter uses requirements/shorterbetter.lock (TRL 0.24.0 and vLLM 0.10.2), a separate environment from the main replay stack. Do not upgrade the inference stack and then label the result the original contract.

`self-replay init workspaces/replay --recipe self-replay` creates an independent workspace containing scientific source, the evaluator, configuration and empty data/output directories. It copies neither credentials nor historical cloud spending authorizations. The workspace source manifest detects later changes. A relative compatibility symlink maps the older preparers’ lab/experiments/data path to the same data directory used by workers.

## Main self-replay data

The separate `self-replay-frozen-data.zip` asset contains the exact original plan, 3,840 balanced training questions and 256 development questions, plus source provenance. These prepared data retain their upstream licenses and attribution. Their hashes match configs/self-replay-plan.json; no benchmark examples become training rows.

```bash
python -m zipfile -e self-replay-frozen-data.zip artifacts/frozen-data
self-replay stage-data artifacts/frozen-data workspaces/replay/data/replay-scaled
self-replay train self-replay --workspace workspaces/replay --dry-run
```

To regenerate from upstream instead, the preserved data pipeline exposes pinned download/splitting/screening and the four-domain preparation stages. `self-replay prepare initial --workspace WORKSPACE -- --help` and the corresponding `four-domain`/`self-replay` commands describe their inputs. Later data preparations intentionally retain earlier exclusions and splits: provide the parent manifest rather than resampling a new split and calling it identical.

Preparation records exact/lexical benchmark overlap exclusions, source revisions, split IDs, reference checks and hashes. Lexical screening does not prove zero semantic overlap. No-call development coverage is small; coding references are sandbox-checked, not assumed correct merely because a dataset provides them.

## Main training

The default main recipe processes up to 256 questions with the original 2×A100 scientific implementation. A finite worker lease identifying a newly approved job and verified hardware is required. Use the cloud workflow or an equivalently explicit local lease on already available hardware. The command verifies actual GPU names before executing the trainer.

To reproduce the longer trajectory, use the same immutable inputs and complete state with an explicit later stop boundary, for example append `-- --resume /absolute/group-000256 --stop-after 850`. An adapter-only inference download cannot provide optimizer/replay/RNG resume. The complete states remain research archives; do not claim exact continuation from only a safetensors adapter.

The source overlay preserves the main historical numerical code. New orchestration/source locks are labeled as refactor-time records, not rewritten historical evidence. Multi-GPU parity tests use tiny CPU tensors; historical A100 preflight evidence is separate from fresh GPU validation.

## Other methods

All original method implementations are retained and exposed in [METHODS.md](METHODS.md). They need their declared prepared inputs, parent checkpoints and, for teacher methods, the pinned teacher/cache. They are not interchangeable datasets. `--dry-run` validates source materialization and prints a command; it does not certify a trained result.

For the isolated historical branches, init retains historical source locks and writes a distinct current source lock/change record. Frozen benchmark assets remain required with their expected hashes; they are downloaded/staged separately and used for evaluation only. Exact historical response data and source fingerprints are recorded in studies/experiments.

## Verify locally

```bash
pytest -q
python scripts/check_science.py
python scripts/check_numerics.py
python scripts/build_figures.py
python scripts/reproduce_results.py
python scripts/verify_release.py
```

The scientific/numerical suites need a CPU environment with torch 2.8, Transformers 4.57.1, PEFT 0.17.1 and numpy; requirements/cpu-test.txt describes it. Tests use synthetic tensors/models, never a training run of the submitted checkpoint. Response reproduction needs the released response assets. Release verification also needs the local model assets.

No full training or fresh official inference is automatically launched by these checks. New paid GPU smoke tests have a separate funding decision.
