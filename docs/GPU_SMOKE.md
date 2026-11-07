# GPU smoke checks

`scripts/gpu_smoke.py` tests the public submission interfaces on an already leased
2×A100-SXM4-80GB worker. It does not allocate hardware or authorize spending.

The suite materializes the `self-replay` recipe, verifies and stages the frozen
training inputs, then calls the public CLI to run the four-question synthetic
pilot and resume it from checkpoint 2. The archived checker requires bitwise-equal
final adapter and Adam state, replay state and per-rank RNG; it also requires a
2,048-token backward check and actual replay/optimizer activity. Pilot generations
use the original 256-token technical cap, not the scientific training cap.

A separate four-question probe exercises the non-pilot training path at the full
2,048-token cap, with one frozen training row and one development row per domain.
It uses a separately hashed technical plan and one calibration question per domain,
so its calibration and cohort differ from the scientific run. The original source
dataset, reward, replay rule and released checkpoints remain intact. At least one
optimizer update and a complete SHA-verified final checkpoint are required.

For both released adapters (256 and 850), the suite verifies model-file hashes,
tests base→adapter→base vLLM routing and measurable adapter effect, and invokes
`self-replay evaluate -- --smoke` with the original eight responses and token caps.
All four official domains must generate and grade without a failed benchmark.
The base routing check is a tiny synthetic inference check, not regeneration of
the full official baseline. The official smoke cohort remains evaluation-only.

Stage a clean source checkout, the released adapters, frozen training data and
the pinned official question/grader directory. On the worker, install the pinned
GPU requirements and provide the active finite worker lease:

```bash
PYTHONPATH=/path/to/checkout/src python /path/to/checkout/scripts/gpu_smoke.py \
  --data /path/to/frozen-data \
  --adapters /path/to/released-adapters \
  --questions /path/to/official-questions \
  --lease /path/to/worker-lease.json \
  --out /path/to/fresh-output
```

The adapter directory contains `checkpoint-256/` and `checkpoint-850/`. The cloud
entrypoint `experiments/run_submission_smoke.sh` expects these inputs and the source
checkout beneath the staged `data/` directory. It deliberately invokes only the
technical pilot, never the 256-question scientific training job.

`report.json` records each subprocess, duration and outcome; the original evaluator
and resume checker write their own evidence. A successful worker run is only part
of the lifecycle: the controller must separately verify backup and provider release.
These checks establish execution and recovery behavior, not model quality, a full
benchmark rerun, or fresh validation of both cloud providers.
