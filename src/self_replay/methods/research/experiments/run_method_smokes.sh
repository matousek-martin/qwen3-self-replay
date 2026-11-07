#!/usr/bin/env bash
set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
source /workspace/project/experiments/setup_gpu.sh
export SELF_REPLAY_REPO=/workspace/project/data/public-checkout
export PYTHONPATH="$SELF_REPLAY_REPO/src${PYTHONPATH:+:$PYTHONPATH}"
if python -c 'import json; raise SystemExit(not json.load(open("/workspace/project/data/smoke/suite.json")).get("needs_shorterbetter", False))'; then
  ./uv venv --python /opt/thinkingcap-venv/bin/python /opt/shorterbetter-venv
  ./uv pip sync --python /opt/shorterbetter-venv/bin/python "$SELF_REPLAY_REPO/requirements/shorterbetter.lock"
  ./uv pip freeze --python /opt/shorterbetter-venv/bin/python > output/shorterbetter-packages.txt
fi
python "$SELF_REPLAY_REPO/scripts/gpu_method_smoke.py" \
  --suite /workspace/project/data/smoke/suite.json \
  --data /workspace/project/data/smoke \
  --lease /workspace/project/worker-lease.json \
  --out /workspace/project/output/method-smoke
