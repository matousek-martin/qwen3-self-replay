#!/usr/bin/env bash
# One finite technical job; never dispatch the 256-question training run here.
set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
source /workspace/project/experiments/setup_gpu.sh
export SELF_REPLAY_REPO=/workspace/project/data/public-checkout
export PYTHONPATH="$SELF_REPLAY_REPO/src${PYTHONPATH:+:$PYTHONPATH}"
python "$SELF_REPLAY_REPO/scripts/gpu_smoke.py" \
    --data /workspace/project/data/replay-scaled \
    --adapters /workspace/project/data/released-adapters \
    --questions /workspace/project/data/official-questions \
    --lease /workspace/project/worker-lease.json \
    --out /workspace/project/output/submission-smoke
