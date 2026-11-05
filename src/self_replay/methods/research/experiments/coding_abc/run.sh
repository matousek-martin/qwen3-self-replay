#!/usr/bin/env bash
set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
source /workspace/project/experiments/setup_gpu.sh
python -m experiments.replay_scaled.preload
python -m experiments.coding_abc.worker
# full_replay_eval uses only the mandatory compatible cached official baseline.
python -m experiments.coding_abc.evaluate
