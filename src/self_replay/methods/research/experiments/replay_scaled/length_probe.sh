#!/usr/bin/env bash
set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
source /workspace/project/experiments/setup_gpu.sh
python -m experiments.replay_scaled.length_probe --preload
python -m experiments.replay_scaled.length_probe
