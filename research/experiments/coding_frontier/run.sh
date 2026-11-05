#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
source /workspace/project/experiments/setup_gpu.sh
python -m experiments.replay_scaled.preload
python -m experiments.vllm_compat --out output/vllm-compat.json
python -m experiments.coding_frontier.worker
