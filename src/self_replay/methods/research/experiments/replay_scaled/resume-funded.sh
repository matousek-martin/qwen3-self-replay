#!/usr/bin/env bash
set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export THINKINGCAP_REUSE_ENV=1
source /workspace/project/experiments/setup_gpu.sh
COUNT=$(python -c 'import json; print(json.load(open("worker-lease.json"))["gpu_count"])')
CHECKPOINT=$(python -c 'import json; print(json.load(open("runtime-resume.json"))["checkpoint"])')
export REPLAY_WORLD="$COUNT"
torchrun --standalone --nproc_per_node="$COUNT" -m experiments.replay_scaled.worker --data data/replay-scaled --out output/training --resume "$CHECKPOINT"
