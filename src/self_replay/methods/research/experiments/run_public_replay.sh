#!/usr/bin/env bash
# Public fresh-run entrypoint; the historical worker/objective remain unchanged.
set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
source /workspace/project/experiments/setup_gpu.sh
COUNT=$(python -c 'import json; print(json.load(open("worker-lease.json"))["gpu_count"])')
export REPLAY_WORLD="$COUNT"
torchrun --standalone --nproc_per_node="$COUNT" -m experiments.replay_scaled.communication
torchrun --standalone --nproc_per_node="$COUNT" -m experiments.replay_scaled.worker --data data/replay-scaled --out output/preflight-reference --pilot
torchrun --standalone --nproc_per_node="$COUNT" -m experiments.replay_scaled.worker --data data/replay-scaled --out output/preflight-resumed --pilot --resume output/preflight-reference/checkpoints/group-000002
python -m experiments.replay_scaled.check --reference output/preflight-reference --resumed output/preflight-resumed --world "$COUNT" --out output/gpu-verification.json
python -c 'import json; assert json.load(open("output/gpu-verification.json"))["passed"]'
torchrun --standalone --nproc_per_node="$COUNT" -m experiments.replay_scaled.worker --data data/replay-scaled --out output/training --stop-after 256
