#!/usr/bin/env bash
set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
source /workspace/project/experiments/setup_gpu.sh
COUNT=$(python -c 'import json; print(json.load(open("worker-lease.json"))["gpu_count"])')
export REPLAY_WORLD="$COUNT"
MODE=$(python -c 'import json; print(json.load(open("data/replay-scaled/launch-mode.json"))["mode"])')
if [[ "$MODE" == "pilot" ]]; then
    torchrun --standalone --nproc_per_node="$COUNT" -m experiments.replay_scaled.worker --data data/replay-scaled --out output/pilot-reference --pilot
    torchrun --standalone --nproc_per_node="$COUNT" -m experiments.replay_scaled.worker --data data/replay-scaled --out output/pilot-resumed --pilot --resume output/pilot-reference/checkpoints/group-000002
    python -m experiments.replay_scaled.check --reference output/pilot-reference --resumed output/pilot-resumed --world "$COUNT" --out output/gpu-verification.json
    torchrun --standalone --nproc_per_node="$COUNT" -m experiments.replay_scaled.worker --data data/replay-scaled --out output/generation-benchmark --benchmark-generation
elif [[ "$MODE" == "training" ]]; then
    PREFLIGHT=$(python -c 'import json; print(int(json.load(open("data/replay-scaled/launch-mode.json")).get("verify_before_training",False)))')
    if [[ "$PREFLIGHT" == "1" ]]; then
        torchrun --standalone --nproc_per_node="$COUNT" -m experiments.replay_scaled.communication
        torchrun --standalone --nproc_per_node="$COUNT" -m experiments.replay_scaled.worker --data data/replay-scaled --out output/preflight-reference --pilot
        torchrun --standalone --nproc_per_node="$COUNT" -m experiments.replay_scaled.worker --data data/replay-scaled --out output/preflight-resumed --pilot --resume output/preflight-reference/checkpoints/group-000002
        python -m experiments.replay_scaled.check --reference output/preflight-reference --resumed output/preflight-resumed --world "$COUNT" --out output/gpu-verification.json
        cp output/gpu-verification.json data/replay-scaled/gpu-verification.json
    fi
    python -c 'import json; from experiments.replay_scaled.generation import validate_settings; v=json.load(open("data/replay-scaled/gpu-verification.json")); p=json.load(open("data/replay-scaled/plan.json")); g=validate_settings(p.get("generation")); assert v["passed"] and v["world_size"]==int(__import__("os").environ["REPLAY_WORLD"]) and v.get("generation_protocol")==g["protocol"] and v.get("generation_settings")==g' # Main launch requires proof of this generation configuration and world size.
    torchrun --standalone --nproc_per_node="$COUNT" -m experiments.replay_scaled.worker --data data/replay-scaled --out output/training
else
    echo 'Unregistered launch mode' >&2
    exit 2
fi
