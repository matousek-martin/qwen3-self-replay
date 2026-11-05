#!/usr/bin/env bash
set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# Scoped workaround for this host's failed direct GPU reduction probe.
# Still require exact collective and full optimizer/resume checks before training.
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
source /workspace/project/experiments/setup_gpu.sh
python -m experiments.replay_scaled.preload
torchrun --standalone --nproc_per_node=2 -m experiments.replay_scaled.communication
torchrun --standalone --nproc_per_node=2 -m experiments.replay_scaled.worker --data data/coding-curriculum/kodcode --out output/preflight-reference --pilot
torchrun --standalone --nproc_per_node=2 -m experiments.replay_scaled.worker --data data/coding-curriculum/kodcode --out output/preflight-resumed --pilot --resume output/preflight-reference/checkpoints/group-000002
python -m experiments.replay_scaled.check --reference output/preflight-reference --resumed output/preflight-resumed --world 2 --out output/gpu-verification.json
for branch in kodcode taco; do
    export CURRICULUM_BRANCH="$branch"
    python -c 'import os,json;from pathlib import Path;Path("output/current-branch.json").write_text(json.dumps({"dataset":os.environ["CURRICULUM_BRANCH"]}))'
    torchrun --standalone --nproc_per_node=2 -m experiments.replay_scaled.worker --data "data/coding-curriculum/$branch" --out "output/$branch"
    python -c 'import os,json;from pathlib import Path;p=Path("output")/os.environ["CURRICULUM_BRANCH"]/"completion.json";assert json.loads(p.read_text())["complete"], "Incomplete branch; preserve checkpoint without starting another branch"'
done
