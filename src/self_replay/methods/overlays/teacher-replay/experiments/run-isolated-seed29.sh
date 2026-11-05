#!/usr/bin/env bash
set -euo pipefail
source /workspace/project/experiments/setup_gpu.sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m experiments.teacher_generation.paired_worker --seed 29
