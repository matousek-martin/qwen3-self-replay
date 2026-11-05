#!/usr/bin/env bash
set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
source /workspace/project/experiments/setup_gpu.sh
export NLTK_DATA=/root/thinkingcap_nltk_data
mkdir -p -m 700 "$NLTK_DATA"
# full_replay_eval uses the mandatory compatible baseline cache.
python -m experiments.opencoder_sft.campaign
