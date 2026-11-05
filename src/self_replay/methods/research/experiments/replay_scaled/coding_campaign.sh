#!/usr/bin/env bash
set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
source /workspace/project/experiments/setup_gpu.sh
export NLTK_DATA=/root/thinkingcap_nltk_data
mkdir -p -m 700 "$NLTK_DATA"
# Includes full_replay_eval; compatible official baseline cache is mandatory.
python -m experiments.replay_scaled.coding_campaign
