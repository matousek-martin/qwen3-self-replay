#!/usr/bin/env bash
set -euo pipefail
source /workspace/project/experiments/setup_gpu.sh
export NLTK_DATA=/root/thinkingcap_nltk_data
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
mkdir -p -m 700 "$NLTK_DATA"
chmod 700 "$NLTK_DATA"
python -m experiments.replay_round2 worker --trial control-seed17
