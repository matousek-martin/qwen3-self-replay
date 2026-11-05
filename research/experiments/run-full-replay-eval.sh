#!/usr/bin/env bash
set -euo pipefail
source /workspace/project/experiments/setup_gpu.sh
export NLTK_DATA=/root/thinkingcap_nltk_data
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
mkdir -p -m 700 "$NLTK_DATA"
chmod 700 "$NLTK_DATA"
python -m experiments.vllm_compat --out output/vllm-compat.json
python -m experiments.vllm_lora_smoke --config data/study/full-replay-evaluation.json --out output/vllm-lora-smoke.json
python -m experiments.full_replay_eval --config data/study/full-replay-evaluation.json --out output/full-evaluation
