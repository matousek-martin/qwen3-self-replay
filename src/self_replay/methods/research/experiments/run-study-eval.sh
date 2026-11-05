#!/usr/bin/env bash
set -euo pipefail
source /workspace/project/experiments/setup_gpu.sh
export NLTK_DATA=/root/thinkingcap_nltk_data
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
mkdir -p -m 700 "$NLTK_DATA"
chmod 700 "$NLTK_DATA"
python -m experiments.vllm_compat --out output/vllm-compat.json
study_config="${THINKINGCAP_STUDY_CONFIG:-/workspace/project/data/study/evaluation.json}"
python -m experiments.vllm_lora_smoke --config "$study_config" --out output/vllm-lora-smoke.json
python -m experiments.study_eval --config "$study_config" --out /workspace/project/output/study
