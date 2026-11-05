#!/usr/bin/env bash
# Runs inside the disposable GPU pod; no cloud API credentials are copied here.
set -euo pipefail
cd /workspace/ThinkingCap-Test
mkdir -p /workspace/logs /workspace/artifacts /workspace/cache
phase=setup
finish() {
  rc=$?
  printf '{"phase":"%s","exit_code":%s,"finished_at":"%s"}\n' "$phase" "$rc" "$(date -u +%FT%TZ)" > /workspace/artifacts/exit.json
}
trap finish EXIT
export HF_HOME=/workspace/cache/huggingface
export NLTK_DATA=/workspace/nltk_data
export UV_CACHE_DIR=/workspace/cache/uv
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0
date -u +%FT%TZ > /workspace/artifacts/started_at.txt
nvidia-smi > /workspace/artifacts/gpu.txt
/workspace/ops/uv venv --python 3.11 /workspace/venv
/workspace/ops/uv pip install --python /workspace/venv/bin/python -r /workspace/ops/gpu-requirements.lock
/workspace/ops/uv pip freeze --python /workspace/venv/bin/python > /workspace/artifacts/packages.txt
source /workspace/venv/bin/activate
python - <<'PY'
import torch
assert torch.cuda.is_available(), 'CUDA is not available'
assert torch.cuda.is_bf16_supported(), 'BF16 is not supported'
print('GPU:', torch.cuda.get_device_name(), 'torch:', torch.__version__)
PY
phase=smoke
printf '%s\n' "$phase" > /workspace/artifacts/phase
python evaluate.py --out runs/check --smoke 2>&1 | tee /workspace/logs/smoke.log
python - <<'PY'
import json
r=json.load(open('runs/check/results.json'))
assert not r['failed_benchmarks'], r['failed_benchmarks']
assert sum(v['n'] for v in r['results'].values()) == 240
print('Smoke passed: all 240 responses graded.')
PY
phase=baseline
printf '%s\n' "$phase" > /workspace/artifacts/phase
python evaluate.py --out runs/base 2>&1 | tee /workspace/logs/baseline.log
python - <<'PY'
import json
r=json.load(open('runs/base/results.json'))
assert not r['failed_benchmarks'], r['failed_benchmarks']
assert r['samples']==8 and r['cohort_total']==2759 and not r['smoke']
assert sum(v['n'] for v in r['results'].values())==22072
print('Baseline complete: all 22072 responses graded.')
PY
phase=complete
printf '%s\n' "$phase" > /workspace/artifacts/phase
sha256sum runs/base/results.json runs/base/records.jsonl > /workspace/artifacts/baseline.sha256
