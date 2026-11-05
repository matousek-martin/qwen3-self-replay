#!/usr/bin/env bash
set -euo pipefail
source /workspace/project/experiments/setup_gpu.sh
python - <<'PY'
import hashlib
from pathlib import Path
expected = {
    'data/val-pilot.jsonl': '20fca2946aac355bf478aaec0542c1fcfe208129d1243f07e63fd33d96db5ed6',
    'data/study/adapters/rl-replay/adapter_model.safetensors': '1428e5c50390574a226a5590a8766a46a0fe33d7b947d2c518406b1b3f951b05',
    'data/study/adapters/rl-replay/adapter_config.json': 'fafc1062b9b6e990ddeca3c700ea8ab8fcdc97d9af63e923670a215958214c25',
}
for name, digest in expected.items():
    if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
        raise RuntimeError('Throughput probe input mismatch: ' + name)
PY
python experiments/offline_training/generate.py \
    --prompts data/val-pilot.jsonl \
    --output output/validation/rl-replay.jsonl \
    --adapter data/study/adapters/rl-replay \
    --samples 2 --max-new-tokens 4096 --max-model-len 8192 \
    --seed 42 --backend vllm --batch-size 16 --max-seconds 0
