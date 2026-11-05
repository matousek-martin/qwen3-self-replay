#!/usr/bin/env bash
set -euo pipefail
cd /workspace/project
export UV_FIND_LINKS=/workspace/project/data/wheels
export UV_OFFLINE=true
(cd data/wheels && sha256sum -c SHA256SUMS > /dev/null)
if python3 -c 'import json,sys; sys.exit(0 if json.load(open("experiments/shorterbetter/launch/protocol.json"))["execution"].get("resume_repair") else 1)'; then
    export THINKINGCAP_REUSE_ENV=1
fi
source experiments/setup_gpu.sh
mkdir -p lab/experiments
if [[ ! -e lab/experiments/shorterbetter-preparation-20261026 ]]; then
    ln -s /workspace/project/data/shorterbetter lab/experiments/shorterbetter-preparation-20261026
fi
python -m experiments.shorterbetter.start_scaled
