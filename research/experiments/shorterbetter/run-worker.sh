#!/usr/bin/env bash
set -euo pipefail
cd /workspace/project
# Public, pinned wheels are staged before rental; do not spend GPU time waiting
# for the provider's slow package downloads. Hashes are sealed with the payload.
export UV_FIND_LINKS=/workspace/project/data/wheels
export UV_OFFLINE=true
(cd data/wheels && sha256sum -c SHA256SUMS > /dev/null)
source experiments/setup_gpu.sh
mkdir -p lab/experiments
ln -s /workspace/project/data/shorterbetter lab/experiments/shorterbetter-preparation-20261026
python -m experiments.shorterbetter.start_worker
