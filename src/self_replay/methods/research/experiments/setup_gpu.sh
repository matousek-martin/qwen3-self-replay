#!/usr/bin/env bash
set -euo pipefail
cd /workspace/project
export HF_HOME=/root/.cache/thinkingcap/huggingface
export UV_CACHE_DIR=/root/.cache/thinkingcap/uv
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p output "$HF_HOME" "$UV_CACHE_DIR"
finish() {
    rc=$?
    trap - EXIT
    # Hash a stable result first; publish exit.json only after all checksums exist.
    # The controller treats exit.json as the completion marker.
    printf '{"exit_code":%s,"finished_at":"%s"}\n' "$rc" "$(date -u +%FT%TZ)" > output/result.json
    (cd output && find . -type f ! -name 'SHA256SUMS*' ! -name 'exit.json*' ! -name runner.log ! -name supervisor.log ! -name 'worker-state.json*' ! -name '.worker-state.*.tmp' ! -name worker.pid -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS.tmp)
    mv output/SHA256SUMS.tmp output/SHA256SUMS
    cp output/result.json output/exit.json.tmp
    mv output/exit.json.tmp output/exit.json
    exit "$rc"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
nvidia-smi > output/gpu.txt
python3 -m experiments.gpu_policy
# Install large dependencies on local container disk. /workspace may be a
# network-backed FUSE mount with very slow small-file writes. Outputs stay there.
if [[ "${THINKINGCAP_REUSE_ENV:-0}" != "1" || ! -x /opt/thinkingcap-venv/bin/python ]]; then
    ./uv venv --python 3.11 /opt/thinkingcap-venv
fi
./uv pip install --python /opt/thinkingcap-venv/bin/python -r requirements.lock
./uv pip freeze --python /opt/thinkingcap-venv/bin/python > output/packages.txt
source /opt/thinkingcap-venv/bin/activate
python -c 'import torch; assert torch.cuda.is_available(); assert torch.cuda.is_bf16_supported(); print(torch.cuda.get_device_name())'
