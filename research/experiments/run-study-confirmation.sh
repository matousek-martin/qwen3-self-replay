#!/usr/bin/env bash
set -euo pipefail
export THINKINGCAP_STUDY_CONFIG=/workspace/project/data/study/confirmation-evaluation.json
exec bash /workspace/project/experiments/run-study-eval.sh
