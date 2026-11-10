#!/usr/bin/env bash
# Run the whole pipeline in order. Usage: scripts/run_all.sh [path/to/config.yaml]
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG="${1:-configs/default.yaml}"
for step in 01_download_audit 02_prepare_seeds 03_train_centralized 04_train_fedavg \
            05_finetune 06_lock_selection 07_evaluate_test 08_make_report; do
  echo "=== ${step} ==="
  uv run python "scripts/${step}.py" --config "${CONFIG}"
done
