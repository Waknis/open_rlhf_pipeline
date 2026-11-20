#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
exec python -m pytest -q tests "$@"
