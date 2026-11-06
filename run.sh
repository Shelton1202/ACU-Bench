#!/usr/bin/env bash
# Run the six locally hosted models with and without text, then score them.
#
#   ./run.sh [OUTPUT_DIR] [REPORT_DIR]
#
# Requirements: Linux, Python 3.11, and exactly two NVIDIA Tesla T4 GPUs.
# Completed environments and predictions are reused on rerun.
set -euo pipefail

cd "$(dirname "$0")"
OUTPUT="${1:-outputs}"
REPORT="${2:-results}"
PYTHON="${PYTHON:-python3.11}"

if [ "$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -c T4)" != 2 ]; then
  echo "The six local models require exactly two NVIDIA Tesla T4 GPUs." >&2
  exit 1
fi

if [ ! -x .venv/bin/python ]; then
  "$PYTHON" -m venv .venv
  .venv/bin/python -m pip install -e .
fi
.venv/bin/python -m forecast_bench check

for profile in gpu timebraid; do
  if [ ! -x ".venv-$profile/bin/python" ]; then
    .venv/bin/python scripts/setup_environment.py "$profile"
  fi
done
.venv/bin/python scripts/run_local_models.py --output "$OUTPUT"

# Validate all 3,158 forecasts per model and write the result tables.
.venv/bin/python -m forecast_bench evaluate --output "$OUTPUT" --report "$REPORT"
echo "Results: $REPORT/benchmark_results.md and $REPORT/benchmark_results.tex"
