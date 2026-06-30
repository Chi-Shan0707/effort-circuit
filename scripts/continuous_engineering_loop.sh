#!/usr/bin/env bash
set -euo pipefail

COMMAND="${1:-python -m pytest -q}"
INTERVAL="${INTERVAL:-30}"
ITERATIONS="${ITERATIONS:-0}"

python -m src.loop_engineering --command "$COMMAND" --interval "$INTERVAL" --iterations "$ITERATIONS"
