#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL_PATH="${MODEL_PATH:-../TinyLoRA-GRPO-Coder/models/Qwen3-0.6B}"
THREADS="${NUM_THREADS:-4}"

export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export NUMEXPR_NUM_THREADS="$THREADS"

printf '[closure] running unit tests\n'
python -m pytest -q

printf '[closure] checking CLI help\n'
python -m src.site_audit --help >/dev/null
python -m src.stop_after_final_experiment --help >/dev/null
python -m src.paired_stats --help >/dev/null

if [[ -d "$MODEL_PATH" ]]; then
  printf '[closure] model path found; running small smoke experiments with %s\n' "$MODEL_PATH"
  python -m src.site_audit \
    --model "$MODEL_PATH" \
    --dataset synthetic_math \
    --n 1 \
    --layers 18,27 \
    --out outputs/closure_site_audit_smoke.json \
    --device cpu \
    --dtype float32 \
    --num-threads "$THREADS"

  python -m src.stop_after_final_experiment \
    --model "$MODEL_PATH" \
    --dataset heldout_synthetic_math \
    --n 2 \
    --conditions 27:0.75 \
    --max-new-tokens 32 \
    --out outputs/closure_stop_after_final_smoke.json \
    --device cpu \
    --dtype float32 \
    --num-threads "$THREADS"

  python -m src.paired_stats \
    --input outputs/closure_stop_after_final_smoke.json \
    --out outputs/closure_stop_after_final_stats_smoke.json \
    --bootstrap 1000 \
    --seed 123
else
  printf '[closure] model path not found; skipped model smoke experiments: %s\n' "$MODEL_PATH"
fi

printf '[closure] all closure checks completed\n'
