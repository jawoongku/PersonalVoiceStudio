#!/usr/bin/env bash
set -u

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT" || exit 2
export PYTHONPATH="$PROJECT_ROOT"
MODEL_DIR="${COSYVOICE_MODEL_DIR:-/Users/jawoongku/Models/Fun-CosyVoice3-0.5B}"
TRAIN_LIST="${PVS_MPS_TRAIN_LIST:-$PROJECT_ROOT/data/my_voice_prepared/train/parquet/data.list}"
DEV_LIST="${PVS_MPS_DEV_LIST:-$PROJECT_ROOT/data/my_voice_prepared/dev/parquet/data.list}"
OUTPUT="${PVS_MPS_TRAIN_OUTPUT:-$PROJECT_ROOT/artifacts/runs/mps_user_epoch/adapter.pt}"

conda run -n cosyvoice python -m mac_voice mps-parquet-train \
  --train-data-list "$TRAIN_LIST" \
  --dev-data-list "$DEV_LIST" \
  --model-dir "$MODEL_DIR" \
  --output "$OUTPUT"
