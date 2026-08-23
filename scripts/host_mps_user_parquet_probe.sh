#!/usr/bin/env bash
set -u

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT" || exit 2
export PYTHONPATH="$PROJECT_ROOT"
MODEL_DIR="${COSYVOICE_MODEL_DIR:-/Users/jawoongku/Models/Fun-CosyVoice3-0.5B}"
DATA_LIST="${PVS_MPS_DATA_LIST:-$PROJECT_ROOT/data/my_voice_prepared/train/parquet/data.list}"

conda run -n cosyvoice python -m mac_voice mps-parquet-backward \
  --data-list "$DATA_LIST" \
  --model-dir "$MODEL_DIR"
