#!/usr/bin/env bash
set -u

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT" || exit 2
export PYTHONPATH="$PROJECT_ROOT"
MODEL_DIR="${COSYVOICE_MODEL_DIR:-/Users/jawoongku/Models/Fun-CosyVoice3-0.5B}"
VOICE="${PVS_MPS_VOICE:-$PROJECT_ROOT/artifacts/voices/my_voice_mps_epoch}"
OUTPUT="${PVS_MPS_SYNTH_OUTPUT:-$PROJECT_ROOT/artifacts/mps_adapter_tts.wav}"

conda run -n cosyvoice python -m mac_voice mps-synth \
  --voice "$VOICE" \
  --text "결국 제가 원하는 건 복잡하지 않고 오래 사용할 수 있는 방법입니다." \
  --output "$OUTPUT" \
  --model-dir "$MODEL_DIR"
