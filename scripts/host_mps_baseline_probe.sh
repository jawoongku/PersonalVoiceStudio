#!/usr/bin/env bash
set -u

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT" || exit 2
export PYTHONPATH="$PROJECT_ROOT"
MODEL_DIR="${COSYVOICE_MODEL_DIR:-/Users/jawoongku/Models/Fun-CosyVoice3-0.5B}"
OUTPUT="${PVS_MPS_BASELINE_OUTPUT:-$PROJECT_ROOT/artifacts/mps_baseline_ko.wav}"

conda run -n cosyvoice python -m mac_voice mps-baseline \
  --model-dir "$MODEL_DIR" \
  --text "오늘도 좋은 하루 보내세요." \
  --output "$OUTPUT" \
  --reference "$PROJECT_ROOT/artifacts/voices/my_voice_cpu_train20_verified/reference.wav" \
  --reference-text "오늘 아침에는 평소보다 조금 일찍일어났습니다."
status=$?
if [[ "$status" -ne 0 ]]; then
  exit "$status"
fi
python3 - "$OUTPUT" <<'PY'
import sys, wave, struct, math
path = sys.argv[1]
with wave.open(path, "rb") as handle:
    frames = handle.getnframes()
    rate = handle.getframerate()
    channels = handle.getnchannels()
raw = handle.readframes(frames)
samples = struct.unpack("<%dh" % (len(raw) // 2), raw)
rms = math.sqrt(sum(value * value for value in samples) / max(1, len(samples))) / 32768.0
print(f"[HOST] output={path} frames={frames} sample_rate={rate} channels={channels} rms={rms:.6f}")
if frames <= 0 or rate != 24000 or channels != 1:
    raise SystemExit("invalid MPS baseline WAV")
if rms <= 0.001:
    raise SystemExit("MPS baseline WAV is silent")
PY
