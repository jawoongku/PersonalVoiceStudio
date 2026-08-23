#!/usr/bin/env bash
set -euo pipefail

# Create an isolated candidate environment. This script never changes or
# removes the existing `cosyvoice` environment.
ENV_NAME="${PVS_MPS_ENV_NAME:-pvs-mps}"
PYTHON_VERSION="${PVS_MPS_PYTHON_VERSION:-3.10}"
TORCH_VERSION="${PVS_MPS_TORCH_VERSION:-2.5.1}"
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: scripts/create_mps_env.sh [--dry-run] [--env NAME] [--python VERSION] [--torch VERSION]

Creates an isolated candidate environment and verifies it with `mps-doctor`.
The existing `cosyvoice` environment is never modified or removed.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --env) ENV_NAME="$2"; shift 2 ;;
    --python) PYTHON_VERSION="$2"; shift 2 ;;
    --torch) TORCH_VERSION="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "$(uname -m)" != "arm64" ]]; then
  echo "[ERROR] Apple Silicon arm64 is required" >&2
  exit 1
fi
if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda is required" >&2
  exit 1
fi
if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  echo "[ERROR] environment already exists: $ENV_NAME (refusing to modify it)" >&2
  exit 1
fi

echo "[INFO] candidate environment: $ENV_NAME"
echo "[INFO] Python: $PYTHON_VERSION"
echo "[INFO] torch/torchaudio: $TORCH_VERSION"
echo "[INFO] This is an isolated environment; existing environments are unchanged."

if [[ "$DRY_RUN" == 1 ]]; then
  echo "conda create -n $ENV_NAME python=$PYTHON_VERSION -y"
  echo "conda run -n $ENV_NAME python -m pip install torch==$TORCH_VERSION torchaudio==$TORCH_VERSION"
  echo "conda run -n $ENV_NAME python -m mac_voice mps-doctor"
  exit 0
fi

conda create -n "$ENV_NAME" "python=$PYTHON_VERSION" -y
conda run -n "$ENV_NAME" python -m pip install "torch==$TORCH_VERSION" "torchaudio==$TORCH_VERSION"

# Run the check from the working tree without installing the application.
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHONPATH="$PROJECT_ROOT" conda run -n "$ENV_NAME" python -m mac_voice mps-doctor
