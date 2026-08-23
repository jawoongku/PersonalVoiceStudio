#!/usr/bin/env bash
set -u

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT" || exit 2
export PYTHONPATH="$PROJECT_ROOT"

echo "[HOST] $(date)"
sw_vers
uname -m
echo "[HOST] mps-doctor"
conda run -n pvs-mps python -m mac_voice mps-doctor
doctor_status=$?
echo "[HOST] mps-smoke"
conda run -n pvs-mps python -m mac_voice mps-smoke
smoke_status=$?
echo "[HOST] doctor_exit=$doctor_status smoke_exit=$smoke_status"
if [[ "$doctor_status" -eq 0 && "$smoke_status" -eq 0 ]]; then
  exit 0
fi
exit 1
