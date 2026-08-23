#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
APP_PATH="${APP_PATH:-$ROOT_DIR/artifacts/PersonalVoiceStudio.app}"
ARCHIVE_PATH="${ARCHIVE_PATH:-$ROOT_DIR/artifacts/PersonalVoiceStudio.zip}"

"$ROOT_DIR/scripts/build_macos_app.sh"
if [[ -n "${SIGNING_IDENTITY:-}" ]]; then
  echo "[OK] app signed during build"
else
  echo "[INFO] packaging unsigned development app"
fi

rm -f "$ARCHIVE_PATH" "$ARCHIVE_PATH.sha256"
ditto -c -k --keepParent "$APP_PATH" "$ARCHIVE_PATH"
shasum -a 256 "$ARCHIVE_PATH" > "$ARCHIVE_PATH.sha256"
echo "[OK] archive: $ARCHIVE_PATH"
echo "[OK] checksum: $ARCHIVE_PATH.sha256"
