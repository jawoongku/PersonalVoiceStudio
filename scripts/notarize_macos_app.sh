#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
APP_PATH="${1:-$ROOT_DIR/artifacts/PersonalVoiceStudio.app}"
PROFILE="${NOTARY_PROFILE:-}"

if [[ -z "$PROFILE" ]]; then
  echo "[ERROR] set NOTARY_PROFILE to an xcrun notarytool keychain profile" >&2
  exit 1
fi
if [[ ! -d "$APP_PATH" ]]; then
  echo "[ERROR] app bundle not found: $APP_PATH" >&2
  exit 1
fi

ARCHIVE="$ROOT_DIR/artifacts/PersonalVoiceStudio.zip"
ditto -c -k --keepParent "$APP_PATH" "$ARCHIVE"
xcrun notarytool submit "$ARCHIVE" --keychain-profile "$PROFILE" --wait
xcrun stapler staple "$APP_PATH"
echo "[OK] notarized and stapled: $APP_PATH"
