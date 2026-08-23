#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
APP_DIR="${1:-$ROOT_DIR/artifacts/PersonalVoiceStudio.app}"
BIN="$APP_DIR/Contents/MacOS/PersonalVoiceStudio"
PLIST="$APP_DIR/Contents/Info.plist"

test -x "$BIN" || { echo "[ERROR] executable missing: $BIN" >&2; exit 1; }
test -f "$PLIST" || { echo "[ERROR] Info.plist missing: $PLIST" >&2; exit 1; }
plutil -extract CFBundleIdentifier raw "$PLIST" | grep -qx 'com.jawoongku.PersonalVoiceStudio'
echo "[OK] app bundle structure: $APP_DIR"
