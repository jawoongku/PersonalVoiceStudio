#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
APP_DIR="$ROOT_DIR/artifacts/PersonalVoiceStudio.app"

cd "$ROOT_DIR/mac_app"
swift build -c release
rm -rf "$APP_DIR"
mkdir -p "$APP_DIR/Contents/MacOS" "$APP_DIR/Contents/Resources"
cp .build/arm64-apple-macosx/release/PersonalVoiceStudio "$APP_DIR/Contents/MacOS/PersonalVoiceStudio"
cp Info.plist "$APP_DIR/Contents/Info.plist"
if [[ -n "${SIGNING_IDENTITY:-}" ]]; then
  codesign --force --deep --options runtime --sign "$SIGNING_IDENTITY" "$APP_DIR"
  codesign --verify --deep --strict "$APP_DIR"
  echo "[OK] signed with: $SIGNING_IDENTITY"
else
  echo "[INFO] unsigned development bundle (set SIGNING_IDENTITY to sign)"
fi
echo "[OK] app bundle: $APP_DIR"
