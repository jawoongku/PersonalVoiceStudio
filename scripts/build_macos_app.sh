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
echo "[OK] app bundle: $APP_DIR"
