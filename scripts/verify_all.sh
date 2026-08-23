#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"
conda run -n cosyvoice python -m unittest discover -s tests -q
(cd "$ROOT_DIR/mac_app" && swift build -c release)
"$ROOT_DIR/scripts/build_macos_app.sh" >/dev/null
"$ROOT_DIR/scripts/check_app_bundle.sh"
echo "[OK] Python tests and Swift release build passed"
