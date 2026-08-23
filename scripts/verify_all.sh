#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"
conda run -n cosyvoice python -m unittest discover -s tests -q
(cd "$ROOT_DIR/mac_app" && swift build -c release)
echo "[OK] Python tests and Swift release build passed"
