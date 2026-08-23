"""Integrity manifests for retained training/model artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_manifest(root: str | Path, output: str | Path | None = None) -> Path:
    base = Path(root).expanduser().resolve()
    if not base.is_dir():
        raise FileNotFoundError(base)
    destination = Path(output).expanduser().resolve() if output else base / "artifact-manifest.json"
    files = []
    for path in sorted(item for item in base.rglob("*") if item.is_file() and item.resolve() != destination):
        files.append({"path": str(path.relative_to(base)), "size": path.stat().st_size, "sha256": _digest(path)})
    payload = {"format": 1, "root": str(base), "files": files}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return destination


def verify_manifest(manifest: str | Path) -> list[str]:
    manifest_path = Path(manifest).expanduser().resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    base = Path(payload.get("root", manifest_path.parent)).expanduser()
    errors = []
    for entry in payload.get("files", []):
        path = base / entry["path"]
        if not path.is_file():
            errors.append(f"missing: {entry['path']}")
            continue
        if path.stat().st_size != entry["size"]:
            errors.append(f"size mismatch: {entry['path']}")
        elif _digest(path) != entry["sha256"]:
            errors.append(f"sha256 mismatch: {entry['path']}")
    return errors
