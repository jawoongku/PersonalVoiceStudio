"""Voice Package discovery for the local UI and CLI."""

from __future__ import annotations

from pathlib import Path

from .voice import validate_voice_package


def list_voice_packages(root: str | Path) -> list[dict]:
    base = Path(root).expanduser()
    if not base.is_dir():
        return []
    results: list[dict] = []
    for path in sorted(item for item in base.iterdir() if item.is_dir()):
        metadata, errors = validate_voice_package(path)
        results.append({"path": str(path), "name": metadata.get("name", path.name), "language": metadata.get("language"), "sample_rate": metadata.get("sample_rate"), "valid": not errors, "errors": errors})
    return results
