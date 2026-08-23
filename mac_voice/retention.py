"""Non-destructive training run retention planning."""

from __future__ import annotations

from pathlib import Path


def retention_plan(root: str | Path, keep: int = 3) -> dict[str, list[str]]:
    if keep <= 0:
        raise ValueError("keep must be positive")
    base = Path(root).expanduser()
    runs = sorted((item for item in base.iterdir() if item.is_dir()), key=lambda item: item.stat().st_mtime, reverse=True) if base.is_dir() else []
    return {"keep": [str(path) for path in runs[:keep]], "candidates": [str(path) for path in runs[keep:]]}
