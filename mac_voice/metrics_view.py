"""Read-only helpers for presenting JSONL training metrics."""

from __future__ import annotations

import json
from pathlib import Path


def summarize_metrics(path: str | Path, *, limit: int = 10) -> str:
    source = Path(path).expanduser()
    if not source.is_file():
        return f"metrics 파일을 찾을 수 없습니다: {source}"
    rows = []
    for line in source.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    if not rows:
        return "유효한 metrics 기록이 없습니다."
    lines = []
    for row in rows[-max(1, limit):]:
        lines.append(
            f"step={row.get('step', '?')} train={row.get('train_loss', '-') } "
            f"val={row.get('val_loss', '-')} lr={row.get('learning_rate', '-')}"
        )
    return "\n".join(lines)
