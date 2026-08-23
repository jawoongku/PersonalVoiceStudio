"""Append-only TTS generation history."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def append_tts_history(path: str | Path, *, voice: str, text: str, output: str) -> Path:
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    record = {"created_at": datetime.now(timezone.utc).isoformat(), "voice": voice, "text": text, "output": output}
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return destination


def read_tts_history(path: str | Path, *, limit: int = 10) -> list[dict]:
    source = Path(path).expanduser()
    if not source.is_file():
        return []
    rows: list[dict] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows[-max(0, limit):]
