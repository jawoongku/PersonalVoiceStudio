"""Small, filesystem-backed job state used by the future training UI."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_job(output_dir: str | Path, *, command: str, config: str) -> Path:
    root = Path(output_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    path = root / "job.json"
    if path.exists():
        raise FileExistsError(path)
    payload = {"format": 1, "status": "queued", "command": command, "config": config, "created_at": _now(), "updated_at": _now()}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def update_job(path: str | Path, status: str, **fields: Any) -> dict[str, Any]:
    job_path = Path(path).expanduser()
    if status not in {"queued", "running", "completed", "failed", "cancelled"}:
        raise ValueError(f"invalid job status: {status}")
    payload = read_job(job_path)
    current = payload["status"]
    allowed = {
        "queued": {"queued", "running", "cancelled", "failed"},
        "running": {"running", "completed", "failed", "cancelled"},
        "completed": {"completed"},
        "failed": {"failed"},
        "cancelled": {"cancelled"},
    }
    if status not in allowed[current]:
        raise ValueError(f"invalid job transition: {current} -> {status}")
    payload.update(fields)
    payload["status"] = status
    payload["updated_at"] = _now()
    job_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def read_job(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("format") != 1 or "status" not in payload:
        raise ValueError("invalid job metadata")
    return payload
