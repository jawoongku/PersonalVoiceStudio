"""Discovery of local training run artifacts."""

from __future__ import annotations

from pathlib import Path
import json

from .jobs import read_job


def list_runs(root: str | Path) -> list[dict]:
    base = Path(root).expanduser()
    if not base.is_dir():
        return []
    rows = []
    for path in sorted(item for item in base.iterdir() if item.is_dir()):
        row = {"name": path.name, "path": str(path), "job_status": None, "checkpoint": None, "metrics": None, "last_metrics": None}
        job = path / "job.json"
        if job.is_file():
            try:
                row["job_status"] = read_job(job).get("status")
            except (OSError, ValueError):
                row["job_status"] = "invalid"
        for candidate in (path / "adapter.pt", path / "checkpoints" / "adapter_best.pt", path / "checkpoints" / "adapter_latest.pt"):
            if candidate.is_file():
                row["checkpoint"] = str(candidate)
                break
        for candidate in (path / "metrics.jsonl", path / "adapter.metrics.jsonl"):
            if candidate.is_file():
                row["metrics"] = str(candidate)
                for line in reversed(candidate.read_text(encoding="utf-8").splitlines()):
                    try:
                        parsed = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(parsed, dict):
                        row["last_metrics"] = {key: value for key, value in parsed.items() if key in {"train_loss", "val_loss", "learning_rate"} and isinstance(value, (int, float))}
                        break
                break
        rows.append(row)
    return rows
