"""Resume state and JSONL metrics utilities."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for training state") from exc
    return torch


def save_training_state(path: str | Path, *, optimizer, scheduler=None, step: int, epoch: int, config: dict | None = None) -> Path:
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": 1,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "step": step,
        "epoch": epoch,
        "config": config or {},
    }
    _torch().save(payload, destination)
    return destination


def load_training_state(path: str | Path, *, optimizer, scheduler=None, map_location: str = "cpu") -> dict[str, Any]:
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = _torch().load(source, map_location=map_location)
    if not isinstance(payload, dict) or "optimizer" not in payload:
        raise ValueError("invalid training state format")
    optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    return {key: payload[key] for key in ("step", "epoch", "config") if key in payload}


class MetricsLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, *, step: int, train_loss: float | None = None, val_loss: float | None = None, learning_rate: float | None = None, step_time: float | None = None, samples_per_sec: float | None = None, mps_memory: int | None = None, rss_memory: int | None = None, driver_memory: int | None = None) -> None:
        record: dict[str, Any] = {"timestamp": datetime.now(timezone.utc).isoformat(), "step": step}
        for key, value in (("train_loss", train_loss), ("val_loss", val_loss), ("learning_rate", learning_rate), ("step_time", step_time), ("samples_per_sec", samples_per_sec), ("mps_memory", mps_memory), ("rss_memory", rss_memory), ("driver_memory", driver_memory)):
            if value is not None:
                record[key] = value
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
