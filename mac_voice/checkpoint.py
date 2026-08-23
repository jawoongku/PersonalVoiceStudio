"""Adapter-only checkpoint save/load helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for checkpoints") from exc
    return torch


def adapter_state_dict(model) -> dict[str, Any]:
    return {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and ("lora_a" in name or "lora_b" in name)
    }


def save_adapter_checkpoint(model, path: str | Path, *, step: int, epoch: int, val_loss: float | None = None, config: dict | None = None) -> Path:
    torch = _torch()
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": 1,
        "adapter": adapter_state_dict(model),
        "step": step,
        "epoch": epoch,
        "val_loss": val_loss,
        "config": config or {},
    }
    if not payload["adapter"]:
        raise ValueError("no LoRA parameters found; refusing to save an empty adapter")
    torch.save(payload, destination)
    return destination


def load_adapter_checkpoint(model, path: str | Path, *, map_location: str = "cpu") -> dict[str, Any]:
    torch = _torch()
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = torch.load(source, map_location=map_location)
    if not isinstance(payload, dict) or "adapter" not in payload:
        raise ValueError("invalid adapter checkpoint format")
    current = adapter_state_dict(model)
    missing = sorted(set(current) - set(payload["adapter"]))
    unexpected = sorted(set(payload["adapter"]) - set(current))
    if missing or unexpected:
        raise ValueError(f"adapter keys mismatch; missing={missing}, unexpected={unexpected}")
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in payload["adapter"]:
                parameter.copy_(payload["adapter"][name].to(parameter.device, dtype=parameter.dtype))
    return {key: payload[key] for key in ("step", "epoch", "val_loss", "config") if key in payload}
