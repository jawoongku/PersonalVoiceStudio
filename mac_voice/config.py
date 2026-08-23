"""Configuration loading and validation for local training runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read training YAML config") from exc
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("training config must contain a mapping at the root")
    return data


def validate_training_config(config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    model_dir = config.get("model_dir") or config.get("model", {}).get("dir")
    if not model_dir:
        errors.append("model_dir is required")
    training = config.get("training", {})
    lora = config.get("lora", {})
    if not isinstance(training, dict):
        errors.append("training must be a mapping")
    if not isinstance(lora, dict):
        errors.append("lora must be a mapping")
    if isinstance(training, dict):
        for key in ("batch_size", "grad_accum_steps", "max_epochs", "grad_clip"):
            if key in training and float(training[key]) <= 0:
                errors.append(f"training.{key} must be positive")
    if isinstance(lora, dict):
        for key in ("rank", "alpha"):
            if key in lora and int(lora[key]) <= 0:
                errors.append(f"lora.{key} must be positive")
        if "dropout" in lora and not 0 <= float(lora["dropout"]) < 1:
            errors.append("lora.dropout must be in [0, 1)")
    return errors
