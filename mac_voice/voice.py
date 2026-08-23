"""Voice Package validation and inference prerequisites."""

from __future__ import annotations

import json
from pathlib import Path


REQUIRED_FILES = ("adapter.pt", "voice.json", "provenance.json", "reference.wav", "reference.txt")


def validate_voice_package(path: str | Path) -> tuple[dict, list[str]]:
    root = Path(path).expanduser()
    errors: list[str] = []
    for filename in REQUIRED_FILES:
        if not (root / filename).is_file():
            errors.append(f"missing package file: {filename}")
    voice: dict = {}
    voice_path = root / "voice.json"
    if voice_path.is_file():
        try:
            loaded = json.loads(voice_path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                errors.append("voice.json must contain an object")
            else:
                voice = loaded
                for key in ("name", "base_model", "adapter", "speaker_id", "language", "sample_rate"):
                    if key not in voice:
                        errors.append(f"voice.json missing key: {key}")
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"invalid voice.json: {exc}")
    return voice, errors


def require_adapter_inference_support(voice: dict, model_dir: str | Path) -> None:
    """Validate prerequisites for the connected adapter inference path."""
    model_path = Path(model_dir).expanduser()
    if not model_path.is_dir():
        raise FileNotFoundError(f"base model directory not found: {model_path}")
    if not voice.get("adapter"):
        raise ValueError("Voice Package metadata does not name an adapter")
