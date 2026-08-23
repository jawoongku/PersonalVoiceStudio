"""Build a portable Voice Package without copying the base model."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _runtime_version(module_name: str) -> str | None:
    try:
        module = __import__(module_name)
        return str(getattr(module, "__version__", "unknown"))
    except Exception:
        return None


def _upstream_commit(upstream_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(upstream_root), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or None
    except OSError:
        return None


def build_voice_package(
    run_dir: str | Path,
    name: str,
    output: str | Path,
    *,
    base_model: str | Path,
    upstream_root: str | Path = "/Users/jawoongku/CosyVoice",
    speaker_id: str = "owner",
    language: str = "ko",
    sample_rate: int = 24000,
) -> Path:
    run = Path(run_dir).expanduser()
    destination = Path(output).expanduser()
    checkpoint = run / "checkpoints" / "adapter_best.pt"
    if not checkpoint.is_file():
        checkpoint = run / "checkpoints" / "adapter_latest.pt"
    if not checkpoint.is_file():
        checkpoint = run / "adapter.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"adapter checkpoint not found under {run}")
    reference_wav = run / "reference.wav"
    reference_txt = run / "reference.txt"
    if not reference_wav.is_file() or not reference_txt.is_file():
        raise FileNotFoundError("run must contain reference.wav and reference.txt")
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, destination / "adapter.pt")
    shutil.copy2(reference_wav, destination / "reference.wav")
    shutil.copy2(reference_txt, destination / "reference.txt")
    for optional_name in ("training_config.yaml", "metrics.json", "speaker_embedding.pt"):
        source = run / optional_name
        if source.is_file():
            shutil.copy2(source, destination / optional_name)
    metrics_source = run / "metrics.json"
    metrics = json.loads(metrics_source.read_text(encoding="utf-8")) if metrics_source.is_file() else {}
    voice = {
        "name": name,
        "base_model": Path(base_model).name,
        "adapter": "adapter.pt",
        "speaker_id": speaker_id,
        "language": language,
        "sample_rate": sample_rate,
    }
    provenance = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "upstream_commit": _upstream_commit(Path(upstream_root).expanduser()),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": _runtime_version("torch"),
        "torchaudio": _runtime_version("torchaudio"),
        "base_model_path": str(Path(base_model).expanduser()),
        "lora_config": run / "training_config.yaml".name if (run / "training_config.yaml").is_file() else None,
        "metrics": metrics,
    }
    (destination / "voice.json").write_text(json.dumps(voice, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (destination / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (destination / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return destination
