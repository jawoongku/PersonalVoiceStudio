"""Before/after zero-shot versus adapter comparison orchestration."""

from __future__ import annotations

import json
from pathlib import Path

from .baseline import _install_wave_loader, run_zero_shot
from .synth import run_synth
from .voice import validate_voice_package


def _audio_summary(path: Path) -> dict:
    import torch
    import torchaudio
    _install_wave_loader(torchaudio, torch)

    audio, sample_rate = torchaudio.load(str(path))
    return {
        "path": str(path),
        "sample_rate": sample_rate,
        "channels": audio.shape[0],
        "samples": audio.shape[1],
        "duration": audio.shape[1] / sample_rate,
        "rms": float(torch.sqrt(torch.mean(audio.float() ** 2))),
        "peak": float(audio.abs().max()),
    }


def run_comparison(voice_dir: str | Path, text: str, output_dir: str | Path, *, model_dir: str | Path, upstream_root: str | Path = "/Users/jawoongku/CosyVoice") -> Path:
    voice_path = Path(voice_dir).expanduser()
    voice, errors = validate_voice_package(voice_path)
    if errors:
        raise ValueError("invalid Voice Package: " + "; ".join(errors))
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    before = destination / "before_zero_shot.wav"
    after = destination / "after_lora.wav"
    reference_text = (voice_path / "reference.txt").read_text(encoding="utf-8").strip()
    run_zero_shot(model_dir, voice_path / "reference.wav", reference_text, text, before, upstream_root=upstream_root)
    run_synth(voice_path, text, after, model_dir=model_dir, upstream_root=upstream_root)
    comparison = {"text": text, "before": _audio_summary(before), "after": _audio_summary(after)}
    report = destination / "comparison.json"
    report.write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
