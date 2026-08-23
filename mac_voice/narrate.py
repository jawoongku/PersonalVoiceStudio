"""Long-text chunking utility for adapter-backed synthesis."""

from __future__ import annotations

import re
from pathlib import Path

from .baseline import _install_wave_loader, _save_pcm16
from .synth import run_synth


def split_text(text: str, max_chars: int = 180) -> list[str]:
    sentences = [part.strip() for part in re.split(r"(?<=[.!?。！？])\s*", text) if part.strip()]
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if current and len(current) + 1 + len(sentence) > max_chars:
            chunks.append(current)
            current = ""
        current = sentence if not current else f"{current} {sentence}"
    if current:
        chunks.append(current)
    return chunks or ([text.strip()] if text.strip() else [])


def run_narrate(voice_dir: str | Path, input_path: str | Path, output: str | Path, *, model_dir: str | Path, upstream_root: str | Path = "/Users/jawoongku/CosyVoice", max_chars: int = 180) -> Path:
    import torch
    import torchaudio
    _install_wave_loader(torchaudio, torch)

    text = Path(input_path).expanduser().read_text(encoding="utf-8")
    chunks = split_text(text, max_chars=max_chars)
    if not chunks:
        raise ValueError("input script is empty")
    output_path = Path(output).expanduser()
    chunk_dir = output_path.parent / f".{output_path.stem}_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    audio_parts = []
    sample_rate = None
    for index, chunk in enumerate(chunks):
        chunk_path = chunk_dir / f"{index:04d}.wav"
        run_synth(voice_dir, chunk, chunk_path, model_dir=model_dir, upstream_root=upstream_root)
        audio, current_rate = torchaudio.load(str(chunk_path))
        sample_rate = sample_rate or current_rate
        if current_rate != sample_rate:
            raise ValueError("chunk sample rates differ")
        audio_parts.append(audio)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _save_pcm16(output_path, torch.cat(audio_parts, dim=1), sample_rate, torch)
    return output_path
