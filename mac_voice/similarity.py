"""Speaker embedding similarity primitives; model extraction remains caller-owned."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("embeddings must be non-empty and have equal dimensions")
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = math.sqrt(sum(float(a) ** 2 for a in left))
    right_norm = math.sqrt(sum(float(b) ** 2 for b in right))
    if left_norm == 0 or right_norm == 0:
        raise ValueError("embeddings must not be zero vectors")
    return dot / (left_norm * right_norm)


def extract_campplus_embedding(audio_path: str | Path, model_path: str | Path) -> list[float]:
    """Extract one speaker embedding using the local CAMPPlus ONNX model."""
    import importlib
    import torch
    import torchaudio
    from .features import _load_pcm_wav

    ort = importlib.import_module("onnxruntime")
    audio, sample_rate = _load_pcm_wav(audio_path, torch)
    if sample_rate != 16000:
        audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(audio)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    kaldi = torchaudio.compliance.kaldi
    feat = kaldi.fbank(audio, num_mel_bins=80, dither=0, sample_frequency=16000)
    feat = feat - feat.mean(dim=0, keepdim=True)
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    return session.run(None, {input_name: feat.unsqueeze(0).cpu().numpy()})[0].flatten().tolist()


def evaluate_audio_similarity(reference: str | Path, generated: str | Path, model_path: str | Path) -> dict[str, object]:
    left = extract_campplus_embedding(reference, model_path)
    right = extract_campplus_embedding(generated, model_path)
    return {
        "reference": str(reference),
        "generated": str(generated),
        "model": str(model_path),
        "embedding_dimension": len(left),
        "speaker_similarity": cosine_similarity(left, right),
        "scorer": "CAMPPlus-ONNX cosine similarity",
    }
