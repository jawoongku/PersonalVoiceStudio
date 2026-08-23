"""Feature-pipeline prerequisites and ONNX provider selection."""

from __future__ import annotations

import importlib
import json
import logging
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path


def _load_pcm_wav(path: str | Path, torch):
    """Load PCM WAV without torchaudio's optional torchcodec dependency."""
    import numpy as np
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_rate = handle.getframerate()
        width = handle.getsampwidth()
        frames = handle.readframes(handle.getnframes())
    if width == 2:
        values = np.frombuffer(frames, dtype="<i2").astype("float32") / 32768.0
    elif width == 1:
        values = (np.frombuffer(frames, dtype="u1").astype("float32") - 128.0) / 128.0
    else:
        raise ValueError(f"unsupported PCM sample width {width} in {path}")
    values = values.reshape(-1, channels).T
    return torch.from_numpy(values.copy()), sample_rate


@dataclass
class ProviderResult:
    requested: str
    selected: str | None
    available: list[str]
    detail: str


def validate_feature_artifacts(split_dir: str | Path) -> list[str]:
    """Validate extracted tensors for completeness and finite numeric values."""
    import math
    import torch
    split = Path(split_dir)
    errors: list[str] = []
    wav_ids = set(_read_kaldi_map(split / "wav.scp"))
    for filename, label in (("utt2embedding.pt", "embedding"), ("utt2speech_token.pt", "speech token")):
        path = split / filename
        if not path.is_file():
            errors.append(f"missing {filename}")
            continue
        values = torch.load(path)
        missing = wav_ids - set(values)
        if missing:
            errors.append(f"{filename} missing utterances: {sorted(missing)}")
        for utt, value in values.items():
            if not value:
                errors.append(f"{filename} empty for {utt}")
            if label == "embedding" and not all(math.isfinite(float(item)) for item in value):
                errors.append(f"{filename} non-finite for {utt}")
    spk_path = split / "spk2embedding.pt"
    if not spk_path.is_file():
        errors.append("missing spk2embedding.pt")
    else:
        for speaker, value in torch.load(spk_path).items():
            if not value or not all(math.isfinite(float(item)) for item in value):
                errors.append(f"spk2embedding.pt invalid for {speaker}")
    return errors


def _load_onnxruntime():
    try:
        return importlib.import_module("onnxruntime")
    except ImportError as exc:
        raise RuntimeError("onnxruntime is required for feature extraction") from exc


def select_provider(requested: str = "auto", model_path: str | Path | None = None) -> ProviderResult:
    """Select CoreML only after a real session can be constructed.

    A model path is required for CoreML validation. CPU selection is always
    safe when the CPUExecutionProvider is present. CUDA is intentionally
    rejected because this project targets Apple Silicon.
    """
    requested = requested.lower()
    if requested == "cuda":
        raise ValueError("CUDA provider is not allowed for the Mac pipeline")
    ort = _load_onnxruntime()
    available = list(ort.get_available_providers())
    cpu = "CPUExecutionProvider"
    coreml = "CoreMLExecutionProvider"
    if requested not in {"auto", "coreml", "cpu"}:
        raise ValueError(f"unsupported ONNX provider: {requested}")
    if requested == "cpu":
        if cpu not in available:
            raise RuntimeError("CPUExecutionProvider is unavailable")
        return ProviderResult(requested, cpu, available, "explicit CPU selection")
    if requested in {"auto", "coreml"} and coreml in available and model_path:
        model = Path(model_path)
        if model.is_file():
            try:
                session = ort.InferenceSession(str(model), providers=[coreml])
                session.get_inputs()
                return ProviderResult(requested, coreml, available, "CoreML session creation succeeded")
            except Exception as exc:
                if requested == "coreml":
                    raise RuntimeError(f"CoreML session validation failed: {exc}") from exc
                detail = f"CoreML validation failed ({exc.__class__.__name__}); fell back to CPU"
            else:  # pragma: no cover
                detail = "CoreML validation succeeded"
        elif requested == "coreml":
            raise RuntimeError(f"ONNX model not found: {model}")
        else:
            detail = "ONNX model unavailable for CoreML validation; fell back to CPU"
    else:
        if requested == "coreml":
            raise RuntimeError("CoreML provider or model path unavailable")
        detail = "CoreML not selected; fell back to CPU"
    if cpu not in available:
        raise RuntimeError("CPUExecutionProvider is unavailable")
    return ProviderResult(requested, cpu, available, detail)


def inspect_feature_inputs(dataset: str | Path, model_dir: str | Path) -> list[str]:
    """Return actionable prerequisite errors for the feature stage."""
    dataset_root = Path(dataset).expanduser()
    model_root = Path(model_dir).expanduser()
    errors: list[str] = []
    for split in ("train", "dev"):
        split_root = dataset_root / split
        for filename in ("wav.scp", "text", "utt2spk", "spk2utt", "instruct"):
            if not (split_root / filename).is_file():
                errors.append(f"missing {split}/{filename}")
    if not model_root.is_dir():
        errors.append(f"model directory not found: {model_root}")
    for filename in ("speech_tokenizer_v3.onnx", "campplus.onnx"):
        if not (model_root / filename).is_file():
            errors.append(f"missing model asset: {model_root / filename}")
    return errors


def _read_kaldi_map(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.strip().split(maxsplit=1)
            if len(fields) == 2:
                mapping[fields[0]] = fields[1]
    return mapping


def extract_embeddings(split_dir: str | Path, onnx_path: str | Path, threads: int = 1) -> tuple[Path, Path]:
    """Extract Campplus embeddings with the CPU ONNX provider."""
    torch = importlib.import_module("torch")
    torchaudio = importlib.import_module("torchaudio")
    kaldi = importlib.import_module("torchaudio.compliance.kaldi")
    ort = _load_onnxruntime()
    split = Path(split_dir)
    utt2wav = _read_kaldi_map(split / "wav.scp")
    utt2spk = _read_kaldi_map(split / "utt2spk")
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    utt2embedding: dict[str, list[float]] = {}
    spk2values: dict[str, list[list[float]]] = {}
    for utt, wav_path in utt2wav.items():
        audio, sample_rate = _load_pcm_wav(wav_path, torch)
        if sample_rate != 16000:
            audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(audio)
        feat = kaldi.fbank(audio, num_mel_bins=80, dither=0, sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        embedding = session.run(None, {input_name: feat.unsqueeze(dim=0).cpu().numpy()})[0].flatten().tolist()
        utt2embedding[utt] = embedding
        spk2values.setdefault(utt2spk[utt], []).append(embedding)
    spk2embedding = {speaker: torch.tensor(values).mean(dim=0).tolist() for speaker, values in spk2values.items()}
    utt_path = split / "utt2embedding.pt"
    spk_path = split / "spk2embedding.pt"
    torch.save(utt2embedding, utt_path)
    torch.save(spk2embedding, spk_path)
    return utt_path, spk_path


def extract_speech_tokens(split_dir: str | Path, onnx_path: str | Path, provider: str = "auto") -> Path:
    """Extract speech tokens with CoreML validation or CPU fallback."""
    torch = importlib.import_module("torch")
    torchaudio = importlib.import_module("torchaudio")
    whisper = importlib.import_module("whisper")
    ort = _load_onnxruntime()
    selected = select_provider(provider, onnx_path)
    session = ort.InferenceSession(str(onnx_path), providers=[selected.selected])
    input_names = [item.name for item in session.get_inputs()]
    split = Path(split_dir)
    utt2speech_token: dict[str, list[int]] = {}
    for utt, wav_path in _read_kaldi_map(split / "wav.scp").items():
        audio, sample_rate = _load_pcm_wav(wav_path, torch)
        if sample_rate != 16000:
            audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(audio)
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        if audio.shape[1] / 16000 > 30:
            raise ValueError(f"speech token extraction does not support audio longer than 30s: {utt}")
        feat = whisper.log_mel_spectrogram(audio, n_mels=128)
        token = session.run(None, {input_names[0]: feat.detach().cpu().numpy(), input_names[1]: __import__("numpy").array([feat.shape[2]], dtype=__import__("numpy").int32)})[0].flatten().tolist()
        utt2speech_token[utt] = token
    output = split / "utt2speech_token.pt"
    torch.save(utt2speech_token, output)
    return output


def build_parquet(split_dir: str | Path, upstream_root: str | Path, processes: int = 1) -> Path:
    """Use the pinned upstream parquet builder without modifying upstream."""
    split = Path(split_dir)
    script = Path(upstream_root) / "tools" / "make_parquet_list.py"
    if not script.is_file():
        raise FileNotFoundError(f"upstream parquet builder not found: {script}")
    destination = split / "parquet"
    destination.mkdir(parents=True, exist_ok=True)
    command = [
        "python", str(script), "--num_utts_per_parquet", "1000",
        "--num_processes", str(processes), "--src_dir", str(split), "--des_dir", str(destination),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "parquet builder failed")
    data_list = destination / "data.list"
    # The pinned upstream script uses multiprocessing and can swallow worker
    # exceptions. Build the single shard locally when it did not materialize.
    listed = [Path(line.strip()) for line in data_list.read_text(encoding="utf-8").splitlines()] if data_list.is_file() else []
    needs_features = False
    if listed and all(path.is_file() for path in listed):
        try:
            import pyarrow.parquet as pq
            parquet_file = pq.ParquetFile(listed[0])
            columns = set(parquet_file.schema_arrow.names)
            needs_features = not {"utt_embedding", "spk_embedding", "speech_token"}.issubset(columns)
        except Exception:
            needs_features = True
    if not listed or not all(path.is_file() for path in listed) or needs_features:
        import pandas as pd
        import torch
        utt2wav = _read_kaldi_map(split / "wav.scp")
        utt2text = _read_kaldi_map(split / "text")
        utt2spk = _read_kaldi_map(split / "utt2spk")
        utt2instruct = _read_kaldi_map(split / "instruct")
        utt2embedding = torch.load(split / "utt2embedding.pt")
        spk2embedding = torch.load(split / "spk2embedding.pt")
        utt2speech_token = torch.load(split / "utt2speech_token.pt")
        rows = []
        for utt, wav in utt2wav.items():
            rows.append({
                "utt": utt, "audio_data": Path(wav).read_bytes(), "wav": wav,
                "text": utt2text[utt], "spk": utt2spk[utt],
                "utt_embedding": utt2embedding[utt].tolist() if hasattr(utt2embedding[utt], "tolist") else utt2embedding[utt],
                "spk_embedding": spk2embedding[utt2spk[utt]].tolist() if hasattr(spk2embedding[utt2spk[utt]], "tolist") else spk2embedding[utt2spk[utt]],
                "speech_token": utt2speech_token[utt].tolist() if hasattr(utt2speech_token[utt], "tolist") else utt2speech_token[utt],
                "instruct": utt2instruct.get(utt, ""),
            })
        shard = destination / "parquet_000000000.tar"
        pd.DataFrame(rows).to_parquet(shard)
        data_list.write_text(str(shard) + "\n", encoding="utf-8")
    return data_list
