"""CosyVoice3 baseline inference adapter."""

from __future__ import annotations

import importlib
import os
import sys
import wave
from pathlib import Path


def _load_upstream(upstream_root: str | Path):
    # The desktop sandbox may make the user's cache directories read-only.
    # Keep upstream imports deterministic without modifying upstream files.
    os.environ.setdefault("NUMBA_DISABLE_CACHING", "1")
    cache_root = Path("/tmp/cosyvoice-cache")
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_root / "numba"))
    Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    ffmpeg_lib = Path("/opt/homebrew/opt/ffmpeg/lib")
    if ffmpeg_lib.is_dir():
        existing_dyld = os.environ.get("DYLD_LIBRARY_PATH", "")
        os.environ["DYLD_LIBRARY_PATH"] = f"{ffmpeg_lib}:{existing_dyld}" if existing_dyld else str(ffmpeg_lib)
    root = str(Path(upstream_root).expanduser())
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        module = importlib.import_module("cosyvoice.cli.cosyvoice")
    except ImportError as exc:
        raise RuntimeError(f"cannot import CosyVoice from {root}: {exc}") from exc
    return module


def _install_wave_loader(torchaudio, torch) -> None:
    """Avoid torchaudio 2.11's optional torchcodec dependency for PCM WAV."""
    import numpy as np

    def load(path, *args, **kwargs):
        with wave.open(str(path), "rb") as handle:
            channels = handle.getnchannels()
            sample_rate = handle.getframerate()
            width = handle.getsampwidth()
            frames = handle.readframes(handle.getnframes())
        if width == 1:
            values = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
            values = (values - 128.0) / 128.0
        elif width == 2:
            values = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        elif width == 4:
            values = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            raise RuntimeError(f"unsupported PCM WAV sample width: {width}")
        tensor = torch.from_numpy(values.copy())
        if channels > 1:
            tensor = tensor.reshape(-1, channels).transpose(0, 1)
        else:
            tensor = tensor.unsqueeze(0)
        return tensor, sample_rate

    torchaudio.load = load


def _save_pcm16(path: str | Path, speech, sample_rate: int, torch) -> None:
    import numpy as np

    array = speech.detach().cpu().float().clamp(-1, 1).numpy()
    if array.ndim == 2:
        array = array[0] if array.shape[0] == 1 else array.T.reshape(-1)
    pcm = (array * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def run_baseline(
    model_dir: str | Path,
    text: str,
    output: str | Path,
    *,
    upstream_root: str | Path = "/Users/jawoongku/CosyVoice",
    speaker_id: str | None = None,
) -> Path:
    model_path = Path(model_dir).expanduser()
    if not model_path.is_dir():
        raise FileNotFoundError(f"model directory not found: {model_path}")
    module = _load_upstream(upstream_root)
    try:
        import torch
        import torchaudio
    except ImportError as exc:
        raise RuntimeError("PyTorch and torchaudio are required for baseline inference") from exc
    _install_wave_loader(torchaudio, torch)
    model = module.AutoModel(model_dir=str(model_path), load_trt=False, fp16=False)
    speakers = model.list_available_spks()
    if not speakers:
        raise RuntimeError("model has no SFT speaker profiles; use clone with a reference WAV")
    selected = speaker_id or speakers[0]
    if selected not in speakers:
        raise ValueError(f"speaker not found: {selected}; available speakers: {speakers}")
    outputs = list(model.inference_sft(text, selected, stream=False, text_frontend=True))
    if not outputs:
        raise RuntimeError("CosyVoice returned no audio output")
    speech = torch.cat([item["tts_speech"] for item in outputs], dim=1).detach().cpu()
    if speech.numel() == 0 or not torch.isfinite(speech).all() or float(speech.abs().max()) == 0:
        raise RuntimeError("generated audio is empty, non-finite, or silent")
    destination = Path(output).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    _save_pcm16(destination, speech, model.sample_rate, torch)
    return destination


def run_zero_shot(
    model_dir: str | Path,
    reference: str | Path,
    reference_text: str,
    text: str,
    output: str | Path,
    *,
    upstream_root: str | Path = "/Users/jawoongku/CosyVoice",
) -> Path:
    model_path = Path(model_dir).expanduser()
    reference_path = Path(reference).expanduser()
    if not model_path.is_dir():
        raise FileNotFoundError(f"model directory not found: {model_path}")
    if not reference_path.is_file():
        raise FileNotFoundError(f"reference WAV not found: {reference_path}")
    if not reference_text.strip():
        raise ValueError("reference text must not be empty")
    module = _load_upstream(upstream_root)
    try:
        import torch
        import torchaudio
    except ImportError as exc:
        raise RuntimeError("PyTorch and torchaudio are required for zero-shot inference") from exc
    _install_wave_loader(torchaudio, torch)
    model = module.AutoModel(model_dir=str(model_path), load_trt=False, fp16=False)
    prompt = reference_text.strip()
    if not prompt.endswith("<|endofprompt|>"):
        prompt += "<|endofprompt|>"
    outputs = list(model.inference_zero_shot(text, prompt, str(reference_path), stream=False, text_frontend=True))
    if not outputs:
        raise RuntimeError("CosyVoice returned no audio output")
    speech = torch.cat([item["tts_speech"] for item in outputs], dim=1).detach().cpu()
    if speech.numel() == 0 or not torch.isfinite(speech).all() or float(speech.abs().max()) == 0:
        raise RuntimeError("generated audio is empty, non-finite, or silent")
    destination = Path(output).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    _save_pcm16(destination, speech, model.sample_rate, torch)
    return destination
