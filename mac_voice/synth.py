"""Adapter-backed Voice Package synthesis."""

from __future__ import annotations

from pathlib import Path

from .adapter import inject_voice_adapter
from .baseline import _install_wave_loader, _load_upstream, _save_pcm16
from .voice import validate_voice_package


def run_synth(
    voice_dir: str | Path,
    text: str,
    output: str | Path,
    *,
    model_dir: str | Path,
    upstream_root: str | Path = "/Users/jawoongku/CosyVoice",
) -> Path:
    voice_path = Path(voice_dir).expanduser()
    voice, errors = validate_voice_package(voice_path)
    if errors:
        raise ValueError("invalid Voice Package: " + "; ".join(errors))
    model_path = Path(model_dir).expanduser()
    if not model_path.is_dir():
        raise FileNotFoundError(f"base model directory not found: {model_path}")
    try:
        import importlib
        torch = importlib.import_module("torch")
        torchaudio = importlib.import_module("torchaudio")
    except ImportError as exc:
        raise RuntimeError("PyTorch and torchaudio are required for synth") from exc
    module = _load_upstream(upstream_root)
    # The upstream zero-shot path uses torchaudio.load for the reference WAV;
    # install the standard-library PCM loader to avoid torchcodec dylib issues.
    _install_wave_loader(torchaudio, torch)
    model = module.AutoModel(model_dir=str(model_path), load_trt=False, fp16=False)
    adapter_path = voice_path / "adapter.pt"
    checkpoint = torch.load(adapter_path, map_location="cpu")
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    inject_voice_adapter(
        model,
        adapter_path,
        rank=int(config.get("rank", 16)),
        alpha=int(config.get("alpha", 64)),
        dropout=float(config.get("dropout", 0.05)),
    )
    reference_text = (voice_path / "reference.txt").read_text(encoding="utf-8").strip()
    if not reference_text.endswith("<|endofprompt|>"):
        reference_text += "<|endofprompt|>"
    outputs = list(model.inference_zero_shot(text, reference_text, str(voice_path / "reference.wav"), stream=False, text_frontend=True))
    if not outputs:
        raise RuntimeError("CosyVoice returned no audio output")
    speech = torch.cat([item["tts_speech"] for item in outputs], dim=1).detach().cpu()
    if speech.numel() == 0 or not torch.isfinite(speech).all() or float(speech.abs().max()) == 0:
        raise RuntimeError("generated audio is empty, non-finite, or silent")
    destination = Path(output).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    _save_pcm16(destination, speech, model.sample_rate, torch)
    return destination
