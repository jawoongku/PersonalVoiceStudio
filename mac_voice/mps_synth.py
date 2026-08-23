"""MPS inference for an adapter-backed Voice Package."""

from __future__ import annotations

from pathlib import Path

from .adapter import inject_voice_adapter
from .baseline import _install_wave_loader, _load_upstream, _save_pcm16
from .trainer_mps import resolve_device
from .voice import validate_voice_package


def run_mps_synth(voice_dir: str | Path, text: str, output: str | Path, *, model_dir: str | Path, upstream_root: str | Path = "/Users/jawoongku/CosyVoice") -> Path:
    import torch
    import torchaudio

    device = resolve_device("mps")
    voice_path = Path(voice_dir).expanduser()
    voice, errors = validate_voice_package(voice_path)
    if errors:
        raise ValueError("invalid Voice Package: " + "; ".join(errors))
    model_path = Path(model_dir).expanduser()
    module = _load_upstream(upstream_root)
    _install_wave_loader(torchaudio, torch)
    model = module.AutoModel(model_dir=str(model_path), load_trt=False, fp16=False)
    checkpoint = torch.load(voice_path / "adapter.pt", map_location="cpu")
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    inject_voice_adapter(model, voice_path / "adapter.pt", rank=int(config.get("rank", 16)), alpha=int(config.get("alpha", 64)), dropout=float(config.get("dropout", 0.05)))
    core = model.model
    core.device = device
    core.llm_context = __import__("contextlib").nullcontext()
    core.llm.to(device).eval()
    core.flow.to(device).eval()
    core.hift.to("cpu").eval()
    hift_inference = core.hift.inference

    def cpu_hift_inference(speech_feat, finalize=True):
        return hift_inference(speech_feat.detach().to("cpu"), finalize=finalize)

    core.hift.inference = cpu_hift_inference
    reference_text = (voice_path / "reference.txt").read_text(encoding="utf-8").strip()
    if not reference_text.endswith("<|endofprompt|>"):
        reference_text += "<|endofprompt|>"
    with torch.no_grad():
        outputs = list(model.inference_zero_shot(text, reference_text, str(voice_path / "reference.wav"), stream=False, text_frontend=True))
    if not outputs:
        raise RuntimeError("CosyVoice returned no MPS adapter output")
    speech = torch.cat([item["tts_speech"] for item in outputs], dim=1).detach().cpu()
    if speech.numel() == 0 or not torch.isfinite(speech).all() or float(speech.abs().max()) == 0:
        raise RuntimeError("MPS adapter audio is empty, non-finite, or silent")
    destination = Path(output).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    _save_pcm16(destination, speech, model.sample_rate, torch)
    return destination
