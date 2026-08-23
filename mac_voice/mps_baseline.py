"""CosyVoice3 baseline inference on Apple Silicon MPS.

The pinned upstream loader only chooses CUDA or CPU.  This adapter keeps the
upstream checkout untouched, loads normally, then moves the three inference
modules and runtime device to MPS before synthesis.
"""

from __future__ import annotations

from pathlib import Path

from .baseline import _install_wave_loader, _load_upstream, _save_pcm16
from .trainer_mps import resolve_device


def run_mps_baseline(
    model_dir: str | Path,
    text: str,
    output: str | Path,
    *,
    upstream_root: str | Path = "/Users/jawoongku/CosyVoice",
    speaker_id: str | None = None,
    reference: str | Path | None = None,
    reference_text: str | None = None,
) -> Path:
    import torch
    import torchaudio

    device = resolve_device("mps")
    model_path = Path(model_dir).expanduser()
    if not model_path.is_dir():
        raise FileNotFoundError(f"model directory not found: {model_path}")
    module = _load_upstream(upstream_root)
    _install_wave_loader(torchaudio, torch)
    model = module.AutoModel(model_dir=str(model_path), load_trt=False, fp16=False)
    speakers = model.list_available_spks()
    core = model.model
    core.device = device
    core.llm_context = __import__("contextlib").nullcontext()
    for component in (core.llm, core.flow):
        component.to(device)
        component.eval()
    # HiFiGAN's causal f0 predictor explicitly requires float64, which MPS
    # cannot represent. Keep only the vocoder on CPU and move its mel input
    # across the boundary; LLM and flow remain on MPS.
    core.hift.to("cpu")
    core.hift.eval()
    hift_inference = core.hift.inference

    def cpu_hift_inference(speech_feat, finalize=True):
        return hift_inference(speech_feat.detach().to("cpu"), finalize=finalize)

    core.hift.inference = cpu_hift_inference
    with torch.no_grad():
        if speakers:
            selected = speaker_id or speakers[0]
            if selected not in speakers:
                raise ValueError(f"speaker not found: {selected}; available speakers: {speakers}")
            outputs = list(model.inference_sft(text, selected, stream=False, text_frontend=True))
        else:
            if not reference or not reference_text:
                raise RuntimeError("model has no SFT speaker profiles; provide --reference and --reference-text")
            prompt = reference_text.strip()
            if not prompt.endswith("<|endofprompt|>"):
                prompt += "<|endofprompt|>"
            outputs = list(model.inference_zero_shot(text, prompt, str(reference), stream=False, text_frontend=True))
    if not outputs:
        raise RuntimeError("CosyVoice returned no MPS audio output")
    speech = torch.cat([item["tts_speech"] for item in outputs], dim=1).detach().cpu()
    if speech.numel() == 0 or not torch.isfinite(speech).all() or float(speech.abs().max()) == 0:
        raise RuntimeError("MPS generated audio is empty, non-finite, or silent")
    destination = Path(output).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    _save_pcm16(destination, speech, model.sample_rate, torch)
    return destination
