"""Connect a saved LoRA adapter to a loaded CosyVoice3 model."""

from __future__ import annotations

from pathlib import Path

from .checkpoint import load_adapter_checkpoint
from .lora import freeze_base_parameters, inject_lora


def inspect_lora_targets(cosyvoice_model, target_modules=None) -> dict[str, list[str]]:
    """Report runtime Linear matches before mutating the upstream LLM."""
    import torch
    targets = tuple(target_modules or ("q_proj", "k_proj", "v_proj", "o_proj"))
    llm = getattr(getattr(cosyvoice_model, "model", None), "llm", None)
    if llm is None:
        raise RuntimeError("loaded CosyVoice model does not expose model.llm")
    names = [name for name, module in llm.named_modules() if isinstance(module, torch.nn.Linear)]
    matches = {target: [name for name in names if name.endswith(target)] for target in targets}
    if not any(matches.values()):
        raise ValueError(f"no Linear modules matched LoRA targets: {list(targets)}")
    return matches


def inject_voice_adapter(cosyvoice_model, adapter_path: str | Path, *, target_modules=None, rank: int = 16, alpha: int = 64, dropout: float = 0.05):
    """Inject and load LoRA into the upstream model's LLM module.

    The upstream CosyVoice wrapper exposes the trainable LLM as
    ``cosyvoice_model.model.llm``. The target names are still discovered at
    runtime by ``inject_lora``; no module is silently accepted when none match.
    """
    target_modules = tuple(target_modules or ("q_proj", "k_proj", "v_proj", "o_proj"))
    llm = getattr(getattr(cosyvoice_model, "model", None), "llm", None)
    if llm is None:
        raise RuntimeError("loaded CosyVoice model does not expose model.llm")
    llm, matched = inject_lora(llm, target_modules, rank=rank, alpha=alpha, dropout=dropout)
    stats = freeze_base_parameters(llm)
    if stats.trainable <= 0 or stats.trainable >= stats.total:
        raise RuntimeError(f"invalid LoRA trainable parameter count: {stats.trainable}/{stats.total}")
    load_metadata = load_adapter_checkpoint(llm, adapter_path, map_location="cpu")
    llm.eval()
    return {"matched_modules": matched, "parameter_stats": stats, "checkpoint": load_metadata}
