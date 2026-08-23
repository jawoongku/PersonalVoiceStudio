"""Minimal native LoRA injection helpers for torch.nn.Linear modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


def _torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("PyTorch is required for LoRA operations") from exc
    return torch, nn


@dataclass
class TrainableStats:
    total: int
    frozen: int
    trainable: int
    matched_modules: list[str]

    @property
    def ratio(self) -> float:
        return self.trainable / self.total if self.total else 0.0


def inject_lora(model, target_suffixes: Iterable[str], rank: int = 16, alpha: int = 32, dropout: float = 0.05):
    """Wrap matching Linear modules and return their fully-qualified names."""
    torch, nn = _torch()
    if rank <= 0 or alpha <= 0:
        raise ValueError("LoRA rank and alpha must be positive")
    if not 0 <= dropout < 1:
        raise ValueError("LoRA dropout must be in [0, 1)")
    targets = tuple(target_suffixes)
    matched: list[str] = []

    class LoRALinear(nn.Module):
        def __init__(self, base: nn.Linear):
            super().__init__()
            self.base = base
            self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=base.weight.dtype))
            self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=base.weight.dtype))
            self.scaling = alpha / rank
            self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
            nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)

        def forward(self, x):
            update = self.dropout(x) @ self.lora_a.t() @ self.lora_b.t()
            return self.base(x) + update * self.scaling

    def replace(parent, prefix: str = ""):
        for name, child in list(parent.named_children()):
            qualified = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and any(qualified.endswith(target) or name == target for target in targets):
                setattr(parent, name, LoRALinear(child))
                matched.append(qualified)
            else:
                replace(child, qualified)

    replace(model)
    if not matched:
        raise ValueError(f"no Linear modules matched LoRA targets: {list(targets)}")
    return model, matched


def freeze_base_parameters(model) -> TrainableStats:
    """Freeze all parameters except LoRA A/B parameters and report counts."""
    matched: list[str] = []
    for name, parameter in model.named_parameters():
        is_lora = ".lora_a" in name or ".lora_b" in name
        parameter.requires_grad = is_lora
        if is_lora:
            matched.append(name)
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return TrainableStats(total=total, frozen=total - trainable, trainable=trainable, matched_modules=matched)


def trainable_parameters(model):
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def validate_gradients(model) -> tuple[bool, bool, list[str]]:
    """Return (LoRA gradients valid, frozen gradients absent, problems)."""
    torch, _ = _torch()
    lora_ok = True
    nonzero_lora_gradient = False
    frozen_ok = True
    problems: list[str] = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                lora_ok = False
                problems.append(f"invalid LoRA gradient: {name}")
            elif torch.any(parameter.grad != 0):
                nonzero_lora_gradient = True
        elif parameter.grad is not None:
            frozen_ok = False
            problems.append(f"frozen parameter received gradient: {name}")
    if not nonzero_lora_gradient:
        lora_ok = False
        problems.append("no LoRA parameter received a non-zero gradient")
    return lora_ok, frozen_ok, problems
