"""Single-process trainer primitives for Apple Silicon MPS.

The model-specific forward function is injected by the CosyVoice adapter. This
keeps the training loop free of CUDA/DDP/torchrun assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for training") from exc
    return torch


@dataclass
class TrainerConfig:
    device: str = "mps"
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    grad_accum_steps: int = 1


def resolve_device(requested: str = "mps"):
    torch = _torch()
    if requested == "mps":
        if not torch.backends.mps.is_available():
            from .mps_runtime import probe

            report = probe()
            detail = report.get("error") or report.get("status") or "unavailable"
            action = report.get("action") or "run mps-doctor"
            raise RuntimeError(
                f"MPS is unavailable ({detail}); refusing to silently switch training to another device. "
                f"Action: {action}"
            )
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError("only mps or cpu are supported by the local trainer")


def move_batch(batch: Any, device):
    """Recursively move tensors while preserving ordinary metadata."""
    torch = _torch()
    if torch.is_tensor(batch):
        return batch.to(device)
    if isinstance(batch, dict):
        return {key: move_batch(value, device) for key, value in batch.items()}
    if isinstance(batch, list):
        return [move_batch(value, device) for value in batch]
    if isinstance(batch, tuple):
        return tuple(move_batch(value, device) for value in batch)
    return batch


def build_optimizer(model, config: TrainerConfig):
    torch = _torch()
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not params:
        raise ValueError("no trainable parameters found")
    return torch.optim.AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)


def train_one_step(
    model,
    batch: Any,
    forward_fn: Callable[[Any], Any],
    optimizer,
    config: TrainerConfig,
    device,
) -> float:
    """Run forward, finite-loss check, backward, clipping, and optimizer step."""
    torch = _torch()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    moved = move_batch(batch, device)
    loss = forward_fn(moved)
    if not torch.is_tensor(loss) or loss.ndim != 0:
        raise ValueError("forward_fn must return a scalar tensor loss")
    if not torch.isfinite(loss).item():
        raise FloatingPointError(f"non-finite loss: {loss.detach().item()}")
    loss.backward()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if config.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(trainable, config.grad_clip)
    optimizer.step()
    return float(loss.detach().cpu().item())


def parameter_summary(model) -> dict[str, int | float]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "total": total,
        "frozen": total - trainable,
        "trainable": trainable,
        "trainable_ratio": trainable / total if total else 0.0,
    }
