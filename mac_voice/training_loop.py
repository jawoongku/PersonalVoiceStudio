"""Reusable LoRA training loop with validation and resumable artifacts."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Iterable

from .checkpoint import save_adapter_checkpoint
from .lora import validate_gradients
from .training_state import MetricsLogger, load_training_state, save_training_state
from .trainer_mps import TrainerConfig, build_optimizer, move_batch, train_one_step


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for the training loop") from exc
    return torch


def _validate_loss(model, batches: Iterable[Any], forward_fn: Callable[[Any], Any], device) -> float:
    torch = _torch()
    model.eval()
    values: list[float] = []
    with torch.no_grad():
        for batch in batches:
            loss = forward_fn(move_batch(batch, device))
            if not torch.is_tensor(loss) or loss.ndim != 0 or not torch.isfinite(loss).item():
                raise FloatingPointError("validation produced a non-finite or non-scalar loss")
            values.append(float(loss.detach().cpu().item()))
    if not values:
        raise ValueError("validation dataset is empty")
    return sum(values) / len(values)


def fit(
    model,
    train_batches: Iterable[Any],
    val_batches: Iterable[Any],
    forward_fn: Callable[[Any], Any],
    *,
    output_dir: str | Path,
    device,
    config: TrainerConfig | None = None,
    max_epochs: int = 1,
    max_steps: int | None = None,
    validate_every: int = 1,
    start_step: int = 0,
    start_epoch: int = 0,
    resume_from: str | Path | None = None,
    cancel_path: str | Path | None = None,
    progress: Callable[[int, dict[str, Any]], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    if max_epochs <= 0 or validate_every <= 0:
        raise ValueError("max_epochs and validate_every must be positive")
    torch = _torch()
    config = config or TrainerConfig(device=str(device))
    output = Path(output_dir).expanduser()
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    optimizer = build_optimizer(model, config)
    scheduler = None
    if resume_from is not None:
        metadata = load_training_state(resume_from, optimizer=optimizer, scheduler=scheduler, map_location="cpu")
        start_step = int(metadata.get("step", start_step))
        start_epoch = int(metadata.get("epoch", start_epoch))
    metrics = MetricsLogger(output / "metrics.jsonl")
    step = start_step
    best_val: float | None = None
    last_loss: float | None = None
    for epoch in range(start_epoch, start_epoch + max_epochs):
        for batch in train_batches:
            cancelled = cancel_path is not None and Path(cancel_path).expanduser().exists()
            if should_cancel is not None:
                cancelled = cancelled or should_cancel()
            if cancelled:
                return {"step": step, "epoch": epoch, "train_loss": last_loss, "best_val_loss": best_val, "status": "cancelled"}
            started = time.perf_counter()
            last_loss = train_one_step(model, batch, forward_fn, optimizer, config, device)
            step += 1
            val_loss = None
            if step % validate_every == 0:
                val_loss = _validate_loss(model, val_batches, forward_fn, device)
            elapsed = time.perf_counter() - started
            learning_rate = optimizer.param_groups[0]["lr"]
            metrics.log(step=step, train_loss=last_loss, val_loss=val_loss, learning_rate=learning_rate, step_time=elapsed)
            if progress is not None:
                progress(step, {"step": step, "train_loss": last_loss, "val_loss": val_loss, "learning_rate": learning_rate, "step_time": elapsed})
            save_adapter_checkpoint(model, checkpoints / "adapter_latest.pt", step=step, epoch=epoch, val_loss=val_loss, config={"trainer": config.__dict__})
            save_training_state(output / "training_state.pt", optimizer=optimizer, scheduler=scheduler, step=step, epoch=epoch, config={"trainer": config.__dict__})
            if val_loss is not None and (best_val is None or val_loss < best_val):
                best_val = val_loss
                save_adapter_checkpoint(model, checkpoints / "adapter_best.pt", step=step, epoch=epoch, val_loss=val_loss, config={"trainer": config.__dict__})
            if max_steps is not None and step >= max_steps:
                return {"step": step, "epoch": epoch, "train_loss": last_loss, "best_val_loss": best_val}
    return {"step": step, "epoch": start_epoch + max_epochs - 1, "train_loss": last_loss, "best_val_loss": best_val}
