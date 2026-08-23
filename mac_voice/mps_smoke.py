"""Minimal real MPS forward/backward/optimizer probe."""

from __future__ import annotations

from .trainer_mps import resolve_device


def run_mps_smoke() -> dict[str, str | float]:
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for MPS smoke test") from exc
    device = resolve_device("mps")
    model = nn.Linear(4, 2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = torch.ones(2, 4, device=device)
    target = torch.zeros(2, 2, device=device)
    optimizer.zero_grad(set_to_none=True)
    loss = ((model(batch) - target) ** 2).mean()
    if not torch.isfinite(loss).item():
        raise FloatingPointError("MPS smoke loss is non-finite")
    loss.backward()
    optimizer.step()
    return {"device": str(device), "loss": float(loss.detach().cpu().item()), "status": "ok"}
