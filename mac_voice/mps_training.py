"""Real user parquet batch forward/backward validation on Apple MPS."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Callable

from .baseline import _load_upstream
from .checkpoint import load_adapter_checkpoint, save_adapter_checkpoint
from .lora import freeze_base_parameters, inject_lora, trainable_parameters, validate_gradients
from .trainer_mps import move_batch, resolve_device
from .training_state import load_training_state, save_training_state


def _read_batch(data_list: str | Path, tokenizer, allowed_special, torch):
    import pandas as pd

    listed = [line.strip() for line in Path(data_list).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not listed:
        raise ValueError(f"empty data.list: {data_list}")
    row = pd.read_parquet(listed[0]).iloc[0]
    text = tokenizer.encode(str(row["text"]), allowed_special=allowed_special)
    instruct = tokenizer.encode(str(row.get("instruct", "")), allowed_special=allowed_special)
    speech = row["speech_token"]
    speech = speech.tolist() if hasattr(speech, "tolist") else speech
    return {
        "utt": str(row["utt"]),
        "text_token": torch.tensor([text], dtype=torch.int64),
        "text_token_len": torch.tensor([len(text)], dtype=torch.int32),
        "speech_token": torch.tensor([speech], dtype=torch.int64),
        "speech_token_len": torch.tensor([len(speech)], dtype=torch.int32),
        "instruct_token": torch.tensor([instruct], dtype=torch.int64),
        "instruct_token_len": torch.tensor([len(instruct)], dtype=torch.int32),
    }


def _read_rows(data_list: str | Path, tokenizer, allowed_special, torch):
    import pandas as pd

    rows = []
    for listed in [line.strip() for line in Path(data_list).read_text(encoding="utf-8").splitlines() if line.strip()]:
        frame = pd.read_parquet(listed)
        for index in range(len(frame)):
            row = frame.iloc[index]
            text = tokenizer.encode(str(row["text"]), allowed_special=allowed_special)
            instruct = tokenizer.encode(str(row.get("instruct", "")), allowed_special=allowed_special)
            speech = row["speech_token"]
            speech = speech.tolist() if hasattr(speech, "tolist") else speech
            rows.append({
                "utt": str(row["utt"]),
                "text_token": torch.tensor([text], dtype=torch.int64),
                "text_token_len": torch.tensor([len(text)], dtype=torch.int32),
                "speech_token": torch.tensor([speech], dtype=torch.int64),
                "speech_token_len": torch.tensor([len(speech)], dtype=torch.int32),
                "instruct_token": torch.tensor([instruct], dtype=torch.int64),
                "instruct_token_len": torch.tensor([len(instruct)], dtype=torch.int32),
            })
    if not rows:
        raise ValueError(f"no rows found in data.list: {data_list}")
    return rows


def run_user_parquet_mps_backward(
    data_list: str | Path,
    model_dir: str | Path,
    upstream_root: str | Path = "/Users/jawoongku/CosyVoice",
    *,
    rank: int = 2,
    alpha: int = 4,
) -> dict[str, object]:
    import torch

    device = resolve_device("mps")
    module = _load_upstream(upstream_root)
    model = module.AutoModel(model_dir=str(Path(model_dir).expanduser()), load_trt=False, fp16=False)
    llm, matched = inject_lora(model.model.llm, ("q_proj", "k_proj", "v_proj", "o_proj"), rank=rank, alpha=alpha, dropout=0.0)
    stats = freeze_base_parameters(llm)
    llm.to(device).train()
    tokenizer = model.frontend.tokenizer
    batch = _read_batch(data_list, tokenizer, model.frontend.allowed_special, torch)
    utt = batch.pop("utt")
    batch = move_batch(batch, device)
    optimizer = torch.optim.AdamW(trainable_parameters(llm), lr=1e-4)
    optimizer.zero_grad(set_to_none=True)
    result = llm.forward(batch, device)
    loss = result.get("loss")
    if loss is None or not torch.isfinite(loss).item():
        raise FloatingPointError("MPS user parquet forward produced a non-finite loss")
    loss.backward()
    lora_ok, frozen_ok, problems = validate_gradients(llm)
    if not lora_ok or not frozen_ok:
        raise RuntimeError("MPS user parquet gradient validation failed: " + "; ".join(problems))
    optimizer.step()
    return {
        "status": "ok",
        "device": str(device),
        "utt": utt,
        "loss": float(loss.detach().cpu().item()),
        "matched_modules": len(matched),
        "trainable": stats.trainable,
        "frozen": stats.frozen,
        "optimizer_step": True,
    }


def run_user_parquet_mps_train(
    train_data_list: str | Path,
    dev_data_list: str | Path,
    model_dir: str | Path,
    output: str | Path,
    upstream_root: str | Path = "/Users/jawoongku/CosyVoice",
    *,
    rank: int = 2,
    alpha: int = 4,
    progress: Callable[[int, dict[str, object]], None] | None = None,
) -> dict[str, object]:
    import torch
    from .checkpoint import save_adapter_checkpoint
    from .training_state import MetricsLogger, save_training_state

    device = resolve_device("mps")
    module = _load_upstream(upstream_root)
    model = module.AutoModel(model_dir=str(Path(model_dir).expanduser()), load_trt=False, fp16=False)
    llm, matched = inject_lora(model.model.llm, ("q_proj", "k_proj", "v_proj", "o_proj"), rank=rank, alpha=alpha, dropout=0.0)
    stats = freeze_base_parameters(llm)
    llm.to(device).train()
    tokenizer = model.frontend.tokenizer
    allowed = model.frontend.allowed_special
    train_rows = _read_rows(train_data_list, tokenizer, allowed, torch)
    dev_rows = _read_rows(dev_data_list, tokenizer, allowed, torch)
    optimizer = torch.optim.AdamW(trainable_parameters(llm), lr=1e-4)
    output_path = Path(output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = MetricsLogger(output_path.with_suffix(".metrics.jsonl"))
    last_loss = None
    for step, row in enumerate(train_rows, start=1):
        utt = row.pop("utt")
        batch = move_batch(row, device)
        optimizer.zero_grad(set_to_none=True)
        loss = llm.forward(batch, device).get("loss")
        if loss is None or not torch.isfinite(loss).item():
            raise FloatingPointError(f"MPS train loss is non-finite at step {step} ({utt})")
        loss.backward()
        lora_ok, frozen_ok, problems = validate_gradients(llm)
        if not lora_ok or not frozen_ok:
            raise RuntimeError(f"MPS gradient validation failed at step {step}: " + "; ".join(problems))
        optimizer.step()
        last_loss = float(loss.detach().cpu().item())
        metrics.log(step=step, train_loss=last_loss, learning_rate=1e-4)
        if progress is not None:
            progress(step, {"step": step, "train_loss": last_loss, "learning_rate": 1e-4})
    llm.eval()
    dev_losses = []
    with torch.no_grad():
        for row in dev_rows:
            row.pop("utt")
            dev_loss = llm.forward(move_batch(row, device), device).get("loss")
            if dev_loss is None or not torch.isfinite(dev_loss).item():
                raise FloatingPointError("MPS dev loss is non-finite")
            dev_losses.append(float(dev_loss.detach().cpu().item()))
    dev_loss_value = sum(dev_losses) / len(dev_losses)
    metrics.log(step=len(train_rows), val_loss=dev_loss_value, learning_rate=1e-4)
    if progress is not None:
        progress(len(train_rows), {"step": len(train_rows), "val_loss": dev_loss_value, "learning_rate": 1e-4})
    checkpoint = save_adapter_checkpoint(llm, output_path, step=len(train_rows), epoch=0, val_loss=dev_loss_value, config={"device": "mps", "rank": rank, "alpha": alpha, "train_rows": len(train_rows)})
    state_path = save_training_state(output_path.with_suffix(".state.pt"), optimizer=optimizer, scheduler=None, step=len(train_rows), epoch=0, config={"device": "mps", "rank": rank, "alpha": alpha})
    return {"status": "ok", "device": str(device), "train_rows": len(train_rows), "dev_rows": len(dev_rows), "steps": len(train_rows), "train_loss": last_loss, "dev_loss": dev_loss_value, "checkpoint": str(checkpoint), "state": str(state_path), "metrics": str(output_path.with_suffix(".metrics.jsonl")), "matched_modules": len(matched), "trainable": stats.trainable}


def run_user_parquet_mps_resume(
    data_list: str | Path,
    model_dir: str | Path,
    adapter: str | Path,
    state: str | Path,
    output: str | Path,
    upstream_root: str | Path = "/Users/jawoongku/CosyVoice",
) -> dict[str, object]:
    import torch

    device = resolve_device("mps")
    module = _load_upstream(upstream_root)
    model = module.AutoModel(model_dir=str(Path(model_dir).expanduser()), load_trt=False, fp16=False)
    llm, matched = inject_lora(model.model.llm, ("q_proj", "k_proj", "v_proj", "o_proj"), rank=2, alpha=4, dropout=0.0)
    stats = freeze_base_parameters(llm)
    adapter_meta = load_adapter_checkpoint(llm, adapter, map_location="cpu")
    llm.to(device).train()
    optimizer = torch.optim.AdamW(trainable_parameters(llm), lr=1e-4)
    state_meta = load_training_state(state, optimizer=optimizer, scheduler=None, map_location=device)
    start_step = int(state_meta.get("step", adapter_meta.get("step", 0)))
    tokenizer = model.frontend.tokenizer
    row = _read_batch(data_list, tokenizer, model.frontend.allowed_special, torch)
    utt = row.pop("utt")
    optimizer.zero_grad(set_to_none=True)
    loss = llm.forward(move_batch(row, device), device).get("loss")
    if loss is None or not torch.isfinite(loss).item():
        raise FloatingPointError("MPS resume loss is non-finite")
    loss.backward()
    lora_ok, frozen_ok, problems = validate_gradients(llm)
    if not lora_ok or not frozen_ok:
        raise RuntimeError("MPS resume gradient validation failed: " + "; ".join(problems))
    optimizer.step()
    output_path = Path(output).expanduser()
    checkpoint = save_adapter_checkpoint(llm, output_path, step=start_step + 1, epoch=int(state_meta.get("epoch", 0)), val_loss=None, config={"device": "mps", "rank": 2, "alpha": 4, "dropout": 0.0, "resumed_from": str(adapter)})
    state_path = save_training_state(output_path.with_suffix(".state.pt"), optimizer=optimizer, scheduler=None, step=start_step + 1, epoch=int(state_meta.get("epoch", 0)), config={"device": "mps", "resumed_from": str(state)})
    return {"status": "ok", "device": str(device), "utt": utt, "start_step": start_step, "step": start_step + 1, "loss": float(loss.detach().cpu().item()), "checkpoint": str(checkpoint), "state": str(state_path), "matched_modules": len(matched), "trainable": stats.trainable}
