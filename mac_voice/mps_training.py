"""Real user parquet batch forward/backward validation on Apple MPS."""

from __future__ import annotations

from pathlib import Path

from .baseline import _load_upstream
from .lora import freeze_base_parameters, inject_lora, trainable_parameters, validate_gradients
from .trainer_mps import move_batch, resolve_device


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
