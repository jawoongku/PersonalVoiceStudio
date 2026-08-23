"""CPU preflight for the pinned CosyVoice3 LLM training forward contract."""

from __future__ import annotations

from pathlib import Path

from .adapter import inject_voice_adapter
from .baseline import _load_upstream
from .lora import freeze_base_parameters, inject_lora


def _synthetic_batch():
    import torch
    return {
        "text_token": torch.tensor([[10, 11, 12, 13]], dtype=torch.int64),
        "text_token_len": torch.tensor([4], dtype=torch.int32),
        "speech_token": torch.tensor([[2, 3, 4, 5, 6, 7]], dtype=torch.int64),
        "speech_token_len": torch.tensor([6], dtype=torch.int32),
        "instruct_token": torch.tensor([[10, 11]], dtype=torch.int64),
        "instruct_token_len": torch.tensor([2], dtype=torch.int32),
    }


def run_model_forward_smoke(
    model_dir: str | Path,
    upstream_root: str | Path,
    *,
    rank: int = 2,
    alpha: int = 4,
) -> dict[str, float | int | str]:
    """Run one finite CPU forward/loss pass with a synthetic valid token batch.

    This deliberately does not claim training success: no backward or optimizer
    step is performed, and the tokens are synthetic rather than user data.
    """
    import torch

    module = _load_upstream(upstream_root)
    model = module.AutoModel(model_dir=str(Path(model_dir).expanduser()), load_trt=False, fp16=False)
    llm, matched = inject_lora(model.model.llm, ("q_proj", "k_proj", "v_proj", "o_proj"), rank=rank, alpha=alpha, dropout=0.0)
    stats = freeze_base_parameters(llm)
    text_vocab = model.model.llm.llm.model.model.embed_tokens.num_embeddings
    batch = _synthetic_batch()
    if text_vocab <= 13:
        raise RuntimeError(f"unexpected text vocabulary size: {text_vocab}")
    result = llm.forward(batch, torch.device("cpu"))
    loss = result.get("loss")
    if loss is None or not torch.isfinite(loss).item():
        raise FloatingPointError("CosyVoice3 CPU forward produced a non-finite loss")
    return {"status": "ok", "loss": float(loss.detach()), "matched_modules": len(matched), "trainable": stats.trainable}


def run_model_backward_smoke(model_dir: str | Path, upstream_root: str | Path, *, rank: int = 2, alpha: int = 4) -> dict[str, float | int | str]:
    """Run one real CPU LoRA backward and optimizer step on a synthetic batch."""
    import torch
    from .lora import validate_gradients, trainable_parameters

    module = _load_upstream(upstream_root)
    model = module.AutoModel(model_dir=str(Path(model_dir).expanduser()), load_trt=False, fp16=False)
    llm, matched = inject_lora(model.model.llm, ("q_proj", "k_proj", "v_proj", "o_proj"), rank=rank, alpha=alpha, dropout=0.0)
    stats = freeze_base_parameters(llm)
    optimizer = torch.optim.AdamW(trainable_parameters(llm), lr=1e-4)
    result = llm.forward(_synthetic_batch(), torch.device("cpu"))
    loss = result.get("loss")
    if loss is None or not torch.isfinite(loss).item():
        raise FloatingPointError("CosyVoice3 CPU backward preflight received a non-finite loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    lora_ok, frozen_ok, problems = validate_gradients(llm)
    if not lora_ok or not frozen_ok:
        raise RuntimeError("gradient validation failed: " + "; ".join(problems))
    optimizer.step()
    return {"status": "ok", "loss": float(loss.detach()), "matched_modules": len(matched), "trainable": stats.trainable}


def run_parquet_backward_smoke(data_list: str | Path, model_dir: str | Path, upstream_root: str | Path, *, rank: int = 2, alpha: int = 4) -> dict[str, float | int | str]:
    """Run CPU LoRA backward/step using one real feature-bearing parquet row."""
    import torch
    import pandas as pd
    from .lora import validate_gradients, trainable_parameters

    listed = [line.strip() for line in Path(data_list).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not listed:
        raise ValueError("data.list is empty")
    row = pd.read_parquet(listed[0]).iloc[0]
    module = _load_upstream(upstream_root)
    model = module.AutoModel(model_dir=str(Path(model_dir).expanduser()), load_trt=False, fp16=False)
    llm, matched = inject_lora(model.model.llm, ("q_proj", "k_proj", "v_proj", "o_proj"), rank=rank, alpha=alpha, dropout=0.0)
    stats = freeze_base_parameters(llm)
    tokenizer = model.frontend.tokenizer
    allowed = model.frontend.allowed_special
    text_tokens = tokenizer.encode(str(row["text"]), allowed_special=allowed)
    instruct_tokens = tokenizer.encode(str(row.get("instruct", "")), allowed_special=allowed)
    speech_tokens = row["speech_token"]
    if hasattr(speech_tokens, "tolist"):
        speech_tokens = speech_tokens.tolist()
    batch = {
        "text_token": torch.tensor([text_tokens], dtype=torch.int64),
        "text_token_len": torch.tensor([len(text_tokens)], dtype=torch.int32),
        "speech_token": torch.tensor([speech_tokens], dtype=torch.int64),
        "speech_token_len": torch.tensor([len(speech_tokens)], dtype=torch.int32),
        "instruct_token": torch.tensor([instruct_tokens], dtype=torch.int64),
        "instruct_token_len": torch.tensor([len(instruct_tokens)], dtype=torch.int32),
    }
    optimizer = torch.optim.AdamW(trainable_parameters(llm), lr=1e-4)
    result = llm.forward(batch, torch.device("cpu"))
    loss = result.get("loss")
    if loss is None or not torch.isfinite(loss).item():
        raise FloatingPointError("parquet batch produced a non-finite loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    lora_ok, frozen_ok, problems = validate_gradients(llm)
    if not lora_ok or not frozen_ok:
        raise RuntimeError("parquet gradient validation failed: " + "; ".join(problems))
    optimizer.step()
    return {"status": "ok", "loss": float(loss.detach()), "matched_modules": len(matched), "trainable": stats.trainable, "utt": str(row["utt"])}


def run_parquet_train_smoke(train_data_list: str | Path, dev_data_list: str | Path, model_dir: str | Path, upstream_root: str | Path, output: str | Path, *, steps: int = 2, rank: int = 2, alpha: int = 4) -> dict[str, float | int | str]:
    """Run a tiny CPU train/validation loop over real parquet rows."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    import torch
    import pandas as pd
    from .checkpoint import save_adapter_checkpoint
    from .lora import validate_gradients, trainable_parameters
    from .training_state import MetricsLogger, save_training_state

    def read_row(path):
        listed = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
        if not listed:
            raise ValueError(f"empty data.list: {path}")
        return pd.read_parquet(listed[0]).iloc[0]

    module = _load_upstream(upstream_root)
    model = module.AutoModel(model_dir=str(Path(model_dir).expanduser()), load_trt=False, fp16=False)
    llm, matched = inject_lora(model.model.llm, ("q_proj", "k_proj", "v_proj", "o_proj"), rank=rank, alpha=alpha, dropout=0.0)
    stats = freeze_base_parameters(llm)
    tokenizer = model.frontend.tokenizer
    allowed = model.frontend.allowed_special

    def make_batch(row):
        text = tokenizer.encode(str(row["text"]), allowed_special=allowed)
        instruct = tokenizer.encode(str(row.get("instruct", "")), allowed_special=allowed)
        speech = row["speech_token"]
        speech = speech.tolist() if hasattr(speech, "tolist") else speech
        return {
            "text_token": torch.tensor([text], dtype=torch.int64), "text_token_len": torch.tensor([len(text)], dtype=torch.int32),
            "speech_token": torch.tensor([speech], dtype=torch.int64), "speech_token_len": torch.tensor([len(speech)], dtype=torch.int32),
            "instruct_token": torch.tensor([instruct], dtype=torch.int64), "instruct_token_len": torch.tensor([len(instruct)], dtype=torch.int32),
        }

    train_batch = make_batch(read_row(train_data_list))
    dev_batch = make_batch(read_row(dev_data_list))
    optimizer = torch.optim.AdamW(trainable_parameters(llm), lr=1e-4)
    output_path = Path(output).expanduser()
    metrics = MetricsLogger(output_path.with_suffix(".metrics.jsonl"))
    train_loss = None
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        result = llm.forward(train_batch, torch.device("cpu"))
        train_loss = result["loss"]
        if not torch.isfinite(train_loss).item():
            raise FloatingPointError("train loss is non-finite")
        train_loss.backward()
        lora_ok, frozen_ok, problems = validate_gradients(llm)
        if not lora_ok or not frozen_ok:
            raise RuntimeError("train gradient validation failed: " + "; ".join(problems))
        optimizer.step()
        metrics.log(step=step, train_loss=float(train_loss.detach()), learning_rate=1e-4)
    llm.eval()
    with torch.no_grad():
        dev_loss = llm.forward(dev_batch, torch.device("cpu"))["loss"]
    if not torch.isfinite(dev_loss).item():
        raise FloatingPointError("dev loss is non-finite")
    metrics.log(step=steps, val_loss=float(dev_loss.detach()), learning_rate=1e-4)
    checkpoint = save_adapter_checkpoint(llm, output_path, step=steps, epoch=0, val_loss=float(dev_loss), config={"rank": rank, "alpha": alpha, "steps": steps, "device": "cpu"})
    state_path = save_training_state(output_path.with_suffix(".state.pt"), optimizer=optimizer, step=steps, epoch=0, config={"rank": rank, "alpha": alpha, "device": "cpu"})
    return {"status": "ok", "train_loss": float(train_loss.detach()), "dev_loss": float(dev_loss.detach()), "steps": steps, "checkpoint": str(checkpoint), "state": str(state_path), "metrics": str(output_path.with_suffix(".metrics.jsonl")), "matched_modules": len(matched), "trainable": stats.trainable}


def run_parquet_resume_smoke(data_list: str | Path, adapter: str | Path, state: str | Path, model_dir: str | Path, upstream_root: str | Path, output: str | Path, *, rank: int = 2, alpha: int = 4) -> dict[str, float | int | str]:
    """Reload adapter and optimizer state into a fresh model and take one CPU step."""
    import torch
    import pandas as pd
    from .checkpoint import load_adapter_checkpoint, save_adapter_checkpoint
    from .training_state import load_training_state
    from .lora import trainable_parameters, validate_gradients
    listed = [line.strip() for line in Path(data_list).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not listed:
        raise ValueError("data.list is empty")
    row = pd.read_parquet(listed[0]).iloc[0]
    module = _load_upstream(upstream_root)
    model = module.AutoModel(model_dir=str(Path(model_dir).expanduser()), load_trt=False, fp16=False)
    llm, matched = inject_lora(model.model.llm, ("q_proj", "k_proj", "v_proj", "o_proj"), rank=rank, alpha=alpha, dropout=0.0)
    freeze_base_parameters(llm)
    metadata = load_adapter_checkpoint(llm, adapter)
    tokenizer = model.frontend.tokenizer
    text = tokenizer.encode(str(row["text"]), allowed_special=model.frontend.allowed_special)
    instruct = tokenizer.encode(str(row.get("instruct", "")), allowed_special=model.frontend.allowed_special)
    speech = row["speech_token"]; speech = speech.tolist() if hasattr(speech, "tolist") else speech
    batch = {"text_token": torch.tensor([text], dtype=torch.int64), "text_token_len": torch.tensor([len(text)], dtype=torch.int32), "speech_token": torch.tensor([speech], dtype=torch.int64), "speech_token_len": torch.tensor([len(speech)], dtype=torch.int32), "instruct_token": torch.tensor([instruct], dtype=torch.int64), "instruct_token_len": torch.tensor([len(instruct)], dtype=torch.int32)}
    optimizer = torch.optim.AdamW(trainable_parameters(llm), lr=1e-4)
    resume = load_training_state(state, optimizer=optimizer)
    optimizer.zero_grad(set_to_none=True)
    loss = llm.forward(batch, torch.device("cpu"))["loss"]
    if not torch.isfinite(loss).item():
        raise FloatingPointError("resumed loss is non-finite")
    loss.backward()
    lora_ok, frozen_ok, problems = validate_gradients(llm)
    if not lora_ok or not frozen_ok:
        raise RuntimeError("resumed gradient validation failed: " + "; ".join(problems))
    optimizer.step()
    checkpoint = save_adapter_checkpoint(llm, output, step=int(resume.get("step", 0)) + 1, epoch=int(resume.get("epoch", 0)), val_loss=float(loss.detach()), config={"rank": rank, "alpha": alpha, "resumed_from": str(adapter)})
    return {"status": "ok", "loss": float(loss.detach()), "checkpoint": str(checkpoint), "resume_step": int(resume.get("step", 0)), "matched_modules": len(matched), "adapter_step": metadata.get("step", 0)}
