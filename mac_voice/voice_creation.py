"""End-to-end selected-dataset to Voice Package creation pipeline."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable

from .dataset import prepare_dataset
from .features import build_parquet, extract_embeddings, extract_speech_tokens, inspect_feature_inputs, validate_feature_artifacts
from .mps_training import run_user_parquet_mps_train
from .package import build_voice_package


def create_voice_package_from_dataset(
    dataset: str | Path,
    name: str,
    model_dir: str | Path,
    output_root: str | Path,
    voices_root: str | Path,
    upstream_root: str | Path = "/Users/jawoongku/CosyVoice",
    *,
    progress: Callable[[str], None] | None = None,
) -> dict[str, str | int | float]:
    """Prepare, extract features, train on MPS, and package one selected dataset."""
    root = Path(dataset).expanduser()
    model_path = Path(model_dir).expanduser()
    prepared = root.parent / f"{root.name}_prepared"
    run = Path(output_root).expanduser() / name
    package = Path(voices_root).expanduser() / name
    report = lambda message: progress(message) if progress else None
    report("학습 데이터 준비 중…")
    prepare_dataset(root, prepared)
    report("feature/parquet 생성 중…")
    errors = inspect_feature_inputs(prepared, model_path)
    if errors:
        raise ValueError("feature 입력 검증 실패: " + "; ".join(errors))
    embedding_model = model_path / "campplus.onnx"
    speech_model = model_path / "speech_tokenizer_v3.onnx"
    for split in ("train", "dev"):
        split_dir = prepared / split
        extract_embeddings(split_dir, embedding_model)
        extract_speech_tokens(split_dir, speech_model, "auto")
        feature_errors = validate_feature_artifacts(split_dir)
        if feature_errors:
            raise ValueError("feature 검증 실패: " + "; ".join(feature_errors))
        build_parquet(split_dir, upstream_root)
    report("CosyVoice LoRA MPS 학습 중…")
    run.mkdir(parents=True, exist_ok=True)
    manifest = prepared / "manifest.json"
    import json
    data = json.loads(manifest.read_text(encoding="utf-8"))
    first = data["splits"]["train"][0]
    shutil.copy2(first["path"], run / "reference.wav")
    (run / "reference.txt").write_text(str(first["text"]) + "\n", encoding="utf-8")
    result = run_user_parquet_mps_train(
        prepared / "train/parquet/data.list",
        prepared / "dev/parquet/data.list",
        model_path,
        run / "adapter.pt",
        upstream_root,
    )
    report("Voice Package 생성 중…")
    destination = build_voice_package(run, name, package, base_model=model_path, upstream_root=upstream_root)
    report("완료")
    return {"name": name, "dataset": str(root), "run": str(run), "voice_package": str(destination), **result}
