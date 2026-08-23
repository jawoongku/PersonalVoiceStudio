"""Validation helpers for CosyVoice parquet data lists."""

from __future__ import annotations

import importlib
from pathlib import Path


REQUIRED_COLUMNS = {"utt", "audio_data", "wav", "text", "spk"}


def validate_data_list(path: str | Path, *, require_features: bool = False) -> list[str]:
    source = Path(path).expanduser()
    errors: list[str] = []
    if not source.is_file():
        return [f"data.list not found: {source}"]
    parquet_paths = [Path(line.strip()).expanduser() for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not parquet_paths:
        return [f"data.list is empty: {source}"]
    for parquet_path in parquet_paths:
        if not parquet_path.is_file():
            errors.append(f"parquet file not found: {parquet_path}")
            continue
        try:
            pq = importlib.import_module("pyarrow.parquet")
        except ImportError:
            errors.append("pyarrow is required to inspect parquet schema")
            break
        try:
            parquet_file = pq.ParquetFile(parquet_path)
            # schema.names exposes nested leaf names (e.g. ``element`` for
            # list columns); schema_arrow.names preserves the data columns.
            columns = set(parquet_file.schema_arrow.names)
        except Exception as exc:
            errors.append(f"cannot read parquet {parquet_path}: {exc}")
            continue
        missing = REQUIRED_COLUMNS - columns
        if missing:
            errors.append(f"{parquet_path} missing columns: {sorted(missing)}")
        if require_features:
            for column in ("utt_embedding", "spk_embedding", "speech_token"):
                if column not in columns:
                    errors.append(f"{parquet_path} missing feature column: {column}")
    return errors
