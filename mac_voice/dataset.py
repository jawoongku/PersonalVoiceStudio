"""Dataset validation and preparation utilities using local standard tools."""

from __future__ import annotations

import csv
import json
import random
import shutil
import subprocess
import wave
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class AudioRecord:
    filename: str
    text: str
    path: str
    duration: float | None = None
    sample_rate: int | None = None
    channels: int | None = None
    frames: int | None = None
    peak: float | None = None
    rms: float | None = None
    clipping: bool | None = None
    silence_ratio: float | None = None
    errors: list[str] | None = None


def _pcm_stats(raw: bytes, sample_width: int) -> tuple[float, float, bool, float]:
    if sample_width not in (1, 2, 3, 4) or not raw:
        return 0.0, 0.0, False, 1.0
    max_value = float((1 << (8 * sample_width - 1)) - 1)
    values: list[int] = []
    if sample_width == 1:
        values = [byte - 128 for byte in raw]
    elif sample_width == 2:
        values = [int.from_bytes(raw[i : i + 2], "little", signed=True) for i in range(0, len(raw) - 1, 2)]
    elif sample_width == 3:
        for i in range(0, len(raw) - 2, 3):
            chunk = raw[i : i + 3]
            sign = b"\xff" if chunk[2] & 0x80 else b"\x00"
            values.append(int.from_bytes(chunk + sign, "little", signed=True))
    else:
        values = [int.from_bytes(raw[i : i + 4], "little", signed=True) for i in range(0, len(raw) - 3, 4)]
    if not values:
        return 0.0, 0.0, False, 1.0
    normalized = [value / max_value for value in values]
    peak = max(abs(value) for value in normalized)
    rms = (sum(value * value for value in normalized) / len(normalized)) ** 0.5
    clipping = any(abs(value) >= 0.999 for value in normalized)
    silence_ratio = sum(abs(value) < 0.001 for value in normalized) / len(normalized)
    return peak, rms, clipping, silence_ratio


def _read_transcripts(path: Path) -> tuple[dict[str, str], list[str]]:
    errors: list[str] = []
    mapping: dict[str, str] = {}
    if not path.is_file():
        return mapping, [f"missing transcript file: {path}"]
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            rows = csv.DictReader(handle)
            if not rows.fieldnames or not {"filename", "text"}.issubset(rows.fieldnames):
                return mapping, ["transcripts.csv must contain filename,text columns"]
            for line_number, row in enumerate(rows, start=2):
                filename = (row.get("filename") or "").strip()
                text = (row.get("text") or "").strip()
                if not filename:
                    errors.append(f"line {line_number}: empty filename")
                    continue
                if filename in mapping:
                    errors.append(f"line {line_number}: duplicate filename {filename}")
                mapping[filename] = text
    except (OSError, UnicodeError, csv.Error) as exc:
        errors.append(f"cannot read transcript file: {exc}")
    return mapping, errors


def validate_dataset(dataset: str | Path) -> tuple[list[AudioRecord], list[str]]:
    root = Path(dataset).expanduser()
    raw_dir = root / "raw"
    transcripts, errors = _read_transcripts(root / "transcripts.csv")
    records: list[AudioRecord] = []
    if not raw_dir.is_dir():
        errors.append(f"missing raw audio directory: {raw_dir}")
        return records, errors
    wavs = sorted(raw_dir.glob("*.wav"))
    if not wavs:
        errors.append(f"no WAV files found in {raw_dir}")
    for audio_path in wavs:
        filename = audio_path.name
        text = transcripts.get(filename, "")
        record = AudioRecord(filename=filename, text=text, path=str(audio_path), errors=[])
        if filename not in transcripts:
            record.errors.append("transcript missing")
        elif not text:
            record.errors.append("transcript is empty")
        try:
            with wave.open(str(audio_path), "rb") as audio:
                record.frames = audio.getnframes()
                record.sample_rate = audio.getframerate()
                record.channels = audio.getnchannels()
                sample_width = audio.getsampwidth()
                raw = audio.readframes(record.frames)
                record.duration = record.frames / record.sample_rate if record.sample_rate else 0.0
                record.peak, record.rms, record.clipping, record.silence_ratio = _pcm_stats(raw, sample_width)
                if record.duration > 30:
                    record.errors.append("duration exceeds 30 seconds")
                if record.sample_rate <= 0:
                    record.errors.append("invalid sample rate")
                if record.channels <= 0:
                    record.errors.append("invalid channel count")
        except (OSError, wave.Error) as exc:
            record.errors.append(f"WAV decode failed: {exc}")
        records.append(record)
    return records, errors


def render_validation(records: list[AudioRecord], errors: list[str]) -> str:
    lines = []
    for error in errors:
        lines.append(f"[ERROR] {error}")
    for record in records:
        if record.errors:
            lines.append(f"[ERROR] {record.filename}: {', '.join(record.errors)}")
        else:
            lines.append(
                f"[OK] {record.filename}: {record.duration:.2f}s, {record.sample_rate}Hz, "
                f"{record.channels}ch, rms={record.rms:.4f}, silence={record.silence_ratio:.2%}"
            )
    if not records and not errors:
        lines.append("[ERROR] dataset is empty")
    return "\n".join(lines)


def _convert_audio(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required for preparation but was not found")
    command = [
        "ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", str(source),
        "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", str(destination),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or f"ffmpeg failed with code {result.returncode}")


def prepare_dataset(dataset: str | Path, output: str | Path, seed: int = 42, dev_ratio: float = 0.1) -> dict:
    records, errors = validate_dataset(dataset)
    if errors or any(record.errors for record in records):
        raise ValueError(render_validation(records, errors))
    source_root = Path(dataset).expanduser()
    output_root = Path(output).expanduser()
    prepared_root = output_root / "audio"
    output_root.mkdir(parents=True, exist_ok=True)
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    dev_count = 0 if len(shuffled) < 2 else max(1, round(len(shuffled) * dev_ratio))
    dev_names = {record.filename for record in shuffled[:dev_count]}
    splits = {"train": [], "dev": []}
    for record in records:
        split = "dev" if record.filename in dev_names else "train"
        destination = prepared_root / split / record.filename
        _convert_audio(Path(record.path), destination)
        splits[split].append({"filename": record.filename, "text": record.text, "path": str(destination)})
    manifest = {"seed": seed, "dev_ratio": dev_ratio, "sample_rate": 24000, "splits": splits}
    (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for split, items in splits.items():
        split_dir = output_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / "text").write_text(
            "".join(f"{Path(item['filename']).stem}\t{item['text']}\n" for item in items), encoding="utf-8"
        )
        (split_dir / "wav.scp").write_text("".join(f"{Path(item['filename']).stem}\t{item['path']}\n" for item in items), encoding="utf-8")
        (split_dir / "utt2spk").write_text("".join(f"{Path(item['filename']).stem}\towner\n" for item in items), encoding="utf-8")
        (split_dir / "spk2utt").write_text(f"owner\t{' '.join(Path(item['filename']).stem for item in items)}\n", encoding="utf-8")
        (split_dir / "instruct").write_text(
            "".join(
                f"{Path(item['filename']).stem}\\tYou are a helpful assistant.<|endofprompt|>\\n"
                for item in items
            ),
            encoding="utf-8",
        )
    return manifest
