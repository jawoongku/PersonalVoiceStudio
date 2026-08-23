"""Environment checks that do not require the full training stack."""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_MODEL_DIR = Path("/Users/jawoongku/Models/Fun-CosyVoice3-0.5B")
REQUIRED_MODEL_FILES = (
    "llm.pt",
    "flow.pt",
    "hift.pt",
    "speech_tokenizer_v3.onnx",
    "campplus.onnx",
)


@dataclass
class Check:
    label: str
    ok: bool | None
    detail: str


def _version(module_name: str) -> str:
    try:
        module = __import__(module_name)
        return str(getattr(module, "__version__", "unknown"))
    except Exception as exc:  # pragma: no cover - diagnostic path
        return f"unavailable ({exc.__class__.__name__})"


def _torch_info() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {"installed": False, "error": str(exc)}
    mps_built = bool(getattr(torch.backends.mps, "is_built", lambda: False)())
    mps_available = bool(getattr(torch.backends.mps, "is_available", lambda: False)())
    mps_error = None
    if mps_built and not mps_available:
        try:
            torch.ones(1, device="mps")
        except Exception as exc:
            mps_error = f"{exc.__class__.__name__}: {exc}"
    allocated = None
    if mps_available:
        try:
            allocated = int(torch.mps.current_allocated_memory())
        except Exception:
            allocated = None
    return {
        "installed": True,
        "version": str(torch.__version__),
        "mps_built": mps_built,
        "mps_available": mps_available,
        "mps_error": mps_error,
        "allocated_memory": allocated,
    }


def _onnx_info() -> dict[str, Any]:
    try:
        import onnxruntime as ort
    except Exception as exc:
        return {"installed": False, "error": str(exc)}
    return {
        "installed": True,
        "version": str(ort.__version__),
        "providers": list(ort.get_available_providers()),
    }


def _model_dir(value: str | None) -> Path:
    return Path(value or os.environ.get("COSYVOICE_MODEL_DIR") or DEFAULT_MODEL_DIR).expanduser()


def _macos_version() -> str:
    try:
        result = subprocess.run(
            ["sw_vers", "-productVersion"],
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or "unknown"
    except OSError:
        return "unavailable"


def inspect_environment(model_dir: str | None = None, upstream_root: str | None = None) -> list[Check]:
    model_path = _model_dir(model_dir)
    upstream_path = Path(upstream_root or os.environ.get("COSYVOICE_SOURCE_DIR") or "/Users/jawoongku/CosyVoice").expanduser()
    torch_info = _torch_info()
    onnx_info = _onnx_info()
    checks: list[Check] = [
        Check("architecture", platform.machine() == "arm64", platform.machine()),
        Check("Apple Silicon detected", platform.machine() == "arm64", "arm64 required"),
        Check("macOS", True, _macos_version()),
        Check("Python", True, sys.version.split()[0]),
        Check("PyTorch installed", torch_info["installed"], torch_info.get("error", torch_info.get("version", ""))),
        Check("MPS built", torch_info.get("mps_built", False), str(torch_info.get("mps_built", False))),
        Check("MPS available", torch_info.get("mps_available", False), str(torch_info.get("mps_available", False))),
        Check("MPS probe", None if torch_info.get("mps_available") else False, torch_info.get("mps_error") or "not run"),
        Check("torchaudio", importlib.util.find_spec("torchaudio") is not None, _version("torchaudio")),
        Check("ONNX Runtime", onnx_info["installed"], onnx_info.get("error", onnx_info.get("version", ""))),
        Check("ONNX providers", bool(onnx_info.get("providers")), ", ".join(onnx_info.get("providers", [])) or "none"),
        Check("model directory", model_path.is_dir(), str(model_path)),
        Check("ffmpeg", shutil.which("ffmpeg") is not None, shutil.which("ffmpeg") or "not found"),
        Check("sox", shutil.which("sox") is not None, shutil.which("sox") or "not found"),
        Check("CosyVoice checkout", upstream_path.is_dir(), str(upstream_path)),
    ]
    if upstream_path.is_dir():
        try:
            result = subprocess.run(["git", "-C", str(upstream_path), "rev-parse", "HEAD"], check=False, capture_output=True, text=True)
            checks.append(Check("CosyVoice commit", result.returncode == 0, result.stdout.strip() or "not a git checkout"))
        except OSError:
            checks.append(Check("CosyVoice commit", False, "git unavailable"))
    for filename in REQUIRED_MODEL_FILES:
        path = model_path / filename
        checks.append(Check(filename, path.is_file(), str(path)))
    return checks


def render_checks(checks: list[Check]) -> str:
    lines = []
    for check in checks:
        prefix = "[OK]" if check.ok else "[MISSING]" if check.ok is False else "[INFO]"
        lines.append(f"{prefix} {check.label}: {check.detail}")
    return "\n".join(lines)


def run_doctor(model_dir: str | None = None, upstream_root: str | None = None) -> int:
    checks = inspect_environment(model_dir, upstream_root)
    print(render_checks(checks))
    return 0 if all(check.ok is not False for check in checks) else 1
