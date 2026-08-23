"""Actionable MPS runtime diagnostics.

This module does not silently enable CPU fallback for training.  It reports
whether the current interpreter can actually create an MPS tensor and, when
the check fails, gives the next environment-level action.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from typing import Any


def _macos_version() -> str:
    try:
        result = subprocess.run(["sw_vers", "-productVersion"], check=False, capture_output=True, text=True)
        return result.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def _major(version: str) -> int | None:
    try:
        return int(version.split(".", 1)[0])
    except (AttributeError, ValueError):
        return None


def probe() -> dict[str, Any]:
    """Return a serialisable, non-mutating MPS readiness report."""
    report: dict[str, Any] = {
        "architecture": platform.machine(),
        "macos": _macos_version(),
        "python": sys.version.split()[0],
        "torch": None,
        "built": False,
        "available": False,
        "tensor_probe": False,
        "error": None,
        "status": "missing",
        "action": "Install a PyTorch build with MPS support in the selected environment.",
    }
    try:
        import torch
    except Exception as exc:
        report["error"] = f"{exc.__class__.__name__}: {exc}"
        return report
    report["torch"] = str(torch.__version__)
    backend = getattr(torch.backends, "mps", None)
    report["built"] = bool(backend and backend.is_built())
    report["available"] = bool(backend and backend.is_available())
    if report["available"]:
        try:
            torch.ones(1, device="mps")
            report["tensor_probe"] = True
            report["status"] = "ready"
            report["action"] = "MPS is ready for a real model smoke test."
            return report
        except Exception as exc:  # pragma: no cover - backend dependent
            report["error"] = f"{exc.__class__.__name__}: {exc}"
    else:
        try:
            torch.ones(1, device="mps")
        except Exception as exc:
            report["error"] = f"{exc.__class__.__name__}: {exc}"
    if not report["built"]:
        report["status"] = "not-built"
        report["action"] = "Install the Apple Silicon/macOS PyTorch wheel for this environment."
    elif _major(report["macos"]) and _major(report["macos"]) >= 26:
        report["status"] = "os-runtime-mismatch"
        report["action"] = (
            "The installed PyTorch MPS runtime does not recognise this macOS major version. "
            "Use a PyTorch build that explicitly supports this macOS release, or run the project "
            "on a supported macOS release; do not mark MPS training complete until tensor_probe passes."
        )
    else:
        report["status"] = "unavailable"
        report["action"] = "Check Apple Silicon, macOS version, and the selected Python environment."
    return report


def render(report: dict[str, Any]) -> str:
    lines = [
        f"[INFO] architecture: {report['architecture']}",
        f"[INFO] macOS: {report['macos']}",
        f"[INFO] Python: {report['python']}",
        f"[INFO] PyTorch: {report['torch'] or 'unavailable'}",
        f"[INFO] MPS built: {report['built']}",
        f"[INFO] MPS available: {report['available']}",
        f"[INFO] MPS tensor probe: {report['tensor_probe']}",
        f"[{ 'OK' if report['status'] == 'ready' else 'ACTION' }] status: {report['status']}",
        f"[ACTION] {report['action']}",
    ]
    if report.get("error"):
        lines.append(f"[DETAIL] {report['error']}")
    return "\n".join(lines)
