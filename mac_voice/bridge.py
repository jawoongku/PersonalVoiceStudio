"""Read-only data bridge primitives for a future SwiftUI subprocess/API."""

from __future__ import annotations

from .catalog import list_voice_packages
from .jobs import read_job


def job_snapshot(path: str) -> dict:
    return read_job(path)


def voice_catalog(root: str) -> list[dict]:
    return list_voice_packages(root)
