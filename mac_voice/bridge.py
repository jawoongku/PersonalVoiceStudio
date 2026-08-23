"""Read-only data bridge primitives for a future SwiftUI subprocess/API."""

from __future__ import annotations

from .catalog import list_voice_packages
from .jobs import read_job
from .runs import list_runs
from .mps_runtime import probe as probe_mps_runtime


def job_snapshot(path: str) -> dict:
    return read_job(path)


def voice_catalog(root: str) -> list[dict]:
    return list_voice_packages(root)


def run_catalog(root: str) -> list[dict]:
    return list_runs(root)


def mps_snapshot() -> dict:
    return probe_mps_runtime()
