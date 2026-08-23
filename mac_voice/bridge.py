"""Read-only data bridge primitives for a future SwiftUI subprocess/API."""

from __future__ import annotations

from .catalog import list_voice_packages
from .jobs import read_job
from .runs import list_runs
from .mps_runtime import probe as probe_mps_runtime


def job_snapshot(path: str) -> dict:
    job = read_job(path)
    metrics = job.get("metrics")
    if isinstance(metrics, dict):
        job["metrics"] = {key: value for key, value in metrics.items() if isinstance(value, (int, float)) and not isinstance(value, bool)}
    return job


def voice_catalog(root: str) -> list[dict]:
    return list_voice_packages(root)


def run_catalog(root: str) -> list[dict]:
    return list_runs(root)


def mps_snapshot() -> dict:
    return probe_mps_runtime()
