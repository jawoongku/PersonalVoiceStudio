"""Create the on-disk project layout used by the recording workflow."""

from __future__ import annotations

from pathlib import Path


def initialize_project(root: str | Path, *, overwrite: bool = False) -> dict[str, str]:
    project = Path(root).expanduser()
    if project.exists() and any(project.iterdir()) and not overwrite:
        raise FileExistsError(f"project directory is not empty: {project}")
    dataset = project / "data" / "my_voice"
    (dataset / "raw").mkdir(parents=True, exist_ok=True)
    (project / "artifacts").mkdir(parents=True, exist_ok=True)
    transcripts = dataset / "transcripts.csv"
    if overwrite or not transcripts.exists():
        transcripts.write_text("filename,text\n", encoding="utf-8")
    config = project / "training.yaml"
    if overwrite or not config.exists():
        config.write_text(
            "model_dir: /Users/jawoongku/Models/Fun-CosyVoice3-0.5B\n"
            "dataset_dir: data/my_voice_prepared\n"
            "output_dir: artifacts/runs/my_voice\n"
            "training:\n  batch_size: 1\n  grad_accum_steps: 1\n  max_epochs: 1\n"
            "lora:\n  rank: 2\n  alpha: 4\n  dropout: 0.0\n",
            encoding="utf-8",
        )
    return {"project": str(project), "dataset": str(dataset), "config": str(config)}
