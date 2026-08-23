"""CLI dispatcher. Commands are added incrementally by implementation phase."""

from __future__ import annotations

import argparse
import os
import json
from pathlib import Path

from .dataset import prepare_dataset, render_validation, validate_dataset
from .doctor import run_doctor
from .features import build_parquet, extract_embeddings, extract_speech_tokens, inspect_feature_inputs, validate_feature_artifacts
from .package import build_voice_package
from .config import load_config, validate_training_config
from .baseline import _load_upstream, run_baseline, run_zero_shot
from .adapter import inspect_lora_targets
from .voice import require_adapter_inference_support, validate_voice_package
from .synth import run_synth
from .compare import run_comparison
from .narrate import run_narrate
from .parquet import validate_data_list
from .mps_smoke import run_mps_smoke
from .mps_baseline import run_mps_baseline
from .mps_training import run_user_parquet_mps_backward, run_user_parquet_mps_resume, run_user_parquet_mps_train
from .mps_runtime import probe as probe_mps_runtime, render as render_mps_runtime
from .ui_gradio import launch_ui
from .project import initialize_project
from .jobs import read_job
from .catalog import list_voice_packages
from .bridge import job_snapshot, mps_snapshot, run_catalog, voice_catalog
from .similarity import cosine_similarity, evaluate_audio_similarity
from .runs import list_runs
from .retention import retention_plan
from .artifacts import create_manifest, verify_manifest
from .jobs import create_job, update_job
from .upstream_smoke import run_model_backward_smoke, run_model_forward_smoke, run_parquet_backward_smoke, run_parquet_resume_smoke, run_parquet_train_smoke


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m mac_voice")
    subparsers = parser.add_subparsers(dest="command", required=True)
    init_project = subparsers.add_parser("init-project", help="create a new voice project layout")
    init_project.add_argument("--root", required=True, help="new project directory")
    init_project.add_argument("--overwrite", action="store_true")
    job_status = subparsers.add_parser("job-status", help="read a filesystem-backed training job status")
    job_status.add_argument("--job", required=True, help="path to job.json")
    job_status.add_argument("--json", action="store_true", dest="as_json")
    voices = subparsers.add_parser("list-voices", help="list and validate local Voice Packages")
    voices.add_argument("--root", default="artifacts/voices")
    voices.add_argument("--json", action="store_true", dest="as_json")
    bridge_status = subparsers.add_parser("bridge-status", help="emit combined job and Voice Package JSON")
    bridge_status.add_argument("--job", required=True)
    bridge_status.add_argument("--voices", default="artifacts/voices")
    bridge_status.add_argument("--runs", default="artifacts/runs")
    similarity = subparsers.add_parser("similarity", help="calculate cosine similarity for two embedding vectors")
    similarity.add_argument("--left", help="comma-separated float vector")
    similarity.add_argument("--right", help="comma-separated float vector")
    similarity.add_argument("--left-file", help="JSON array embedding file")
    similarity.add_argument("--right-file", help="JSON array embedding file")
    speaker_similarity = subparsers.add_parser("speaker-similarity", help="score two WAV files with CAMPPlus embeddings")
    speaker_similarity.add_argument("--reference", required=True)
    speaker_similarity.add_argument("--generated", required=True)
    speaker_similarity.add_argument("--model", required=True, help="path to campplus.onnx")
    speaker_similarity.add_argument("--output", default=None, help="optional JSON report path")
    runs = subparsers.add_parser("list-runs", help="list local training runs and artifacts")
    runs.add_argument("--root", default="artifacts/runs")
    runs.add_argument("--json", action="store_true", dest="as_json")
    retention = subparsers.add_parser("retention-plan", help="show non-destructive training run retention candidates")
    retention.add_argument("--root", default="artifacts/runs")
    retention.add_argument("--keep", type=int, default=3)
    retention.add_argument("--json", action="store_true", dest="as_json")
    manifest = subparsers.add_parser("artifact-manifest", help="create a SHA-256 manifest for a run or Voice Package")
    manifest.add_argument("--root", required=True)
    manifest.add_argument("--output", default=None)
    artifact_verify = subparsers.add_parser("artifact-verify", help="verify a SHA-256 artifact manifest")
    artifact_verify.add_argument("--manifest", required=True)
    job_create = subparsers.add_parser("job-create", help="create a queued training job metadata file")
    job_create.add_argument("--output", required=True, help="job output directory")
    job_create.add_argument("--command", dest="job_command", default="train")
    job_create.add_argument("--config", required=True)
    job_update = subparsers.add_parser("job-update", help="update a training job status")
    job_update.add_argument("--job", required=True)
    job_update.add_argument("--status", required=True, choices=("queued", "running", "completed", "failed", "cancelled"))
    job_update.add_argument("--step", type=int, default=None)
    job_update.add_argument("--error", default=None)
    train_job = subparsers.add_parser("parquet-train-job", help="run real CPU parquet smoke training with job state")
    train_job.add_argument("--train-data-list", required=True)
    train_job.add_argument("--dev-data-list", required=True)
    train_job.add_argument("--model-dir", required=True)
    train_job.add_argument("--output", required=True)
    train_job.add_argument("--job", required=True)
    train_job.add_argument("--upstream-root", default="/Users/jawoongku/CosyVoice")
    train_job.add_argument("--steps", type=int, default=2)
    resume_job = subparsers.add_parser("parquet-resume-job", help="resume a CPU parquet job with job state")
    resume_job.add_argument("--data-list", required=True)
    resume_job.add_argument("--adapter", required=True)
    resume_job.add_argument("--state", required=True)
    resume_job.add_argument("--model-dir", required=True)
    resume_job.add_argument("--output", required=True)
    resume_job.add_argument("--job", required=True)
    resume_job.add_argument("--upstream-root", default="/Users/jawoongku/CosyVoice")
    package_job = subparsers.add_parser("package-job", help="build a Voice Package from a completed run")
    package_job.add_argument("--run", required=True)
    package_job.add_argument("--name", required=True)
    package_job.add_argument("--output", required=True)
    package_job.add_argument("--base-model", required=True)
    package_job.add_argument("--job", required=True)
    package_job.add_argument("--upstream-root", default="/Users/jawoongku/CosyVoice")
    doctor = subparsers.add_parser("doctor", help="inspect the local Mac/CosyVoice environment")
    doctor.add_argument("--model-dir", default=None, help="CosyVoice3 model directory")
    doctor.add_argument("--upstream-root", default=None, help="CosyVoice checkout directory")
    validate = subparsers.add_parser("validate-data", help="validate WAV and transcript input")
    validate.add_argument("--dataset", required=True)
    prepare = subparsers.add_parser("prepare", help="normalize audio and create train/dev manifests")
    prepare.add_argument("--dataset", required=True)
    prepare.add_argument("--output", default=None)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--dev-ratio", type=float, default=0.1)
    features = subparsers.add_parser("features", help="check feature extraction prerequisites")
    features.add_argument("--dataset", required=True)
    features.add_argument("--model-dir", default=None)
    features.add_argument("--onnx-provider", choices=("auto", "coreml", "cpu"), default="auto")
    features.add_argument("--speech-token-model", default=None)
    features.add_argument("--upstream-root", default="/Users/jawoongku/CosyVoice")
    features.add_argument("--skip-parquet", action="store_true")
    package = subparsers.add_parser("package", help="build a reusable adapter-based Voice Package")
    package.add_argument("--run", required=True)
    package.add_argument("--name", required=True)
    package.add_argument("--output", required=True)
    package.add_argument("--base-model", default=None)
    package.add_argument("--upstream-root", default="/Users/jawoongku/CosyVoice")
    package.add_argument("--speaker-id", default="owner")
    package.add_argument("--language", default="ko")
    package.add_argument("--sample-rate", type=int, default=24000)
    train = subparsers.add_parser("train", help="validate and run the local LoRA trainer")
    train.add_argument("--config", required=True)
    train.add_argument("--max-steps", type=int, default=None)
    train.add_argument("--resume", default=None)
    baseline = subparsers.add_parser("baseline", help="run base CosyVoice inference")
    baseline.add_argument("--model-dir", default=None)
    baseline.add_argument("--text", required=True)
    baseline.add_argument("--output", required=True)
    baseline.add_argument("--upstream-root", default="/Users/jawoongku/CosyVoice")
    baseline.add_argument("--speaker-id", default=None)
    clone = subparsers.add_parser("clone", help="run CosyVoice zero-shot reference inference")
    clone.add_argument("--model-dir", default=None)
    clone.add_argument("--reference", required=True)
    clone.add_argument("--reference-text", required=True)
    clone.add_argument("--text", required=True)
    clone.add_argument("--output", required=True)
    clone.add_argument("--upstream-root", default="/Users/jawoongku/CosyVoice")
    synth = subparsers.add_parser("synth", help="validate a Voice Package for adapter inference")
    synth.add_argument("--voice", required=True)
    synth.add_argument("--text", required=True)
    synth.add_argument("--output", required=True)
    synth.add_argument("--model-dir", default=None)
    compare = subparsers.add_parser("compare", help="compare base zero-shot and adapter output")
    compare.add_argument("--voice", required=True)
    compare.add_argument("--text", required=True)
    compare.add_argument("--output-dir", required=True)
    compare.add_argument("--model-dir", default=None)
    compare.add_argument("--upstream-root", default="/Users/jawoongku/CosyVoice")
    narrate = subparsers.add_parser("narrate", help="synthesize a long script in chunks")
    narrate.add_argument("--voice", required=True)
    narrate.add_argument("--input", required=True)
    narrate.add_argument("--output", required=True)
    narrate.add_argument("--model-dir", default=None)
    narrate.add_argument("--upstream-root", default="/Users/jawoongku/CosyVoice")
    narrate.add_argument("--max-chars", type=int, default=180)
    parquet = subparsers.add_parser("validate-parquet", help="validate CosyVoice parquet data.list")
    parquet.add_argument("--data-list", required=True)
    parquet.add_argument("--require-features", action="store_true")
    mps_smoke = subparsers.add_parser("mps-smoke", help="run a real MPS forward/backward/optimizer probe")
    mps_baseline = subparsers.add_parser("mps-baseline", help="run CosyVoice3 baseline inference on MPS")
    mps_baseline.add_argument("--model-dir", required=True)
    mps_baseline.add_argument("--text", required=True)
    mps_baseline.add_argument("--output", required=True)
    mps_baseline.add_argument("--speaker-id", default=None)
    mps_baseline.add_argument("--reference", default=None)
    mps_baseline.add_argument("--reference-text", default=None)
    mps_baseline.add_argument("--upstream-root", default="/Users/jawoongku/CosyVoice")
    mps_parquet = subparsers.add_parser("mps-parquet-backward", help="run one real user parquet batch on MPS")
    mps_parquet.add_argument("--data-list", required=True)
    mps_parquet.add_argument("--model-dir", required=True)
    mps_parquet.add_argument("--upstream-root", default="/Users/jawoongku/CosyVoice")
    mps_train = subparsers.add_parser("mps-parquet-train", help="run one full user parquet epoch on MPS")
    mps_train.add_argument("--train-data-list", required=True)
    mps_train.add_argument("--dev-data-list", required=True)
    mps_train.add_argument("--model-dir", required=True)
    mps_train.add_argument("--output", required=True)
    mps_train.add_argument("--upstream-root", default="/Users/jawoongku/CosyVoice")
    mps_resume = subparsers.add_parser("mps-parquet-resume", help="resume one user parquet MPS step from adapter/state")
    mps_resume.add_argument("--data-list", required=True)
    mps_resume.add_argument("--model-dir", required=True)
    mps_resume.add_argument("--adapter", required=True)
    mps_resume.add_argument("--state", required=True)
    mps_resume.add_argument("--output", required=True)
    mps_resume.add_argument("--upstream-root", default="/Users/jawoongku/CosyVoice")
    mps_doctor = subparsers.add_parser("mps-doctor", help="diagnose MPS runtime compatibility without changing the environment")
    mps_doctor.add_argument("--json", action="store_true", dest="as_json")
    inspect_model = subparsers.add_parser("inspect-model", help="inspect runtime LoRA target modules")
    inspect_model.add_argument("--model-dir", default=None)
    inspect_model.add_argument("--upstream-root", default="/Users/jawoongku/CosyVoice")
    inspect_model.add_argument("--target", action="append", dest="targets", default=None)
    forward_smoke = subparsers.add_parser("model-forward-smoke", help="run one real CosyVoice3 CPU forward preflight")
    forward_smoke.add_argument("--model-dir", default=None)
    forward_smoke.add_argument("--upstream-root", default="/Users/jawoongku/CosyVoice")
    backward_smoke = subparsers.add_parser("model-backward-smoke", help="run a real CosyVoice3 CPU LoRA backward preflight")
    backward_smoke.add_argument("--model-dir", default=None)
    backward_smoke.add_argument("--upstream-root", default="/Users/jawoongku/CosyVoice")
    parquet_smoke = subparsers.add_parser("parquet-backward-smoke", help="run CPU LoRA backward on one real parquet row")
    parquet_smoke.add_argument("--data-list", required=True)
    parquet_smoke.add_argument("--model-dir", default=None)
    parquet_smoke.add_argument("--upstream-root", default="/Users/jawoongku/CosyVoice")
    parquet_train = subparsers.add_parser("parquet-train-smoke", help="run tiny CPU train/validation on real parquet rows")
    parquet_train.add_argument("--train-data-list", required=True)
    parquet_train.add_argument("--dev-data-list", required=True)
    parquet_train.add_argument("--output", required=True)
    parquet_train.add_argument("--model-dir", default=None)
    parquet_train.add_argument("--upstream-root", default="/Users/jawoongku/CosyVoice")
    parquet_train.add_argument("--steps", type=int, default=2)
    resume_smoke = subparsers.add_parser("parquet-resume-smoke", help="reload adapter/optimizer state and take one CPU step")
    resume_smoke.add_argument("--data-list", required=True)
    resume_smoke.add_argument("--adapter", required=True)
    resume_smoke.add_argument("--state", required=True)
    resume_smoke.add_argument("--output", required=True)
    resume_smoke.add_argument("--model-dir", default=None)
    resume_smoke.add_argument("--upstream-root", default="/Users/jawoongku/CosyVoice")
    subparsers.add_parser("ui", help="launch the optional Gradio recording UI")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "init-project":
        try:
            result = initialize_project(args.root, overwrite=args.overwrite)
        except (OSError, FileExistsError, ValueError) as exc:
            print(f"[ERROR] {exc}")
            return 1
        for key, value in result.items():
            print(f"[OK] {key}: {value}")
        return 0
    if args.command == "job-status":
        try:
            job = read_job(args.job)
        except (OSError, ValueError) as exc:
            print(f"[ERROR] {exc}")
            return 1
        if args.as_json:
            print(json.dumps(job, ensure_ascii=False))
            return 0
        print(f"[OK] status: {job['status']}")
        for key in ("command", "config", "pid", "step", "package", "error", "updated_at"):
            if key in job and job[key] is not None:
                print(f"[INFO] {key}: {job[key]}")
        if job.get("pid"):
            try:
                os.kill(int(job["pid"]), 0)
                print("[INFO] process: alive")
            except (OSError, ValueError):
                print("[INFO] process: not running")
        metrics = job.get("metrics")
        if isinstance(metrics, dict):
            if not job.get("step") and "steps" in metrics:
                print(f"[INFO] step: {metrics['steps']}")
            for key in ("train_loss", "dev_loss", "checkpoint", "state"):
                if key in metrics:
                    print(f"[INFO] {key}: {metrics[key]}")
        return 0
    if args.command == "list-voices":
        voice_rows = list_voice_packages(args.root)
        if args.as_json:
            print(json.dumps(voice_rows, ensure_ascii=False))
            return 0
        for voice in voice_rows:
            status = "valid" if voice["valid"] else "invalid"
            print(f"[{status}] {voice['name']} ({voice['path']})")
            if voice.get("language"):
                print(f"  language={voice['language']} sample_rate={voice.get('sample_rate')}")
            for error in voice["errors"]:
                print(f"  error: {error}")
        return 0
    if args.command == "bridge-status":
        try:
            payload = {"job": job_snapshot(args.job), "voices": voice_catalog(args.voices), "runs": run_catalog(args.runs), "mps": mps_snapshot()}
        except (OSError, ValueError) as exc:
            print(f"[ERROR] {exc}")
            return 1
        print(json.dumps(payload, ensure_ascii=False))
        return 0
    if args.command == "similarity":
        try:
            if args.left_file or args.right_file:
                if not args.left_file or not args.right_file:
                    raise ValueError("left-file and right-file must be provided together")
                left = json.loads(Path(args.left_file).read_text(encoding="utf-8"))
                right = json.loads(Path(args.right_file).read_text(encoding="utf-8"))
            else:
                if not args.left or not args.right:
                    raise ValueError("left/right or left-file/right-file is required")
                left = [float(value) for value in args.left.split(",") if value.strip()]
                right = [float(value) for value in args.right.split(",") if value.strip()]
            print(f"{cosine_similarity(left, right):.6f}")
        except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
            print(f"[ERROR] {exc}")
            return 1
        return 0
    if args.command == "speaker-similarity":
        try:
            report = evaluate_audio_similarity(args.reference, args.generated, args.model)
            if args.output:
                Path(args.output).expanduser().write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(report, ensure_ascii=False))
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            print(f"[ERROR] {exc}")
            return 1
        return 0
    if args.command == "list-runs":
        rows = list_runs(args.root)
        if args.as_json:
            print(json.dumps(rows, ensure_ascii=False))
        else:
            for row in rows:
                print(f"[{row['job_status'] or 'unknown'}] {row['name']} checkpoint={row['checkpoint'] or '-'}")
        return 0
    if args.command == "retention-plan":
        try:
            plan = retention_plan(args.root, keep=args.keep)
        except (OSError, ValueError) as exc:
            print(f"[ERROR] {exc}")
            return 1
        if args.as_json:
            print(json.dumps(plan, ensure_ascii=False))
        else:
            print(f"[KEEP] {len(plan['keep'])} run(s)")
            for path in plan["keep"]:
                print(f"  {path}")
            print(f"[CANDIDATE] {len(plan['candidates'])} run(s); no files were deleted")
            for path in plan["candidates"]:
                print(f"  {path}")
        return 0
    if args.command == "artifact-manifest":
        try:
            print(f"[OK] manifest: {create_manifest(args.root, args.output)}")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"[ERROR] {exc}")
            return 1
        return 0
    if args.command == "artifact-verify":
        try:
            errors = verify_manifest(args.manifest)
        except (OSError, ValueError, json.JSONDecodeError, KeyError) as exc:
            print(f"[ERROR] {exc}")
            return 1
        if errors:
            for error in errors:
                print(f"[ERROR] {error}")
            return 1
        print("[OK] artifact manifest verified")
        return 0
    if args.command == "job-create":
        try:
            path = create_job(args.output, command=args.job_command, config=args.config)
        except (OSError, FileExistsError, ValueError) as exc:
            print(f"[ERROR] {exc}")
            return 1
        print(f"[OK] job: {path}")
        return 0
    if args.command == "job-update":
        fields = {}
        if args.step is not None:
            fields["step"] = args.step
        if args.error is not None:
            fields["error"] = args.error
        try:
            job = update_job(args.job, args.status, **fields)
        except (OSError, ValueError) as exc:
            print(f"[ERROR] {exc}")
            return 1
        print(f"[OK] status: {job['status']}")
        return 0
    if args.command == "parquet-train-job":
        try:
            update_job(args.job, "running")
            result = run_parquet_train_smoke(args.train_data_list, args.dev_data_list, args.model_dir, args.upstream_root, args.output, steps=args.steps)
            update_job(args.job, "completed", step=result.get("step", result.get("steps")), metrics=result)
        except (ImportError, OSError, RuntimeError, ValueError, FloatingPointError, KeyError) as exc:
            try:
                update_job(args.job, "failed", error=str(exc))
            except (OSError, ValueError):
                pass
            print(f"[ERROR] {exc}")
            return 1
        print(f"[OK] training job completed: step={result.get('step', result.get('steps'))} output={args.output}")
        return 0
    if args.command == "parquet-resume-job":
        try:
            update_job(args.job, "running")
            result = run_parquet_resume_smoke(args.data_list, args.adapter, args.state, args.model_dir, args.upstream_root, args.output)
            update_job(args.job, "completed", step=result.get("step"), metrics=result)
        except (ImportError, OSError, RuntimeError, ValueError, FloatingPointError, KeyError) as exc:
            try:
                update_job(args.job, "failed", error=str(exc))
            except (OSError, ValueError):
                pass
            print(f"[ERROR] {exc}")
            return 1
        print(f"[OK] resume job completed: output={args.output}")
        return 0
    if args.command == "package-job":
        try:
            update_job(args.job, "running")
            destination = build_voice_package(args.run, args.name, args.output, base_model=args.base_model, upstream_root=args.upstream_root)
            update_job(args.job, "completed", package=str(destination))
        except (OSError, RuntimeError, ValueError) as exc:
            try:
                update_job(args.job, "failed", error=str(exc))
            except (OSError, ValueError):
                pass
            print(f"[ERROR] {exc}")
            return 1
        print(f"[OK] Voice Package created: {destination}")
        return 0
    if args.command == "doctor":
        return run_doctor(args.model_dir, args.upstream_root)
    if args.command == "mps-doctor":
        report = probe_mps_runtime()
        print(json.dumps(report, ensure_ascii=False) if args.as_json else render_mps_runtime(report))
        return 0 if report["status"] == "ready" else 1
    if args.command == "mps-baseline":
        try:
            output = run_mps_baseline(args.model_dir, args.text, args.output, upstream_root=args.upstream_root, speaker_id=args.speaker_id, reference=args.reference, reference_text=args.reference_text)
        except (ImportError, OSError, RuntimeError, ValueError, FloatingPointError) as exc:
            print(f"[ERROR] {exc}")
            return 1
        print(f"[OK] MPS baseline WAV: {output}")
        return 0
    if args.command == "mps-parquet-backward":
        try:
            result = run_user_parquet_mps_backward(args.data_list, args.model_dir, args.upstream_root)
        except (ImportError, OSError, RuntimeError, ValueError, FloatingPointError, KeyError) as exc:
            print(f"[ERROR] {exc}")
            return 1
        print(json.dumps(result, ensure_ascii=False))
        return 0
    if args.command == "mps-parquet-train":
        try:
            result = run_user_parquet_mps_train(args.train_data_list, args.dev_data_list, args.model_dir, args.output, args.upstream_root)
        except (ImportError, OSError, RuntimeError, ValueError, FloatingPointError, KeyError) as exc:
            print(f"[ERROR] {exc}")
            return 1
        print(json.dumps(result, ensure_ascii=False))
        return 0
    if args.command == "mps-parquet-resume":
        try:
            result = run_user_parquet_mps_resume(args.data_list, args.model_dir, args.adapter, args.state, args.output, args.upstream_root)
        except (ImportError, OSError, RuntimeError, ValueError, FloatingPointError, KeyError) as exc:
            print(f"[ERROR] {exc}")
            return 1
        print(json.dumps(result, ensure_ascii=False))
        return 0
    if args.command == "ui":
        try:
            launch_ui()
        except RuntimeError as exc:
            print(f"[ERROR] {exc}")
            return 1
        return 0
    if args.command == "inspect-model":
        model_dir = args.model_dir or "/Users/jawoongku/Models/Fun-CosyVoice3-0.5B"
        try:
            module = _load_upstream(args.upstream_root)
            model = module.AutoModel(model_dir=model_dir, load_trt=False, fp16=False)
            matches = inspect_lora_targets(model, tuple(args.targets) if args.targets else None)
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            print(f"[ERROR] {exc}")
            return 1
        for target, names in matches.items():
            print(f"[OK] {target}: {len(names)} matches")
            for name in names[:5]:
                print(f"  - {name}")
        return 0
    if args.command == "model-forward-smoke":
        model_dir = args.model_dir or "/Users/jawoongku/Models/Fun-CosyVoice3-0.5B"
        try:
            result = run_model_forward_smoke(model_dir, args.upstream_root)
        except (ImportError, OSError, RuntimeError, ValueError, FloatingPointError) as exc:
            print(f"[ERROR] {exc}")
            return 1
        print(f"[OK] CPU CosyVoice3 forward: loss={result['loss']:.6f}, LoRA matches={result['matched_modules']}, trainable={result['trainable']}")
        print("[INFO] synthetic batch only; backward and optimizer.step remain unverified")
        return 0
    if args.command == "model-backward-smoke":
        model_dir = args.model_dir or "/Users/jawoongku/Models/Fun-CosyVoice3-0.5B"
        try:
            result = run_model_backward_smoke(model_dir, args.upstream_root)
        except (ImportError, OSError, RuntimeError, ValueError, FloatingPointError) as exc:
            print(f"[ERROR] {exc}")
            return 1
        print(f"[OK] CPU CosyVoice3 backward/step: loss={result['loss']:.6f}, LoRA matches={result['matched_modules']}, trainable={result['trainable']}")
        print("[INFO] synthetic batch and CPU only; MPS and user-data training remain unverified")
        return 0
    if args.command == "parquet-backward-smoke":
        model_dir = args.model_dir or "/Users/jawoongku/Models/Fun-CosyVoice3-0.5B"
        try:
            result = run_parquet_backward_smoke(args.data_list, model_dir, args.upstream_root)
        except (ImportError, OSError, RuntimeError, ValueError, FloatingPointError, KeyError) as exc:
            print(f"[ERROR] {exc}")
            return 1
        print(f"[OK] parquet CPU backward/step: utt={result['utt']}, loss={result['loss']:.6f}, LoRA matches={result['matched_modules']}, trainable={result['trainable']}")
        print("[INFO] real feature parquet row but CPU only; MPS multi-step training remains unverified")
        return 0
    if args.command == "parquet-train-smoke":
        model_dir = args.model_dir or "/Users/jawoongku/Models/Fun-CosyVoice3-0.5B"
        try:
            result = run_parquet_train_smoke(args.train_data_list, args.dev_data_list, model_dir, args.upstream_root, args.output, steps=args.steps)
        except (ImportError, OSError, RuntimeError, ValueError, FloatingPointError, KeyError) as exc:
            print(f"[ERROR] {exc}")
            return 1
        print(f"[OK] parquet CPU train smoke: steps={result['steps']}, train_loss={result['train_loss']:.6f}, dev_loss={result['dev_loss']:.6f}")
        print(f"[OK] adapter checkpoint: {result['checkpoint']}")
        print(f"[OK] training state: {result['state']}")
        print(f"[OK] metrics: {result['metrics']}")
        print("[INFO] CPU-only smoke; MPS and full user dataset training remain unverified")
        return 0
    if args.command == "parquet-resume-smoke":
        model_dir = args.model_dir or "/Users/jawoongku/Models/Fun-CosyVoice3-0.5B"
        try:
            result = run_parquet_resume_smoke(args.data_list, args.adapter, args.state, model_dir, args.upstream_root, args.output)
        except (ImportError, OSError, RuntimeError, ValueError, FloatingPointError, KeyError) as exc:
            print(f"[ERROR] {exc}")
            return 1
        print(f"[OK] fresh resume: previous_step={result['resume_step']}, resumed_loss={result['loss']:.6f}")
        print(f"[OK] resumed adapter checkpoint: {result['checkpoint']}")
        return 0
    if args.command == "validate-data":
        records, errors = validate_dataset(args.dataset)
        print(render_validation(records, errors))
        return 0 if not errors and all(not record.errors for record in records) else 1
    if args.command == "prepare":
        output = args.output or f"{args.dataset.rstrip('/')}_prepared"
        try:
            manifest = prepare_dataset(args.dataset, output, seed=args.seed, dev_ratio=args.dev_ratio)
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"[ERROR] {exc}")
            return 1
        print(f"[OK] prepared dataset: {output}")
        print(f"[OK] train items: {len(manifest['splits']['train'])}")
        print(f"[OK] dev items: {len(manifest['splits']['dev'])}")
        return 0
    if args.command == "features":
        model_dir = args.model_dir or "/Users/jawoongku/Models/Fun-CosyVoice3-0.5B"
        errors = inspect_feature_inputs(args.dataset, model_dir)
        if errors:
            for error in errors:
                print(f"[ERROR] {error}")
            return 1
        speech_model = args.speech_token_model or f"{model_dir}/speech_tokenizer_v3.onnx"
        try:
            model_root = str(model_dir)
            embedding_model = f"{model_root}/campplus.onnx"
            for split in ("train", "dev"):
                split_dir = f"{args.dataset.rstrip('/')}/{split}"
                extract_embeddings(split_dir, embedding_model)
                extract_speech_tokens(split_dir, speech_model, args.onnx_provider)
                feature_errors = validate_feature_artifacts(split_dir)
                if feature_errors:
                    raise ValueError("feature validation failed: " + "; ".join(feature_errors))
                if not args.skip_parquet:
                    build_parquet(split_dir, args.upstream_root)
                print(f"[OK] features: {split}")
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            print(f"[ERROR] {exc}")
            return 1
        return 0
    if args.command == "package":
        base_model = args.base_model or "/Users/jawoongku/Models/Fun-CosyVoice3-0.5B"
        try:
            destination = build_voice_package(
                args.run,
                args.name,
                args.output,
                base_model=base_model,
                upstream_root=args.upstream_root,
                speaker_id=args.speaker_id,
                language=args.language,
                sample_rate=args.sample_rate,
            )
        except (OSError, ValueError) as exc:
            print(f"[ERROR] {exc}")
            return 1
        print(f"[OK] Voice Package: {destination}")
        return 0
    if args.command == "train":
        try:
            config = load_config(args.config)
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"[ERROR] {exc}")
            return 1
        errors = validate_training_config(config)
        if errors:
            for error in errors:
                print(f"[ERROR] config: {error}")
            return 1
        model_dir = config.get("model_dir") or config.get("model", {}).get("dir")
        dataset_dir = config.get("dataset_dir")
        if not Path(model_dir).is_dir():
            print(f"[BLOCKED] model directory not found: {model_dir}")
            return 2
        if dataset_dir and not Path(dataset_dir).is_dir():
            print(f"[BLOCKED] prepared dataset not found: {dataset_dir}")
            return 2
        if dataset_dir:
            for split in ("train", "dev"):
                data_list = Path(dataset_dir) / split / "parquet" / "data.list"
                data_errors = validate_data_list(data_list, require_features=True)
                if data_errors:
                    for error in data_errors:
                        print(f"[BLOCKED] {split}: {error}")
                    return 2
        if args.resume and not Path(args.resume).exists():
            print(f"[ERROR] resume checkpoint not found: {args.resume}")
            return 1
        print("[BLOCKED] CosyVoice3 model/LoRA loading is available, but the model-specific training batch adapter is not connected yet")
        print("[INFO] refusing to claim smoke-training success without a real batch forward, backward, and optimizer.step")
        return 2
    if args.command == "baseline":
        model_dir = args.model_dir or "/Users/jawoongku/Models/Fun-CosyVoice3-0.5B"
        try:
            output = run_baseline(model_dir, args.text, args.output, upstream_root=args.upstream_root, speaker_id=args.speaker_id)
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            print(f"[ERROR] {exc}")
            return 1
        print(f"[OK] generated WAV: {output}")
        return 0
    if args.command == "clone":
        model_dir = args.model_dir or "/Users/jawoongku/Models/Fun-CosyVoice3-0.5B"
        try:
            output = run_zero_shot(
                model_dir,
                args.reference,
                args.reference_text,
                args.text,
                args.output,
                upstream_root=args.upstream_root,
            )
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            print(f"[ERROR] {exc}")
            return 1
        print(f"[OK] generated WAV: {output}")
        return 0
    if args.command == "synth":
        voice, errors = validate_voice_package(args.voice)
        if errors:
            for error in errors:
                print(f"[ERROR] {error}")
            return 1
        model_dir = args.model_dir or "/Users/jawoongku/Models/Fun-CosyVoice3-0.5B"
        try:
            output = run_synth(args.voice, args.text, args.output, model_dir=model_dir)
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            print(f"[BLOCKED] {exc}")
            return 2
        print(f"[OK] generated WAV: {output}")
        return 0
    if args.command == "compare":
        model_dir = args.model_dir or "/Users/jawoongku/Models/Fun-CosyVoice3-0.5B"
        try:
            report = run_comparison(args.voice, args.text, args.output_dir, model_dir=model_dir, upstream_root=args.upstream_root)
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            print(f"[BLOCKED] {exc}")
            return 2
        print(f"[OK] comparison report: {report}")
        return 0
    if args.command == "narrate":
        model_dir = args.model_dir or "/Users/jawoongku/Models/Fun-CosyVoice3-0.5B"
        try:
            output = run_narrate(args.voice, args.input, args.output, model_dir=model_dir, upstream_root=args.upstream_root, max_chars=args.max_chars)
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            print(f"[BLOCKED] {exc}")
            return 2
        print(f"[OK] narration WAV: {output}")
        return 0
    if args.command == "validate-parquet":
        errors = validate_data_list(args.data_list, require_features=args.require_features)
        if errors:
            for error in errors:
                print(f"[ERROR] {error}")
            return 1
        print(f"[OK] parquet data.list: {args.data_list}")
        return 0
    if args.command == "mps-smoke":
        try:
            result = run_mps_smoke()
        except (ImportError, RuntimeError, FloatingPointError) as exc:
            print(f"[ERROR] {exc}")
            return 1
        print(f"[OK] MPS smoke: device={result['device']} loss={result['loss']:.6f}")
        return 0
    raise AssertionError(f"unhandled command: {args.command}")
