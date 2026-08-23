"""Optional Gradio prototype for guided recording and quality review."""

from __future__ import annotations

import wave
import inspect as pyinspect
import csv
import shutil
from pathlib import Path

from .dataset import _pcm_stats, prepare_dataset, render_validation, validate_dataset
from .catalog import list_voice_packages
from .synth import run_synth
from .history import append_tts_history, read_tts_history
from .config import load_config, validate_training_config
from .parquet import validate_data_list
from .jobs import read_job
from .metrics_view import summarize_metrics

RECOMMENDED_SENTENCES = [
    "오늘 아침에는 평소보다 조금 일찍 일어났습니다.",
    "창문을 열어 보니 바람이 생각보다 시원하게 불고 있습니다.",
    "아직 결정하지 못한 부분은 조금 더 고민해 볼 생각입니다.",
    "중요한 것은 얼마나 빨리 하느냐가 아니라 제대로 하는 것입니다.",
    "그런데 이 방법이 가장 좋은 방법일까요?",
    "그러면 다른 방법을 한번 찾아보는 게 어떨까요?",
]


def inspect_recording(audio_path: str | Path | None, transcript: str) -> str:
    if not audio_path:
        return "녹음 파일을 먼저 선택하거나 녹음해 주세요."
    if not transcript.strip():
        return "transcript를 입력해 주세요."
    path = Path(audio_path)
    if not path.is_file():
        return f"파일을 찾을 수 없습니다: {path}"
    try:
        with wave.open(str(path), "rb") as handle:
            frames, sample_rate, channels, width = handle.getnframes(), handle.getframerate(), handle.getnchannels(), handle.getsampwidth()
            raw = handle.readframes(frames)
        peak, rms, clipping, silence = _pcm_stats(raw, width)
    except (OSError, wave.Error) as exc:
        return f"WAV를 읽을 수 없습니다: {exc}"
    duration = frames / sample_rate if sample_rate else 0.0
    issues = []
    if duration <= 0 or duration > 30:
        issues.append("길이는 0초 초과 30초 이하가 권장됩니다")
    if clipping:
        issues.append("clipping이 감지되었습니다")
    if rms < 0.005:
        issues.append("음량이 너무 작을 수 있습니다")
    if silence > 0.8:
        issues.append("무음 구간이 많습니다")
    status = "사용 가능" if not issues else "재녹음 검토"
    return (f"판정: {status}\n길이: {duration:.2f}초 | {sample_rate}Hz | {channels}ch | "
            f"RMS: {rms:.4f} | peak: {peak:.4f}\n"
            f"문제: {', '.join(issues) if issues else '없음'}\n"
            "문장 발음 일치 여부는 녹음 재생 후 확인해 주세요.")


def register_recording(dataset_root: str | Path, audio_path: str | Path | None, transcript: str) -> str:
    """Copy a quality-approved WAV into a dataset and append its transcript."""
    report = inspect_recording(audio_path, transcript)
    if not report.startswith("판정: 사용 가능"):
        return report + "\n저장하지 않았습니다."
    root = Path(dataset_root).expanduser()
    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    existing = {path.stem for path in raw_dir.glob("*.wav")}
    index = 1
    while f"{index:04d}" in existing:
        index += 1
    filename = f"{index:04d}.wav"
    destination = raw_dir / filename
    shutil.copyfile(Path(audio_path), destination)
    transcript_path = root / "transcripts.csv"
    needs_header = not transcript_path.exists() or transcript_path.stat().st_size == 0
    with transcript_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if needs_header:
            writer.writerow(["filename", "text"])
        writer.writerow([filename, transcript.strip()])
    return report + f"\n저장 완료: {destination}"


def synthesize_for_ui(voice_root: str, voice_name: str, text: str, model_dir: str, output: str) -> tuple[str, str | None]:
    if not text.strip():
        return "텍스트를 입력해 주세요.", None
    voice_path = Path(voice_root).expanduser() / voice_name
    try:
        result = run_synth(voice_path, text.strip(), output, model_dir=model_dir)
    except (OSError, RuntimeError, ValueError) as exc:
        return f"합성할 수 없습니다: {exc}", None
    append_tts_history("artifacts/tts_history.jsonl", voice=voice_name, text=text.strip(), output=str(result))
    return f"합성 완료: {result}", str(result)


def validate_dataset_for_ui(dataset_root: str) -> str:
    try:
        records, errors = validate_dataset(dataset_root)
    except (OSError, ValueError) as exc:
        return f"검증할 수 없습니다: {exc}"
    return render_validation(records, errors) or "검증할 파일이 없습니다."


def prepare_dataset_for_ui(dataset_root: str, output_root: str) -> str:
    try:
        manifest = prepare_dataset(dataset_root, output_root)
    except (OSError, RuntimeError, ValueError) as exc:
        return f"학습 데이터 준비 실패: {exc}"
    train_count = len(manifest["splits"]["train"])
    dev_count = len(manifest["splits"]["dev"])
    return f"준비 완료: {output_root}\ntrain={train_count}, dev={dev_count}"


def training_preflight_for_ui(config_path: str) -> str:
    try:
        config = load_config(config_path)
    except (OSError, RuntimeError, ValueError) as exc:
        return f"설정 읽기 실패: {exc}"
    errors = validate_training_config(config)
    lines = [f"설정: {config_path}"]
    if errors:
        lines.extend(f"[ERROR] {error}" for error in errors)
        return "\n".join(lines)
    model_dir = config.get("model_dir") or config.get("model", {}).get("dir")
    dataset_dir = config.get("dataset_dir")
    lines.append(f"모델: {'사용 가능' if model_dir and Path(model_dir).is_dir() else '없음'} ({model_dir})")
    if dataset_dir:
        for split in ("train", "dev"):
            data_list = Path(dataset_dir) / split / "parquet" / "data.list"
            data_errors = validate_data_list(data_list, require_features=True)
            lines.append(f"{split}: {'사용 가능' if not data_errors else '오류'}")
            lines.extend(f"  [ERROR] {error}" for error in data_errors)
    else:
        lines.append("dataset_dir: 설정되지 않음")
    return "\n".join(lines)


def job_status_for_ui(job_path: str) -> str:
    try:
        job = read_job(job_path)
    except (OSError, ValueError) as exc:
        return f"작업 상태를 읽을 수 없습니다: {exc}"
    lines = [f"상태: {job['status']}"]
    for key in ("command", "config", "step", "error", "updated_at"):
        if key in job:
            lines.append(f"{key}: {job[key]}")
    return "\n".join(lines)


def build_demo():
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("Gradio UI를 사용하려면 `pip install gradio`가 필요합니다.") from exc
    with gr.Blocks(title="Personal Voice Studio") as demo:
        gr.Markdown("# Personal Voice Studio\n문장을 읽고 녹음한 뒤 학습 가능 여부를 확인합니다.")
        sentence = gr.Dropdown(RECOMMENDED_SENTENCES, value=RECOMMENDED_SENTENCES[0], label="권장 문장")
        # Gradio 3.x (required by matcha-tts) uses ``source`` while newer
        # releases use ``sources``. Keep the prototype compatible with both.
        audio_kwargs = {"type": "filepath", "label": "음성 녹음"}
        if "sources" in pyinspect.signature(gr.Audio).parameters:
            audio = gr.Audio(sources=["microphone", "upload"], **audio_kwargs)
        else:
            audio = gr.Audio(source="microphone", **audio_kwargs)
            gr.Markdown("Gradio 3.x에서는 마이크 입력을 사용하세요. 업로드 파일은 다음 단계에서 통합합니다.")
        transcript = gr.Textbox(label="Transcript", value=RECOMMENDED_SENTENCES[0])
        inspect = gr.Button("품질 검사")
        report = gr.Textbox(label="검사 결과", lines=5)
        dataset_root = gr.Textbox(label="데이터셋 경로", value="data/my_voice")
        save = gr.Button("검사 통과 파일을 데이터셋에 저장")
        save_report = gr.Textbox(label="저장 결과", lines=6)
        sentence.change(lambda value: value, inputs=sentence, outputs=transcript)
        inspect.click(inspect_recording, inputs=[audio, transcript], outputs=report)
        save.click(register_recording, inputs=[dataset_root, audio, transcript], outputs=save_report)
        dataset_check = gr.Button("데이터셋 전체 검증")
        dataset_report = gr.Textbox(label="데이터셋 검증 결과", lines=8)
        dataset_check.click(validate_dataset_for_ui, inputs=dataset_root, outputs=dataset_report)
        prepared_root = gr.Textbox(label="학습 데이터 출력 경로", value="data/my_voice_prepared")
        prepare = gr.Button("학습 데이터 준비")
        prepare_report = gr.Textbox(label="준비 결과", lines=4)
        prepare.click(prepare_dataset_for_ui, inputs=[dataset_root, prepared_root], outputs=prepare_report)
        gr.Markdown("## 학습 사전 점검")
        training_config = gr.Textbox(label="학습 설정 파일", value="configs/my_voice.yaml")
        preflight = gr.Button("학습 준비 상태 점검")
        preflight_report = gr.Textbox(label="학습 사전 점검 결과", lines=8)
        preflight.click(training_preflight_for_ui, inputs=training_config, outputs=preflight_report)
        gr.Markdown("## 학습 작업 상태")
        job_path = gr.Textbox(label="job.json 경로", value="artifacts/runs/my_voice/job.json")
        job_refresh = gr.Button("작업 상태 새로고침")
        job_report = gr.Textbox(label="작업 상태", lines=6)
        job_refresh.click(job_status_for_ui, inputs=job_path, outputs=job_report)
        gr.Markdown("## 학습 metrics")
        metrics_path = gr.Textbox(label="metrics.jsonl 경로", value="artifacts/runs/my_voice/metrics.jsonl")
        metrics_refresh = gr.Button("metrics 새로고침")
        metrics_report = gr.Textbox(label="최근 학습 metrics", lines=8)
        metrics_refresh.click(summarize_metrics, inputs=metrics_path, outputs=metrics_report)
        gr.Markdown("## Voice Package TTS")
        voice_root = gr.Textbox(label="Voice Package 폴더", value="artifacts/voices")
        voice_choices = [item["path"] for item in list_voice_packages("artifacts/voices")]
        voice_names = [Path(path).name for path in voice_choices]
        voice_name = gr.Dropdown(voice_names, value=voice_names[0] if voice_names else None, label="음성 선택")
        tts_text = gr.Textbox(label="TTS 텍스트")
        model_dir = gr.Textbox(label="CosyVoice 모델 경로", value="/Users/jawoongku/Models/Fun-CosyVoice3-0.5B")
        output = gr.Textbox(label="출력 WAV", value="artifacts/ui_tts.wav")
        synth = gr.Button("음성 생성")
        synth_report = gr.Textbox(label="TTS 결과", lines=3)
        synth_audio = gr.Audio(label="생성된 음성", type="filepath", interactive=False)
        synth.click(synthesize_for_ui, inputs=[voice_root, voice_name, tts_text, model_dir, output], outputs=[synth_report, synth_audio])
        history = gr.Textbox(label="최근 생성 기록", lines=6)
        refresh_history = gr.Button("히스토리 새로고침")
        refresh_history.click(lambda: "\n".join(f"{item.get('created_at', '')} | {item.get('voice', '')} | {item.get('text', '')}" for item in read_tts_history("artifacts/tts_history.jsonl")), outputs=history)
    return demo


def launch_ui() -> None:
    build_demo().launch()
