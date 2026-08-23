"""Optional Gradio prototype for guided recording and quality review."""

from __future__ import annotations

import wave
from pathlib import Path

from .dataset import _pcm_stats

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


def build_demo():
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("Gradio UI를 사용하려면 `pip install gradio`가 필요합니다.") from exc
    with gr.Blocks(title="Personal Voice Studio") as demo:
        gr.Markdown("# Personal Voice Studio\n문장을 읽고 녹음한 뒤 학습 가능 여부를 확인합니다.")
        sentence = gr.Dropdown(RECOMMENDED_SENTENCES, value=RECOMMENDED_SENTENCES[0], label="권장 문장")
        audio = gr.Audio(sources=["microphone", "upload"], type="filepath", label="음성 녹음")
        transcript = gr.Textbox(label="Transcript", value=RECOMMENDED_SENTENCES[0])
        inspect = gr.Button("품질 검사")
        report = gr.Textbox(label="검사 결과", lines=5)
        sentence.change(lambda value: value, inputs=sentence, outputs=transcript)
        inspect.click(inspect_recording, inputs=[audio, transcript], outputs=report)
    return demo


def launch_ui() -> None:
    build_demo().launch()
