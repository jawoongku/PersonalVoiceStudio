# CosyVoice3 Mac Voice Training

[![CI](https://github.com/jawoongku/PersonalVoiceStudio/actions/workflows/ci.yml/badge.svg)](https://github.com/jawoongku/PersonalVoiceStudio/actions/workflows/ci.yml)

Apple Silicon Mac에서 CosyVoice3 기반 음성 데이터 준비, LoRA 학습, Voice Package 생성을 수행하기 위한 CLI 프로젝트입니다.

현재 구현은 upstream CosyVoice를 수정하지 않고 `/Users/jawoongku/CosyVoice`를 wrapper로 사용합니다. 기본 모델 경로는 `COSYVOICE_MODEL_DIR` 또는 `/Users/jawoongku/Models/Fun-CosyVoice3-0.5B`입니다.

로컬 전체 검증은 다음 명령으로 실행합니다.

```bash
scripts/verify_all.sh
```

동일한 작업은 `make verify`, `make app`, `make ui`로도 실행할 수 있습니다.

## 현재 상태

구현 및 단위 검증 완료:

- 환경 진단
- WAV/transcript 검증 및 24 kHz mono 전처리
- train/dev manifest 생성
- ONNX provider 선택(CoreML 검증 후 CPU fallback)
- speaker embedding/speech-token/parquet wrapper
- generic LoRA 주입/동결/gradient 검사
- single-device trainer primitives
- adapter checkpoint 저장/재로딩
- Voice Package 생성/검증
- baseline/clone/synth/compare/narrate CLI 진입점

핵심 모델 자산과 CPU zero-shot clone, 초기 adapter를 사용한 Voice Package/synth/compare/narrate 경로까지 검증했습니다. MPS inference/training과 trained adapter 품질은 아직 `NOT TESTED`/`BLOCKED` 상태입니다.

## Gradio UI prototype

`requirements-ui.txt`를 설치한 환경에서는 다음 명령으로 문장 추천·마이크 녹음·transcript·WAV 품질 검사를 시작할 수 있습니다.

```bash
python -m pip install -r requirements-ui.txt
python -m mac_voice ui
```

현재 UI는 녹음 품질 prototype이며, ASR 발음 일치 판정과 SwiftUI macOS 앱은 후속 단계입니다.
품질 검사를 통과한 녹음은 UI에서 `data/my_voice/raw/`와 `transcripts.csv`에 자동 등록할 수 있습니다.
같은 화면의 Voice Package TTS 영역에서 패키지를 선택해 음성 생성을 요청할 수 있습니다. 모델 경로와 CosyVoice 런타임이 준비되지 않은 경우 원인이 결과 창에 표시됩니다.
성공한 생성은 `artifacts/tts_history.jsonl`에 기록되며 UI에서 최근 기록을 새로고침할 수 있습니다.
ASR 자동 인식이 필요하면 별도로 `python -m pip install -r requirements-asr.txt`를 설치하세요. Whisper 모델은 첫 실행 시 다운로드됩니다.

새 작업 폴더는 다음 명령으로 초기화할 수 있습니다.

```bash
python -m mac_voice init-project --root ~/PersonalVoiceProject
```

학습 작업 상태 파일은 다음처럼 조회합니다.

```bash
python -m mac_voice job-status --job artifacts/runs/my_voice/job.json
```

새 작업 상태 파일을 만들려면:

```bash
python -m mac_voice job-create --output artifacts/runs/my_voice --config configs/my_voice.yaml
```

작업 실행 중 상태를 갱신하려면:

```bash
python -m mac_voice job-update --job artifacts/runs/my_voice/job.json --status running --step 2
```

자동화나 UI 브리지에서는 `--json` 옵션으로 상태를 읽을 수 있습니다.

실제 prepared parquet 기반 CPU smoke 학습을 작업 상태와 함께 실행하려면:

```bash
python -m mac_voice parquet-train-job \
  --train-data-list data/my_voice_prepared/train/parquet/data.list \
  --dev-data-list data/my_voice_prepared/dev/parquet/data.list \
  --model-dir "$COSYVOICE_MODEL_DIR" \
  --output artifacts/parquet_job_adapter.pt \
  --job artifacts/runs/parquet_job/job.json
```

중단 후 저장된 adapter와 optimizer state를 이어서 검증하려면:

```bash
python -m mac_voice parquet-resume-job \
  --data-list data/my_voice_prepared/train/parquet/data.list \
  --adapter artifacts/runs/parquet_job/adapter.pt \
  --state artifacts/runs/parquet_job/adapter.state.pt \
  --model-dir "$COSYVOICE_MODEL_DIR" \
  --output artifacts/runs/parquet_resume_job/adapter.pt \
  --job artifacts/runs/parquet_resume_job/job.json
```

저장된 Voice Package를 점검합니다.

```bash
python -m mac_voice list-voices --root artifacts/voices
```
자동화에서는 `python -m mac_voice list-voices --root artifacts/voices --json`을 사용할 수 있습니다.
SwiftUI 브리지는 `python -m mac_voice bridge-status --job <job.json> --voices artifacts/voices`로 통합 상태를 읽을 수 있습니다.

학습 run에서 Voice Package를 만들려면:

```bash
python -m mac_voice package-job \
  --run artifacts/runs/my_voice \
  --name my_voice \
  --output artifacts/voices/my_voice \
  --base-model "$COSYVOICE_MODEL_DIR" \
  --job artifacts/runs/package_job/job.json
```

## 환경 진단

```bash
python -m mac_voice doctor \
  --model-dir "$COSYVOICE_MODEL_DIR" \
  --upstream-root /Users/jawoongku/CosyVoice

python -m mac_voice mps-smoke

python -m mac_voice inspect-model \
  --model-dir "$COSYVOICE_MODEL_DIR" \
  --upstream-root /Users/jawoongku/CosyVoice

python -m mac_voice model-forward-smoke \
  --model-dir "$COSYVOICE_MODEL_DIR" \
  --upstream-root /Users/jawoongku/CosyVoice

python -m mac_voice model-backward-smoke \
  --model-dir "$COSYVOICE_MODEL_DIR" \
  --upstream-root /Users/jawoongku/CosyVoice

python -m mac_voice parquet-backward-smoke \
  --data-list data/my_voice_prepared/train/parquet/data.list \
  --model-dir "$COSYVOICE_MODEL_DIR" \
  --upstream-root /Users/jawoongku/CosyVoice

python -m mac_voice parquet-train-smoke \
  --train-data-list data/my_voice_prepared/train/parquet/data.list \
  --dev-data-list data/my_voice_prepared/dev/parquet/data.list \
  --output artifacts/parquet_smoke_adapter.pt \
  --model-dir "$COSYVOICE_MODEL_DIR" \
  --upstream-root /Users/jawoongku/CosyVoice

python -m mac_voice parquet-resume-smoke \
  --data-list data/my_voice_prepared/train/parquet/data.list \
  --adapter artifacts/parquet_smoke_adapter.pt \
  --state artifacts/parquet_smoke_adapter.state.pt \
  --output artifacts/parquet_smoke_resumed.pt \
  --model-dir "$COSYVOICE_MODEL_DIR" \
  --upstream-root /Users/jawoongku/CosyVoice
```

## 음성 데이터 준비

```text
data/my_voice/raw/0001.wav
data/my_voice/raw/0002.wav
data/my_voice/transcripts.csv
```

`transcripts.csv` 형식:

```csv
filename,text
0001.wav,안녕하세요. 첫 번째 문장입니다.
0002.wav,오늘은 음성 학습 데이터를 준비합니다.
```

```bash
python -m mac_voice validate-data --dataset data/my_voice
python -m mac_voice prepare --dataset data/my_voice --output data/my_voice_prepared
```

## Feature 추출

모델 directory와 prepared dataset이 준비된 뒤 실행합니다.

```bash
python -m mac_voice features \
  --dataset data/my_voice_prepared \
  --model-dir "$COSYVOICE_MODEL_DIR" \
  --onnx-provider auto

python -m mac_voice validate-parquet \
  --data-list data/my_voice_prepared/train/parquet/data.list \
  --require-features
```

## Baseline/clone

```bash
python -m mac_voice baseline \
  --model-dir "$COSYVOICE_MODEL_DIR" \
  --text "안녕하세요." \
  --output artifacts/baseline.wav

python -m mac_voice clone \
  --model-dir "$COSYVOICE_MODEL_DIR" \
  --reference samples/reference.wav \
  --reference-text "참조 음성 문장입니다." \
  --text "새롭게 생성할 문장입니다." \
  --output artifacts/zero_shot.wav
```

## Training

```bash
python -m mac_voice train \
  --config configs/my_voice.yaml \
  --max-steps 1
```

실제 training 완료로 판정하려면 real batch forward, finite loss, backward, non-zero LoRA gradient, frozen-weight no-gradient, optimizer step을 모두 확인해야 합니다.

## Voice Package

학습 run directory에 adapter와 reference 파일이 준비된 뒤:

```bash
python -m mac_voice package \
  --run artifacts/runs/my_voice \
  --name my_voice \
  --output artifacts/voices/my_voice

python -m mac_voice synth \
  --voice artifacts/voices/my_voice \
  --text "제 목소리로 생성하는 문장입니다." \
  --output output/test.wav
```

## 테스트

```bash
conda run -n cosyvoice python -m unittest discover -s tests -v
```

실제 사용자 음성이 없거나 모델이 없는 상태에서는 해당 integration test를 성공으로 표시하지 않습니다.
