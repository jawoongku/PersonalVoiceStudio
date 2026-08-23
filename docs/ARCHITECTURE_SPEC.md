# 아키텍처 명세서

## 목표 구조

```text
SwiftUI macOS App
  ├─ 녹음/재생/파일 권한
  ├─ 문장 큐와 품질 결과
  ├─ 학습·TTS 상태 표시
  └─ Python 로컬 서비스 호출
          ↓
Python Engine
  ├─ dataset / transcript / quality gate
  ├─ CosyVoice3 feature pipeline
  ├─ LoRA trainer / checkpoint / metrics
  ├─ Voice Package
  └─ synth / compare / narrate
          ↓
Local Filesystem
```

## 단계별 구현

### Prototype

Python CLI와 Gradio로 전체 흐름을 검증한다.

### Product

SwiftUI가 Python 엔진을 subprocess 또는 localhost API로 호출한다. 긴 작업은 job ID를 반환하고 상태를 polling/streaming한다.

## 프로세스 원칙

- UI 메인 스레드에서 학습·추론을 실행하지 않는다.
- Python 작업은 취소 가능한 job으로 관리한다.
- 모든 job은 상태(`queued`, `running`, `completed`, `failed`, `cancelled`)와 로그를 가진다.
- 파일 경로 대신 프로젝트 ID와 artifact ID를 UI API에 노출한다.

## 권장 경계

- SwiftUI: AVAudioEngine, 권한, 화면 상태, 사용자 입력
- Python: WAV 분석, ASR/feature, 학습, checkpoint, TTS
- 공통 계약: JSON metadata + WAV/JSONL/PNG artifact
