# 문서 인덱스

이 문서들은 개인 음성 학습·TTS macOS 앱의 실개발 기준이다.

## 먼저 읽을 문서

1. [AGENT.md](AGENT.md) — 작업 규칙과 품질 기준
2. [PRODUCT_SPEC.md](docs/PRODUCT_SPEC.md) — 제품 범위와 사용자 흐름
3. [ARCHITECTURE_SPEC.md](docs/ARCHITECTURE_SPEC.md) — SwiftUI/Python 경계
4. [UI_SPEC.md](docs/UI_SPEC.md) — 화면·상태·상호작용
5. [IMPLEMENTATION_ROADMAP.md](docs/IMPLEMENTATION_ROADMAP.md) — 구현 순서
6. [DEFINITION_OF_DONE.md](docs/DEFINITION_OF_DONE.md) — 완료 조건

## 세부 명세

- [DATA_AND_AUDIO_SPEC.md](docs/DATA_AND_AUDIO_SPEC.md) — 녹음, transcript, 품질 판정
- [TRAINING_AND_TTS_SPEC.md](docs/TRAINING_AND_TTS_SPEC.md) — 학습, checkpoint, TTS
- [COSYVOICE3_MAC_VOICE_TRAINING_PLAN.md](docs/COSYVOICE3_MAC_VOICE_TRAINING_PLAN.md) — 기존 기술 작업계획
- [COSYVOICE3_MAC_VOICE_TRAINING_CODEX_PROMPT.md](docs/COSYVOICE3_MAC_VOICE_TRAINING_CODEX_PROMPT.md) — 원본 지시서
- [DECISIONS.md](DECISIONS.md) — 기존 기술 결정 기록
- [STATUS.md](STATUS.md) — 현재 구현·검증·차단 상태

## 문서 사용 규칙

기능을 구현할 때는 먼저 해당 명세의 체크박스를 갱신하고, 구현 후 실제 검증 결과를 기록한다. 명세와 현재 코드가 다르면 `STATUS.md`에 차이를 남긴다.
