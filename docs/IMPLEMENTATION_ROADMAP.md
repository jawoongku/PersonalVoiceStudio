# 실개발 로드맵

## Phase 0 — 문서·기반

- [x] 기존 Python CLI와 CosyVoice3 wrapper 확인
- [x] 데이터/feature/checkpoint/TTS 경로 검증
- [ ] 새 제품 명세를 코드 구조에 반영

## Phase 1 — Python MVP

- [x] 프로젝트 생성 API/CLI (`init-project`)
- [x] 문장 추천 큐 (초기 정적 문장 세트)
- [x] 녹음 파일 등록 및 transcript 입력 (Gradio prototype)
- [x] 품질 검사 통과 녹음의 dataset/raw 복사 및 transcripts.csv 자동 등록
- [x] UI 데이터셋 전체 검증 및 오류 보고
- [x] UI 학습 데이터 준비(`prepare`) 실행 및 결과 보고
- [x] UI 학습 사전 점검(설정·모델·parquet) 및 오류 보고
- [x] UI 학습 작업 상태 조회(`job.json`)
- [x] 학습 작업 메타데이터 생성 CLI (`job-create`)
- [x] 학습 작업 상태 갱신 CLI (`job-update`)
- [x] 실제 parquet CPU smoke 학습과 job 상태 연동 (`parquet-train-job`)
- [x] adapter·optimizer state resume와 job 상태 연동 (`parquet-resume-job`)
- [x] UI 학습 metrics 요약 조회
- [x] 선택적 ASR 문장 입력과 transcript 유사도 비교 기반
- [x] Whisper 기반 선택적 ASR 자동 인식 UI (모델 다운로드·실행 가능 여부는 환경 의존)
- [x] Gradio 3 호환 마이크·파일 업로드 입력
- [x] UI 품질 gate에 24kHz·mono 권장 조건 추가
- [x] 긴 글 TTS UI 연결 및 생성 WAV 재생
- [x] 긴 글 TTS 문장 분할 길이 UI 조절
- [x] WAV 품질 gate (길이·음량·clipping·무음; ASR 일치도는 후속)
- [x] Gradio recording 화면 골격 및 runtime 생성 검증 (Gradio 3.43.2)

## Phase 2 — 학습 MVP

- [x] Python job 상태 파일 및 조회 CLI 기반 (`job.json`, `job-status`)
- [x] Voice Package 목록·유효성 검증 CLI (`list-voices`)
- [x] 학습 run에서 Voice Package 생성과 job 상태 연동 (`package-job`)
- [x] 실제 사용자 dataset CPU 학습 시작 UI (job 기반; MPS는 별도)
- [x] loss/로그 표시
- [x] 학습 취소 상태 표시 UI 및 training loop 협력적 취소 지점
- [x] Voice Package 목록과 검증

## Phase 3 — TTS MVP

- [x] Voice 선택 (Gradio prototype)
- [x] 텍스트 입력 (Gradio prototype)
- [x] 기존 adapter 합성 엔진 연결 (환경·모델 준비 시 실행)
- [x] 실제 검증 Voice Package 한국어 TTS 생성 확인
- [x] zero-shot·adapter 출력 비교 리포트 생성 확인
- [x] 생성 WAV UI 재생 출력
- [x] TTS 생성 히스토리 기록·조회 (`artifacts/tts_history.jsonl`)
- [x] 오디오 재생·저장·히스토리

## Phase 4 — macOS 제품화

- [ ] SwiftUI shell
- [ ] AVAudioEngine 녹음
- [ ] Python engine subprocess/API bridge
- [ ] microphone/filesystem 권한
- [ ] `.app` 패키징

## Phase 5 — 품질 개선

- [x] dataset transcript 기반 적응형 문장 추천
- [x] ASR transcript 비교 기반 및 선택적 Whisper 연결
- [ ] 화자 유사도 평가
- [ ] MPS 호환 runtime
- [ ] 장시간 학습 및 모델 관리
