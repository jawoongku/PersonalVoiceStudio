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
- [x] WAV 품질 gate (길이·음량·clipping·무음; ASR 일치도는 후속)
- [x] Gradio recording 화면 골격 및 runtime 생성 검증 (Gradio 3.43.2)

## Phase 2 — 학습 MVP

- [x] Python job 상태 파일 및 조회 CLI 기반 (`job.json`, `job-status`)
- [x] Voice Package 목록·유효성 검증 CLI (`list-voices`)
- [ ] 실제 사용자 dataset 학습 UI
- [ ] loss/로그/취소/resume 표시
- [ ] Voice Package 목록과 검증

## Phase 3 — TTS MVP

- [x] Voice 선택 (Gradio prototype)
- [x] 텍스트 입력 (Gradio prototype)
- [x] 기존 adapter 합성 엔진 연결 (환경·모델 준비 시 실행)
- [x] 생성 WAV UI 재생 출력
- [x] TTS 생성 히스토리 기록·조회 (`artifacts/tts_history.jsonl`)
- [ ] 오디오 재생·저장·히스토리

## Phase 4 — macOS 제품화

- [ ] SwiftUI shell
- [ ] AVAudioEngine 녹음
- [ ] Python engine subprocess/API bridge
- [ ] microphone/filesystem 권한
- [ ] `.app` 패키징

## Phase 5 — 품질 개선

- [ ] 적응형 문장 추천
- [ ] ASR transcript 비교
- [ ] 화자 유사도 평가
- [ ] MPS 호환 runtime
- [ ] 장시간 학습 및 모델 관리
