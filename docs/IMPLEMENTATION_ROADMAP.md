# 실개발 로드맵

## Phase 0 — 문서·기반

- [x] 기존 Python CLI와 CosyVoice3 wrapper 확인
- [x] 데이터/feature/checkpoint/TTS 경로 검증
- [ ] 새 제품 명세를 코드 구조에 반영

## Phase 1 — Python MVP

- [ ] 프로젝트 생성 API/CLI
- [ ] 문장 추천 큐
- [ ] 녹음 파일 등록 및 transcript 저장
- [ ] 품질 gate와 accepted/review/rejected 상태
- [ ] Gradio recording 화면

## Phase 2 — 학습 MVP

- [ ] Python job runner
- [ ] 실제 사용자 dataset 학습 UI
- [ ] loss/로그/취소/resume 표시
- [ ] Voice Package 목록과 검증

## Phase 3 — TTS MVP

- [ ] Voice 선택
- [ ] 텍스트 입력
- [ ] 한국어/영어 TTS
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
