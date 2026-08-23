# 학습·TTS 명세서

## 학습

- CosyVoice3 base model은 동결
- LLM stage LoRA 우선
- rank/alpha/dropout은 config와 checkpoint metadata에 저장
- train/dev 분리
- finite loss와 gradient 검사
- latest/best adapter checkpoint
- optimizer/scheduler state와 metrics JSONL
- 취소·resume 지원

## 실행 장치

- Apple Silicon MPS 우선
- MPS unavailable이면 UI에 명시하고 CPU로만 실행
- CUDA/DDP/원격 GPU를 기본 경로로 사용하지 않음

## TTS

1. Voice Package 검증
2. checkpoint metadata에서 LoRA 구조 읽기
3. base model 로드
4. adapter 주입
5. reference 음성·text 적용
6. 텍스트 정규화 및 chunk 분할
7. WAV 생성
8. non-empty/non-silent/sample-rate 검사

## 품질 평가

파일 유효성, 발음 정확도, 화자 유사도를 서로 다른 지표로 기록한다. 학습 loss가 낮아도 화자 유사도가 높다고 단정하지 않는다.
