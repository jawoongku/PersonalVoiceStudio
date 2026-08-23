# 완료 정의

## 기능 완료

- [ ] 정상 흐름과 실패 흐름이 모두 정의되어 있다.
- [ ] 사용자가 다음 조치를 알 수 있는 오류가 표시된다.
- [ ] 원본 음성이 보존된다.
- [ ] 결과 artifact 위치와 metadata가 기록된다.

## 음성 데이터 완료

- [ ] 모든 accepted 파일에 정확한 transcript가 있다.
- [ ] 품질 gate 결과가 저장된다.
- [ ] train/dev split이 재현 가능하다.
- [ ] feature와 parquet schema가 검증된다.

## 학습 완료

- [ ] 실제 batch forward
- [ ] finite loss
- [ ] backward
- [ ] LoRA gradient
- [ ] frozen base gradient 없음
- [ ] optimizer step
- [ ] checkpoint 저장·재로딩
- [ ] validation과 metrics 기록

## TTS 완료

- [ ] Voice Package 검증
- [ ] adapter metadata 기반 주입
- [ ] WAV 생성
- [ ] non-empty/non-silent 검사
- [ ] 샘플레이트·채널 검사
- [ ] 화자 유사도는 별도 평가 결과로 기록

## UI 완료

- [ ] 녹음 권한 처리
- [ ] 학습 중 UI 비차단
- [ ] 취소·재시작
- [ ] 접근 가능한 오류 표시
- [ ] 생성 음성 재생·저장
