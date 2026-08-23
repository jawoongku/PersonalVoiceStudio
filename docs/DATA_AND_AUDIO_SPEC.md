# 데이터·오디오 명세서

## 프로젝트 구조

```text
VoiceProject/
├─ raw/
├─ transcripts.csv
├─ prepared/
├─ features/
├─ runs/
└─ voices/
```

## 입력 규격

- WAV PCM 권장
- mono/stereo 입력 허용
- sample rate는 자동 변환
- segment 권장 3~15초, 최대 30초
- transcript는 실제 발화와 정확히 일치해야 한다.

## 자동 검사

- decode 가능 여부
- duration
- sample rate/channel
- RMS/peak/clipping
- 무음 비율
- transcript 존재·중복·빈 문자열
- ASR transcript와의 불일치 후보

## 판정 정책

- `accepted`: 학습에 사용
- `review`: 사용 가능하지만 사용자 확인 필요
- `rejected`: 학습에서 제외

자동 판정은 원본을 삭제하지 않는다. rejected 파일도 사용자가 다시 복구할 수 있어야 한다.

## 문장 추천 정책

문장 선택기는 음운·받침·억양·문장 길이 coverage를 추적하고, 이미 확보한 발음 패턴을 반복하지 않는다. 기본 10문장 후 부족한 패턴을 보완하는 적응형 큐를 사용한다.
