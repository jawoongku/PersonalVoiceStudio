# CosyVoice3 Mac 음성 파인튜닝 작업계획

## 문서 목적

- [x] 첨부된 `COSYVOICE3_MAC_VOICE_TRAINING_CODEX_PROMPT.md`를 실행 가능한 작업 목록으로 변환한다.
- [x] 각 항목을 실제 검증 결과에 따라 체크한다.
- [x] 검증하지 않은 항목은 완료로 표시하지 않는다.

## 지시사항과 이번 요청의 구분

### 사용자의 실제 요청

- [x] 현재 작업 폴더에 `docs` 폴더를 만든다.
- [x] 첨부 문서를 분석한다.
- [x] 분석 결과를 실제 작업 순서와 체크박스가 포함된 Markdown 문서로 저장한다.

### 첨부 문서에서 도출한 프로젝트 요구사항

- Apple Silicon Mac(M3 Pro 대상)의 PyTorch MPS에서 실행한다.
- NVIDIA/CUDA/DDP/DeepSpeed/원격 GPU/API를 사용하지 않는다.
- `Fun-CosyVoice3-0.5B`를 base model로 사용하고, base weight는 동결한다.
- 1차 학습 대상은 CosyVoice3 LLM stage의 LoRA다.
- 실제 음성 입력부터 전처리, feature 추출, MPS backward/optimizer step, adapter 재로딩, 한국어 WAV 생성까지 로컬에서 연결한다.
- upstream 코드는 보존하고 wrapper/adapter/patch 계층을 우선한다.
- GUI와 성능 최적화는 핵심 CLI 및 end-to-end 검증 이후로 미룬다.

## 완료 판정 규칙

- [ ] MPS inference만 성공한 상태를 학습 완료로 보고하지 않는다.
- [ ] model forward만 성공한 상태를 training 완료로 보고하지 않는다.
- [ ] 실제 `backward`와 `optimizer.step()`을 수행하기 전에는 MPS training을 PASS로 표시하지 않는다.
- [ ] adapter 저장 후 재로딩과 WAV 생성까지 확인하기 전에는 Voice Package 완료로 보고하지 않는다.
- [ ] 실제 사용자 음성이 없으면 음성 모델 완성으로 보고하지 않고, 해당 테스트를 `NOT TESTED` 또는 `SKIPPED`로 남긴다.

## Phase 0 — 프로젝트 및 환경 조사

- [x] 현재 저장소 구조와 기존 변경사항을 확인한다.
- [x] `uname -m`으로 arm64를 확인한다.
- [x] macOS 버전을 확인한다.
- [x] Python 경로와 버전을 확인한다.
- [x] 기존 Conda 환경을 확인한다.
- [x] PyTorch 및 torchaudio 버전을 확인한다.
- [x] `torch.backends.mps.is_available()`와 MPS build 여부를 확인한다.
- [x] 기존 CosyVoice checkout/import 가능 여부를 확인한다.
- [x] `/Users/jawoongku/Models/Fun-CosyVoice3-0.5B` 존재 여부를 먼저 확인한다.
- [x] 모델이 있으면 재다운로드하지 않는다. (기존 경로가 없어 핵심 자산을 provision함)
- [x] `llm.pt`, `flow.pt`, `hift.pt`, `speech_tokenizer_v3.onnx`, `campplus.onnx`, config 존재 여부를 확인한다.
- [x] `ffmpeg`와 `sox` 설치 여부를 확인한다.
- [x] 정상 동작하는 기존 환경을 삭제하거나 불필요하게 재설치하지 않는다.

## Phase 1 — 기본 구조와 기록 체계

- [x] 다음 구조를 프로젝트 상황에 맞게 만든다: `mac_voice/`, `scripts/`, `configs/`, `data/`, `artifacts/`, `tests/`, `third_party/`, `patches/`.
- [x] 모델 경로를 CLI `--model-dir`, `COSYVOICE_MODEL_DIR`, 또는 config로 주입한다.
- [x] 모델 경로를 코드에 하드코딩하지 않는다.
- [x] `STATUS.md`를 만들고 완료/진행/차단/최근 검증 명령을 기록한다.
- [x] `DECISIONS.md`를 만들고 MPS, FP32, CPU provider, LoRA, package 설계 결정을 기록한다.
- [x] 원본 음성과 결과물을 git에 넣지 않도록 `.gitignore`를 확인한다.

## Phase 2 — Environment Doctor

- [x] `python -m mac_voice doctor` CLI를 구현한다.
- [x] architecture, macOS, Python, PyTorch, torchaudio를 사람이 읽기 쉬운 형식으로 출력한다.
- [x] MPS built/available 상태를 출력한다.
- [x] ONNX Runtime 버전과 사용 가능한 provider를 출력한다.
- [x] 모델 경로와 필수 checkpoint/config 존재 여부를 출력한다.
- [x] 가능한 경우 MPS allocated memory를 출력한다. (현재 unavailable이라 값 없음)
- [x] 문제 발생 시 traceback 대신 원인과 조치가 이해되는 오류를 출력한다.

## Phase 3 — Base MPS inference

- [x] `python -m mac_voice baseline --model-dir ... --text ... --output ...`를 구현한다. (upstream adapter 연결)
- [ ] 실제 모델이 MPS에서 로드되는지 확인한다.
- [x] 한국어 텍스트에서 WAV가 생성되는지 확인한다. (CPU zero-shot clone, temporary fixture; MPS/base SFT는 별도 미검증)
- [x] 출력 WAV의 duration이 0보다 큰지 확인한다. (CPU clone/synth)
- [x] 출력 RMS가 무음에 가깝지 않은지 확인한다. (CPU clone/synth)
- [x] sample rate와 NaN/Inf 여부를 확인한다. (CPU clone/synth)
- [x] `clone` 명령과 `<|endofprompt|>` 자동 처리를 구현한다. (실제 CPU 모델 실행)

## Phase 4 — 음성 데이터 입력/검증/전처리

- [x] `data/my_voice/raw/*.wav`와 `transcripts.csv(filename,text)` 입력 규격을 구현한다.
- [x] `python -m mac_voice validate-data --dataset ...`를 구현한다.
- [x] 파일 존재, decode, duration, sample rate, channel, peak, RMS, clipping, silence 비율을 검사한다.
- [x] transcript 누락/빈 문자열/중복 filename을 검사한다.
- [x] 30초 초과 segment를 오류 또는 분할 필요 상태로 표시한다.
- [x] 기본 권장 길이(약 3~15초)는 안내 기준으로만 사용하고 hard failure로 만들지 않는다.
- [x] `python -m mac_voice prepare --dataset ... --output ...`를 구현한다.
- [x] mono, 24 kHz, PCM WAV, 파일명 정규화, 과도한 앞뒤 무음 처리를 수행한다.
- [x] 원본을 수정하지 않고 `raw → prepared` 별도 폴더에 저장한다.
- [x] 재현 가능한 seed 기반 train/dev split과 manifest를 생성한다.

## Phase 5 — CosyVoice feature pipeline

- [x] train/dev 각각에 `wav.scp`, `text`, `utt2spk`, `spk2utt`, `instruct`를 생성한다.
- [x] single speaker 기본값을 `owner`로 사용한다.
- [x] CosyVoice3 기본 instruction 형식을 적용한다.
- [x] speaker embedding을 로컬에서 추출하는 wrapper를 구현한다.
- [x] `utt2embedding.pt`, `spk2embedding.pt`를 생성하고 utterance 누락/빈 값/NaN/Inf를 검사한다. (임시 fixture 실검증)
- [x] speech-token extractor의 CUDA 고정을 제거하거나 wrapper로 provider abstraction을 제공한다.
- [x] `auto → 실제 CoreML 1-sample 검증 → 실패 시 CPU` fallback을 구현한다.
- [x] CUDA provider를 선택하지 않도록 보장한다.
- [x] 모든 segment에 speech token이 생성됐는지 검증한다. (선택한 사용자 음성 10개 실검증)
- [x] 공식 pipeline 형식의 parquet와 `data.list`를 생성하고 feature column을 검증한다.
- [x] 작은 dataset에서 process 수를 1로 제한하고, upstream worker가 실패해도 로컬 fallback으로 검증한다.

## Phase 6 — LoRA 주입 및 동결 검증

- [x] 실제 CosyVoice3/Qwen 모델 구조를 introspect해 LoRA target module을 찾는다. (CPU 모델 로드 검증)
- [x] `q_proj`, `k_proj`, `v_proj`, `o_proj`를 존재 확인 없이 가정하지 않는다.
- [x] target module 매칭 개수와 이름 일부를 확인한다. (각 24개, 총 170 Linear 중)
- [x] 매칭이 0개이면 즉시 실패한다.
- [x] rank, alpha, dropout을 config/checkpoint metadata로 관리한다.
- [x] base parameter를 freeze하고 LoRA만 trainable로 설정한다. (generic helper 및 toy model 검증)
- [x] total/frozen/trainable parameter와 trainable ratio를 출력한다. (generic helper 검증)
- [x] trainable parameter가 0보다 크고 total보다 작은지 검사한다.
- [ ] 의도하지 않은 trainable parameter가 있으면 경고 또는 실패 처리한다.

## Phase 7 — Single-device MPS trainer

- [x] 별도 single-process/single-device MPS trainer를 구현한다. (model-specific adapter 연결 전 generic trainer)
- [x] 기본 correctness 목표를 FP32로 설정한다.
- [x] batch size 1, gradient accumulation, 작은 worker 수를 config로 제공한다. (trainer config 기반)
- [x] optimizer에는 `requires_grad=True`인 parameter만 전달한다.
- [x] gradient clipping과 finite loss 검사를 구현한다.
- [ ] 필요 시 gradient checkpointing을 옵션으로 제공한다.
- [ ] training hot path에서 CUDA AMP, DDP, torchrun, NCCL, DeepSpeed, CUDA_VISIBLE_DEVICES 의존성을 제거한다.
- [ ] `PYTORCH_ENABLE_MPS_FALLBACK`을 무조건 기본 활성화하지 않는다.
- [ ] unsupported op 발생 시 op/stack trace와 fallback 이유를 기록한다.
- [ ] float64가 필요한 작은 연산만 CPU에서 처리하고 MPS tensor를 float64로 cast하지 않는다.
- [ ] MPS memory watermark를 위험하게 무제한 해제하지 않는다.

## Phase 8 — Smoke train 및 gradient 검증

- [x] `python -m mac_voice train --config ... --max-steps 1` 진입점과 설정 검증을 구현한다. (실제 CosyVoice model loader 연결 전)
- [ ] smoke train이 base load → LoRA inject → freeze → 실제 batch → forward까지 수행한다.
- 참고: `model-forward-smoke`로 실제 CosyVoice3 CPU LLM의 synthetic token batch forward/loss는 검증했지만, 사용자 parquet batch와 backward/optimizer를 포함한 smoke train은 아직 완료하지 않았다.
- 참고: `model-backward-smoke`로 synthetic CPU batch의 backward, LoRA gradient, frozen-base gradient, optimizer step까지 검증했지만 MPS 및 사용자 parquet batch 학습은 아직 완료하지 않았다.
- 참고: `parquet-backward-smoke`로 실제 feature-bearing parquet row 1건의 tokenizer→CosyVoice3 LLM→backward→optimizer step을 CPU에서 검증했다. 다중 step, dev validation, MPS는 아직 미검증이다.
- 참고: `parquet-train-smoke`로 실제 train/dev parquet row를 사용한 CPU 2-step train, validation, adapter checkpoint 저장을 검증했다. 전체 dataset, resume/metrics 통합, MPS는 아직 미검증이다.
- 참고: 동일 fixture로 3-step 반복도 통과해 다중 step 안정성을 확인했다. 이는 사용자 전체 dataset과 MPS 학습 완료를 의미하지 않는다.
- 참고: parquet train smoke가 optimizer state와 `metrics.jsonl`을 함께 저장하도록 연결했고, 실제 fixture에서 생성물을 확인했다. fresh-process resume와 전체 trainer 통합은 별도 단계다.
- 참고: `parquet-resume-smoke`로 새 프로세스에서 adapter/optimizer state를 재로딩하고 추가 CPU step을 수행했다. 전체 trainer와 MPS resume는 아직 미검증이다.
- [ ] finite loss를 확인한다.
- [ ] backward까지 수행한다.
- [ ] LoRA gradient가 존재하고 finite이며 non-zero인지 확인한다.
- [ ] frozen base parameter에 gradient가 없는지 확인한다.
- [ ] optimizer step이 실제로 완료되는지 확인한다.
- [x] adapter checkpoint를 저장한다. (실제 CosyVoice3 CPU LoRA 주입 후 adapter-only 저장/재로딩 검증; 학습 step은 미수행)

## Phase 9 — Checkpoint, validation, resume

- [x] adapter-only checkpoint 저장을 구현한다. (`adapter_latest.pt`/`adapter_best.pt` orchestration은 후속)
- [x] base model 전체 state dict를 매번 저장하지 않는다. (adapter-only payload 실검증)
- [x] checkpoint에 LoRA weights, step, epoch, validation loss, config snapshot을 저장한다.
- [x] optimizer/scheduler state를 resume용으로 저장하고 복원하는 helper를 구현한다.
- [x] generic loop의 `resume_from`에서 step/optimizer state를 복원한다.
- [x] 실제 CosyVoice3 parquet CPU smoke에서 optimizer state를 새 프로세스로 재로딩하고 추가 step을 수행한다. (MPS/전체 trainer 연결 전)
- [x] validation을 N step마다 수행하는 generic training loop를 구현한다. (CosyVoice batch adapter 연결 전)
- [x] train/dev loss를 `metrics.jsonl`에 기록하는 logger를 구현하고 parquet CPU smoke에서 검증한다. (full trainer loop 연결 전)
- [x] best/latest checkpoint 저장을 generic loop에 구현한다. (patience 기반 early stopping은 후속)

## Phase 10 — Voice Package와 LoRA inference

- [x] `python -m mac_voice package --run ... --name ... --output ...`를 구현한다.
- [x] package에 adapter, `voice.json`, config, metrics, reference WAV/text, speaker embedding, provenance를 포함한다. (존재하는 선택 파일은 복사)
- [x] `voice.json`에 base model, adapter, speaker id, language, sample rate를 기록한다.
- [x] base model 자체를 package 안에 복사하지 않는다.
- [x] `provenance.json`에 upstream commit, runtime versions, model path, LoRA/config metadata, timestamp를 기록한다.
- [x] `python -m mac_voice synth --voice ... --text ... --output ...`와 CosyVoice3 LLM adapter 주입 경로를 구현한다. (실제 CPU 모델 adapter 주입/재로딩 검증; 합성은 별도)
- [x] base load → adapter load → canonical reference → 한국어 text → WAV 생성 흐름을 검증한다. (CPU, user reference/CPU-smoke adapter)
- [x] 생성 WAV가 non-empty/non-silent인지 확인한다. (3.48초, 24 kHz mono PCM; 음질/화자 유사도는 미평가)
- [x] `compare` 명령으로 동일 조건의 before/after WAV와 comparison JSON을 생성한다. (실제 CPU 모델 실행, 임시 adapter)
- [x] 핵심 학습 검증 이후 `narrate` chunk TTS utility를 추가하고 chunk 결합 WAV를 생성한다. (실제 CPU 모델 실행, 임시 adapter)

## Phase 11 — 테스트와 최종 CLI

- [x] device, dataset, manifest, ONNX provider, LoRA injection, freeze, checkpoint, voice package, nested parquet schema unit test를 작성한다.
- [ ] 실제 model load → LoRA → batch → MPS forward → backward → optimizer step → save → reload integration test를 작성한다.
- [x] 실제 음성이 없으면 음성 의존 테스트를 임시 fixture 기반으로 분리하고, 사용자 음성 품질 검증은 완료로 표시하지 않는다.
- [x] 최종 CLI(`doctor`, `inspect-model`, `baseline`, `validate-data`, `prepare`, `features`, `train`, `package`, `synth`, `compare`, `narrate`)를 README에 문서화한다.
- [x] 전체 wrapper는 핵심 CLI가 검증된 후에만 추가한다.
- [x] wrapper는 어느 단계든 실패하면 즉시 중단하고 원인을 출력한다.
- [ ] GUI(Gradio/Electron/Swift/Web dashboard)를 1차 범위에 추가하지 않는다.

## 최종 인수 기준

- [ ] MPS 사용 가능 및 base model 로컬 로드 PASS
- [ ] Mac MPS base inference가 non-empty WAV 생성 PASS
- [ ] WAV + transcript 입력 및 train/dev manifest 생성 PASS
- [ ] speaker embedding 로컬 생성 PASS
- [ ] CUDA 없이 speech token 생성 PASS
- [ ] parquet/data.list 생성 PASS
- [ ] base weight freeze 및 LoRA trainable 검증 PASS
- [ ] 실제 MPS batch forward 및 finite loss PASS
- [ ] 실제 backward 및 non-zero LoRA gradient PASS
- [ ] frozen weight 무-gradient PASS
- [ ] optimizer.step PASS
- [ ] adapter save/reload 및 resume PASS
- [ ] adapter 기반 한국어 WAV 생성 PASS
- [ ] 재사용 가능한 Voice Package 생성 PASS

## 미완료 보고 형식

검증 전 항목은 아래처럼 기록한다.

```text
PASS       실제 검증 완료
NOT TESTED 아직 실행하지 않음
SKIPPED    필요한 실제 음성/환경이 없어 의도적으로 건너뜀
BLOCKED    원인이 기록된 외부 차단 상태
```

최종 보고에는 `Working`, `Verified`, 실행 명령, 남은 이슈, 사용자의 다음 행동을 포함한다.

## 참고

- 원본 지시서: [`COSYVOICE3_MAC_VOICE_TRAINING_CODEX_PROMPT.md`](./COSYVOICE3_MAC_VOICE_TRAINING_CODEX_PROMPT.md)
- 작성일: 2026-08-23
