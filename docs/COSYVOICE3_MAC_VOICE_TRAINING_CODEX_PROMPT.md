# CosyVoice3 0.5B — Apple Silicon 로컬 보이스 파인튜닝 시스템 구축 지시서

> **용도:** 이 문서는 Codex에게 그대로 전달하는 개발 지시 프롬프트다.  
> **목표:** MacBook Pro Apple Silicon 한 대에서 `내 음성 데이터 준비 → CosyVoice3 0.5B LoRA 학습 → 내 목소리 음성 패키지 생성 → 로컬 TTS 생성`까지 수행할 수 있는 시스템을 구축한다.  
> **작성 기준일:** 2026-08-23

---

## 0. 가장 중요한 목표

이 작업의 최종 목표는 단순한 **Zero-shot Voice Cloning 데모**를 만드는 것이 아니다.

반드시 아래 파이프라인을 구축해야 한다.

```text
내 음성 WAV + 정확한 전사문
        ↓
음성 검증 / 정규화 / 데이터셋 구성
        ↓
CosyVoice3 학습용 manifest / feature 생성
        ↓
Fun-CosyVoice3-0.5B
        +
LoRA Adapter
        ↓
Apple Silicon MPS에서 실제 학습
        ↓
내 목소리 Voice Package 생성
        ↓
한국어 텍스트 입력
        ↓
내 목소리 TTS WAV 생성
```

**학습까지 Mac에서 끝내는 것이 핵심 요구사항이다.**

---

# 1. 절대 조건

다음 조건은 변경하지 않는다.

## 1.1 하드웨어

대상 머신:

- Apple Silicon MacBook Pro
- 현재 주 대상: **M3 Pro**
- GPU backend: **PyTorch MPS**
- CPU fallback은 MPS에서 지원하지 않는 특정 연산에 한해 허용

## 1.2 금지 사항

다음을 사용하지 않는다.

- NVIDIA GPU
- CUDA
- cuDNN
- NCCL
- DeepSpeed
- CUDA 전용 ONNX Runtime
- 원격 GPU
- Colab
- RunPod
- AWS/GCP/Azure GPU
- 외부 학습 API
- 외부 TTS API

인터넷은 **소스 코드/패키지 다운로드와 문서 조사**에는 사용할 수 있다.

그러나 데이터 전처리, feature extraction, 학습, inference는 모두 로컬 Mac에서 수행되어야 한다.

## 1.3 모델

기본 모델은 다음을 사용한다.

```text
Fun-CosyVoice3-0.5B
```

현재 로컬에 이미 모델이 존재할 가능성이 높다.

기본 예상 경로:

```text
/Users/jawoongku/Models/Fun-CosyVoice3-0.5B
```

**먼저 실제 경로가 존재하는지 확인하고 사용한다.**

모델이 이미 존재하면 절대 다시 다운로드하지 않는다.

모델 경로는 코드에 하드코딩하지 말고 다음 중 하나로 받는다.

1. CLI `--model-dir`
2. 환경변수 `COSYVOICE_MODEL_DIR`
3. 프로젝트 config

기본값으로 위 경로를 사용할 수 있다.

---

# 2. 이 작업에서 말하는 “내 목소리 모델”의 정의

CosyVoice3 0.5B 전체 파라미터를 다시 학습하는 Full Fine-Tuning을 목표로 하지 않는다.

우선 구현할 방식은:

```text
CosyVoice3 0.5B base model
        │
        ├── base parameters: freeze
        │
        └── LoRA parameters: train
```

즉 **LoRA 기반 Parameter-Efficient Fine-Tuning**을 구현한다.

최종 결과물은 base model을 복제한 거대한 모델 폴더가 아니라 다음과 같은 **Voice Package** 형태가 되어야 한다.

예:

```text
artifacts/voices/my_voice/
├── adapter.pt
├── voice.json
├── training_config.yaml
├── metrics.json
├── reference.wav
├── reference.txt
├── speaker_embedding.pt
└── provenance.json
```

`voice.json`에는 최소한 다음 정보를 기록한다.

```json
{
  "name": "my_voice",
  "base_model": "Fun-CosyVoice3-0.5B",
  "adapter": "adapter.pt",
  "speaker_id": "owner",
  "language": "ko",
  "sample_rate": 24000
}
```

base model 자체는 Voice Package 안에 복사하지 않는다.

---

# 3. 중요한 upstream 현황

작업 시작 전에 반드시 최신 upstream 상태를 직접 확인한다.

현재 조사 기준으로 다음 사실이 있다.

## 3.1 CosyVoice3 공식 학습 recipe

공식 CosyVoice3 예제:

```text
examples/libritts/cosyvoice3/run.sh
```

현재 공식 recipe는 학습 단계에서 다음 CUDA/distributed 구조를 사용한다.

- `CUDA_VISIBLE_DEVICES`
- `torchrun`
- `nccl`
- `torch_ddp`
- optional DeepSpeed
- CUDA AMP

따라서 이 training entrypoint를 그대로 사용하려고 하지 않는다.

Mac용으로 **single-process / single-device MPS trainer**를 별도로 구현하는 것을 우선한다.

공식 recipe에는 현재 다음과 같은 설명도 있다.

```text
We only support llm training for now
```

따라서 1차 구현의 학습 대상은 **LLM stage + LoRA**로 한정한다.

Flow / HiFiGAN 전체 학습을 1차 범위에 넣지 않는다.

---

## 3.2 Apple Silicon MPS upstream PR

공식 저장소 PR:

```text
QwenAudio/CosyVoice #1869
feat: add Apple Silicon (MPS) support for macOS ARM64
```

이 PR은 다음 device abstraction을 제안한다.

- CUDA
- MPS
- CPU

그리고 inference 코드의 CUDA 의존성을 device-aware 코드로 변경한다.

하지만 작성자가 명시적으로:

```text
Training: Out of scope
```

라고 적었다.

즉 이 PR을 그대로 적용하면 **inference는 개선되지만 training은 해결되지 않는다.**

이 PR은 Mac inference 및 device abstraction 구현 참고자료로 사용한다.

특히 MPS는 `float64`를 지원하지 않으므로, precision-sensitive한 일부 HiFiGAN/F0 연산은 CPU float64 fallback이 필요할 수 있다.

---

## 3.3 Native LoRA upstream PR

공식 저장소 PR:

```text
QwenAudio/CosyVoice #1924
Add generic native LoRA training support
```

현재 조사 시점에는 open PR이다.

이 PR은 다음 기능을 제공한다.

- base weights freeze
- native LoRA adapter injection
- trainable parameter만 optimizer에 전달
- adapter-only checkpoint save/load
- LoRA unit tests

이 코드를 적극 참고한다.

단, **열린 PR의 mutable branch에 프로젝트 전체를 의존하지 않는다.**

가능하면:

1. PR의 구현을 분석
2. 필요한 최소 변경만 현재 프로젝트에 명시적으로 port
3. 또는 검증된 특정 commit을 pin
4. 적용한 upstream commit/hash를 `provenance.json` 및 문서에 기록

한다.

---

# 4. 개발 원칙

## 4.1 upstream 파괴 금지

가능하면 CosyVoice 원본 코드를 무분별하게 수정하지 않는다.

권장 구조:

```text
project/
├── third_party/
│   └── CosyVoice/
├── mac_voice/
├── scripts/
├── configs/
├── data/
├── artifacts/
├── tests/
└── README.md
```

upstream 수정이 필요한 경우:

- patch 파일로 관리하거나
- 별도 adapter/wrapper 계층을 만든다.

예:

```text
patches/
├── cosyvoice_mps.patch
├── onnx_provider.patch
└── lora.patch
```

변경 이유를 반드시 기록한다.

---

## 4.2 먼저 현재 환경을 조사한다

설치를 시작하기 전에 아래를 확인한다.

```bash
uname -m
sw_vers
python --version
which python
conda env list
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.backends.mps.is_available())"
```

그리고 다음도 확인한다.

- 기존 CosyVoice checkout 존재 여부
- 기존 Conda environment 존재 여부
- Fun-CosyVoice3-0.5B 모델 존재 여부
- ffmpeg / sox 존재 여부
- 기존 환경에서 CosyVoice3 import 가능 여부

**정상 동작하는 기존 환경을 이유 없이 삭제하거나 재설치하지 않는다.**

---

# 5. 1단계 — Environment Doctor 구현

다음 명령을 만든다.

```bash
python -m mac_voice doctor
```

출력 예:

```text
[OK] architecture: arm64
[OK] Apple Silicon detected
[OK] PyTorch: 2.x
[OK] MPS built: True
[OK] MPS available: True
[OK] model directory found
[OK] llm.pt found
[OK] flow.pt found
[OK] hift.pt found
[OK] speech_tokenizer_v3.onnx found
[OK] campplus.onnx found
[OK] CosyVoice3 config found
[OK] ffmpeg found
[OK] sox found
```

추가로 다음을 출력한다.

- macOS version
- chip architecture
- Python version
- PyTorch version
- torchaudio version
- ONNX Runtime version
- 사용 가능한 ONNX providers
- MPS current allocated memory (가능한 경우)
- 모델 경로

문제가 있으면 단순 traceback보다 사람이 이해할 수 있는 오류를 출력한다.

---

# 6. 2단계 — Baseline MPS inference

학습 코드를 만들기 전에 반드시 **base model inference가 Mac에서 정상 동작하는지 먼저 증명한다.**

명령 예:

```bash
python -m mac_voice baseline \
  --model-dir "$COSYVOICE_MODEL_DIR" \
  --text "안녕하세요. 이것은 코지보이스 테스트입니다." \
  --output artifacts/baseline.wav
```

검증:

- 모델 로딩 성공
- 실제 device가 `mps`
- 텍스트 → WAV 생성
- WAV duration > 0
- RMS가 0에 가깝지 않음
- sample rate 확인
- NaN/Inf 없음

가능하면 zero-shot 테스트도 제공한다.

```bash
python -m mac_voice clone \
  --reference samples/reference.wav \
  --reference-text "참조 음성의 실제 문장입니다.<|endofprompt|>" \
  --text "새롭게 생성할 문장입니다." \
  --output artifacts/zero_shot.wav
```

CosyVoice3 prompt text에서 필요한 `<|endofprompt|>` 처리를 사용자가 매번 실수하지 않도록 wrapper에서 안전하게 처리한다.

---

# 7. 3단계 — 내 음성 데이터 입력 규격

사용자가 가장 단순하게 데이터를 넣을 수 있어야 한다.

권장 입력:

```text
data/my_voice/raw/
├── 0001.wav
├── 0002.wav
├── 0003.wav
└── ...

data/my_voice/transcripts.csv
```

`transcripts.csv`:

```csv
filename,text
0001.wav,안녕하세요. 첫 번째 녹음 문장입니다.
0002.wav,오늘은 제가 만들고 있는 앱에 대해 이야기해 보겠습니다.
0003.wav,이 음성은 인공지능 학습을 위한 데이터입니다.
```

speaker id 기본값:

```text
owner
```

---

# 8. 데이터 검증

다음 명령을 구현한다.

```bash
python -m mac_voice validate-data \
  --dataset data/my_voice
```

각 WAV에 대해 최소 다음을 검사한다.

- 파일 존재
- decode 가능 여부
- duration
- sample rate
- channel 수
- peak
- RMS
- clipping 여부
- silence 비율
- transcript 존재
- transcript 빈 문자열 여부
- 동일 filename 중복 여부

CosyVoice 공식 speech-token extractor는 30초 초과 음성에 제한이 있으므로:

```text
30초 초과 segment → 오류 또는 분할 필요 표시
```

를 한다.

권장 segment는 지나치게 길지 않도록 한다.

초기 권장:

```text
약 3~15초
```

단, 이 숫자를 hard failure 조건으로 사용하지 않는다.

---

# 9. 음성 전처리

다음 명령을 구현한다.

```bash
python -m mac_voice prepare \
  --dataset data/my_voice \
  --output data/my_voice_prepared
```

기본 처리:

- mono 변환
- 24 kHz 표준화
- WAV PCM 저장
- 파일명 정규화
- leading/trailing excessive silence 처리
- train/dev split
- manifest 생성

**주의:** 음색을 바꿀 정도의 aggressive denoise, compressor, EQ, pitch correction을 기본값으로 사용하지 않는다.

원본은 절대 수정하지 않는다.

항상:

```text
raw → prepared
```

별도 폴더에 생성한다.

---

# 10. CosyVoice dataset 구조 생성

CosyVoice3 공식 형식에 맞춰 최소 다음 파일을 생성한다.

```text
wav.scp
text
utt2spk
spk2utt
instruct
```

single speaker:

```text
speaker = owner
```

CosyVoice3용 instruct에는 upstream 형식을 참고하여 기본 instruction을 넣는다.

예:

```text
You are a helpful assistant.<|endofprompt|>
```

train/dev를 분리한다.

예:

```text
data/my_voice_prepared/train/
data/my_voice_prepared/dev/
```

split은 seed 기반으로 재현 가능해야 한다.

---

# 11. Speaker Embedding 추출

공식:

```text
tools/extract_embedding.py
```

구조를 최대한 재사용한다.

현재 upstream은 speaker embedding ONNX inference에:

```text
CPUExecutionProvider
```

를 사용한다.

이는 Mac에서도 사용할 수 있으므로 억지로 GPU화하지 않는다.

생성 결과:

```text
utt2embedding.pt
spk2embedding.pt
```

검증:

- 모든 utt에 embedding 존재
- speaker `owner` embedding 존재
- NaN/Inf 없음
- tensor shape 기록

---

# 12. Speech Token 추출 — Mac 패치 필요

공식 현재 `tools/extract_speech_token.py`는 ONNX provider를 다음처럼 CUDA에 고정해 놓은 상태를 확인했다.

```python
providers = ["CUDAExecutionProvider"]
```

Mac에서 이 코드를 그대로 사용하지 않는다.

provider abstraction을 만든다.

예:

```text
--onnx-provider auto
--onnx-provider coreml
--onnx-provider cpu
```

`auto`는 다음처럼 동작시킨다.

1. CoreMLExecutionProvider가 실제 모델에서 정상 작동하면 사용
2. 그렇지 않으면 CPUExecutionProvider
3. CUDA는 선택하지 않음

**CoreML 사용 가능 여부는 추측하지 말고 실제 InferenceSession 생성 + 1 sample inference로 검증한다.**

CoreML에서 문제가 있으면 CPU로 안전하게 fallback한다.

결과:

```text
utt2speech_token.pt
```

모든 segment에 token이 생성됐는지 검증한다.

---

# 13. Parquet 생성

공식 CosyVoice pipeline을 따라 parquet data를 생성한다.

결과:

```text
train/parquet/data.list
dev/parquet/data.list
```

또는 현재 upstream이 요구하는 동일 의미의 format.

`make_parquet_list.py`를 최대한 재사용한다.

작은 개인 dataset에 맞게 process 수를 과도하게 높이지 않는다.

---

# 14. LoRA 구현

1차 학습 대상:

```text
CosyVoice3 LLM
```

base parameter는 freeze한다.

LoRA target module은 실제 CosyVoice3/Qwen 구조를 introspect한 후 결정한다.

우선 후보:

```text
q_proj
k_proj
v_proj
o_proj
```

하지만 **이 이름을 존재 확인 없이 가정하지 않는다.**

코드에서 target module 발견 개수와 이름 일부를 출력한다.

예:

```text
LoRA target matched modules: 112
- ...q_proj
- ...k_proj
...
```

0개가 발견되면 즉시 실패시킨다.

초기 default:

```text
rank: 16
alpha: 32 또는 64
dropout: 0.05
```

모두 config에서 변경 가능해야 한다.

---

# 15. Trainable Parameter 검증

학습 직전에 반드시 다음을 출력한다.

```text
Total parameters: ...
Frozen parameters: ...
Trainable parameters: ...
Trainable ratio: ... %
```

그리고 다음 invariant를 검사한다.

```python
assert trainable_parameters > 0
assert trainable_parameters < total_parameters
```

base model 전체가 실수로 unfreeze되면 학습을 시작하지 않는다.

LoRA adapter 및 의도된 adaptation head 이외의 파라미터가 trainable이면 경고/실패 처리한다.

---

# 16. 핵심 — MPS Single-Device Trainer 구현

공식 CUDA/DDP trainer를 억지로 실행시키지 않는다.

별도의 trainer를 구현한다.

권장 파일:

```text
mac_voice/trainer_mps.py
```

또는:

```text
tools/train_mps_lora.py
```

핵심 구조:

```python
device = torch.device("mps")
model.to(device)

for batch in loader:
    batch = move_batch_to_device(batch, device)

    optimizer.zero_grad(set_to_none=True)

    loss = model(...)

    loss.backward()

    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm)

    optimizer.step()
```

---

# 17. MPS trainer에서 절대 제거할 CUDA/DDP 요소

다음이 training hot path에 있으면 안 된다.

```text
torch.cuda.set_device
torch.cuda.amp.*
torch.cuda.synchronize
torch.cuda.empty_cache
DistributedDataParallel
torch.distributed.init_process_group
dist.barrier
NCCL
torchrun 의존
DeepSpeed
CUDA_VISIBLE_DEVICES
```

Mac trainer는:

```text
single process
+
single MPS device
```

이다.

---

# 18. Precision 정책

처음부터 mixed precision 최적화에 집착하지 않는다.

**1차 correctness 목표는 FP32 학습 성공이다.**

순서:

1. FP32로 1 batch forward
2. FP32로 backward
3. optimizer step
4. loss finite 확인
5. checkpoint save/load 확인

이게 모두 성공한 후에만 optional mixed precision을 실험한다.

MPS는 float64를 지원하지 않으므로 MPS tensor를 float64로 cast하지 않는다.

float64가 반드시 필요한 작은 연산은 CPU에서 처리하고 결과를 MPS로 되돌린다.

---

# 19. Unsupported MPS op 처리

PyTorch에는 다음 fallback 옵션이 있다.

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1
```

이를 사용할 수는 있지만 **문제를 숨기기 위해 무조건 켜놓지 않는다.**

권장:

- 기본적으로 MPS 실행
- unsupported op 발생 시 정확한 op / stack trace 기록
- 해당 op만 CPU fallback 가능한지 검토
- fallback이 필요한 이유를 코드 주석과 문서에 기록

사용자 옵션:

```text
--allow-mps-fallback
```

으로 제공하는 것도 좋다.

---

# 20. Memory 정책

M3 Pro unified memory 환경을 고려한다.

기본값:

```text
batch_size = 1
gradient_accumulation_steps = configurable
num_workers = 작게 시작
```

지원한다면 gradient checkpointing을 옵션으로 추가할 수 있다.

단, 먼저 correctness를 확보한다.

학습 중 다음을 주기적으로 기록한다.

- step
- train loss
- validation loss
- learning rate
- step time
- samples/sec
- MPS allocated memory
- process RSS memory

OOM이 발생하면 사용자에게 다음 순서로 안내한다.

1. batch size 1 확인
2. gradient accumulation 조정
3. sequence length 제한
4. gradient checkpointing
5. LoRA rank 감소

MPS memory watermark를 위험하게 무제한으로 해제하는 설정을 기본값으로 넣지 않는다.

---

# 21. Optimizer

optimizer는 **trainable parameter만** 받는다.

예:

```python
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(params, ...)
```

기본 hyperparameter는 config로 관리한다.

예:

```yaml
training:
  batch_size: 1
  grad_accum_steps: 8
  learning_rate: 0.0001
  weight_decay: 0.01
  max_epochs: 20
  grad_clip: 1.0

lora:
  rank: 16
  alpha: 64
  dropout: 0.05
```

값은 초기 default일 뿐이며 hardcode하지 않는다.

---

# 22. Dry-run / Smoke Train

반드시 다음 명령을 만든다.

```bash
python -m mac_voice train \
  --config configs/my_voice.yaml \
  --max-steps 1
```

이 명령이 반드시 실제로 수행해야 하는 것:

```text
load base model
→ inject LoRA
→ freeze base
→ load real batch
→ forward
→ finite loss
→ backward
→ gradient 존재 확인
→ optimizer.step()
→ adapter checkpoint save
```

**forward만 성공하고 training 성공이라고 보고하지 않는다.**

반드시 `backward + optimizer.step`까지 실행해야 한다.

---

# 23. Gradient 검증

1 step 후 최소 다음을 검사한다.

LoRA parameter:

```text
grad != None
finite grad
non-zero grad
```

frozen base parameter:

```text
grad == None
```

이 검증을 자동화한다.

예:

```text
[OK] LoRA params received gradients
[OK] frozen base params have no gradients
[OK] optimizer step completed
```

---

# 24. Checkpoint

최소 지원:

```text
adapter_latest.pt
adapter_best.pt
```

저장 내용:

- LoRA weights
- 필요한 adaptation head weights
- step
- epoch
- validation loss
- optimizer state는 resume용 별도 저장 가능
- config snapshot

base model 전체 state dict를 매 checkpoint마다 저장하지 않는다.

---

# 25. Resume

다음이 가능해야 한다.

```bash
python -m mac_voice train \
  --config configs/my_voice.yaml \
  --resume artifacts/runs/my_voice/checkpoints/latest
```

resume 후:

- step 유지
- optimizer state 유지
- scheduler state 유지
- LoRA weights 동일하게 복원

간단한 자동 test를 만든다.

---

# 26. Validation / Overfitting 방지

single-speaker 소규모 dataset은 overfitting 가능성이 높다.

따라서 train/dev loss를 모두 기록한다.

최소 구현:

- validation every N steps
- best checkpoint
- patience 기반 optional early stopping
- loss history JSONL 저장

예:

```text
artifacts/runs/my_voice/metrics.jsonl
```

각 line:

```json
{"step": 100, "train_loss": 2.91, "val_loss": 3.02}
```

---

# 27. 학습 완료 후 Voice Package 생성

다음 명령을 만든다.

```bash
python -m mac_voice package \
  --run artifacts/runs/my_voice \
  --name my_voice \
  --output artifacts/voices/my_voice
```

포함:

```text
adapter.pt
voice.json
training_config.yaml
metrics.json
reference.wav
reference.txt
speaker_embedding.pt
provenance.json
```

`provenance.json`:

- CosyVoice upstream git commit
- 적용한 PR/patch
- PyTorch version
- Python version
- macOS version
- base model path
- base checkpoint hash 가능하면 기록
- LoRA config
- dataset summary
- build timestamp

---

# 28. LoRA inference

다음 명령을 구현한다.

```bash
python -m mac_voice synth \
  --voice artifacts/voices/my_voice \
  --text "안녕하세요. 이제 제 목소리로 생성한 음성입니다." \
  --output output/test.wav
```

동작:

```text
base CosyVoice3 load
→ LoRA module injection
→ adapter load
→ voice profile/reference load
→ Korean text normalization
→ speech generation
→ WAV save
```

Voice Package 안의 canonical reference를 자동으로 사용할 수 있게 한다.

사용자가 매번 reference WAV와 prompt text를 직접 지정하지 않아도 되게 한다.

다만 필요하면 override 옵션도 제공한다.

---

# 29. Before / After 비교

다음 명령을 제공한다.

```bash
python -m mac_voice compare \
  --voice artifacts/voices/my_voice \
  --text "같은 문장을 학습 전과 학습 후로 비교합니다." \
  --output-dir artifacts/comparison
```

생성:

```text
before_zero_shot.wav
after_lora.wav
comparison.json
```

가능하면 같은 reference / seed / text 조건을 유지한다.

사용자가 직접 A/B 청취할 수 있어야 한다.

---

# 30. 긴 문장 TTS는 별도 utility로 구성

1차 학습 시스템이 성공한 후 사용할 수 있게 다음 기능을 제공한다.

긴 텍스트 입력:

```text
문장 분리
→ chunk TTS
→ WAV 연결
```

예:

```bash
python -m mac_voice narrate \
  --voice artifacts/voices/my_voice \
  --input script.txt \
  --output narration.wav
```

단, **이 기능 때문에 핵심 학습 시스템 구현이 늦어져서는 안 된다.**

---

# 31. CLI 최종 형태

최소 다음 명령을 목표로 한다.

```bash
python -m mac_voice doctor

python -m mac_voice baseline \
  --text "테스트"

python -m mac_voice validate-data \
  --dataset data/my_voice

python -m mac_voice prepare \
  --dataset data/my_voice

python -m mac_voice features \
  --dataset data/my_voice_prepared

python -m mac_voice train \
  --config configs/my_voice.yaml \
  --max-steps 1

python -m mac_voice train \
  --config configs/my_voice.yaml

python -m mac_voice package \
  --run artifacts/runs/my_voice \
  --name my_voice

python -m mac_voice synth \
  --voice artifacts/voices/my_voice \
  --text "안녕하세요." \
  --output output.wav

python -m mac_voice compare \
  --voice artifacts/voices/my_voice \
  --text "비교 테스트입니다."
```

---

# 32. 한 번에 실행하는 shell wrapper

핵심 CLI가 모두 검증된 후에만 convenience script를 추가한다.

예:

```bash
./scripts/build_voice.sh my_voice
```

내부적으로:

```text
doctor
→ validate
→ prepare
→ features
→ smoke train
→ full train
→ package
→ sample synth
```

단계 중 하나가 실패하면 즉시 중단하고 원인을 출력한다.

---

# 33. GUI는 만들지 않는다

이번 1차 작업에서는 다음을 만들지 않는다.

- Gradio UI
- Electron
- Swift GUI
- Web dashboard
- training monitor web app
- fancy visualization

**CLI 시스템 완성이 먼저다.**

GUI는 학습 시스템이 실제로 안정화된 이후 별도 작업이다.

---

# 34. 테스트

최소 unit/integration test를 만든다.

예:

```text
tests/
├── test_device.py
├── test_dataset.py
├── test_manifest.py
├── test_onnx_provider.py
├── test_lora_injection.py
├── test_freeze.py
├── test_checkpoint.py
└── test_voice_package.py
```

중요 integration test:

```text
실제 CosyVoice3 base model load
→ LoRA inject
→ 실제 batch 1개
→ MPS forward
→ backward
→ optimizer step
→ save
→ reload
```

실제 사용자 음성이 아직 없는 경우:

- training 성공을 가짜/synthetic audio로 과장하지 않는다.
- dataset-independent unit test까지만 실행한다.
- 실제 audio가 필요한 test는 명확하게 `SKIPPED: real audio required`로 표시한다.

---

# 35. 완료 조건 — 매우 중요

Codex는 단순히 코드를 작성했다고 완료라고 하지 않는다.

## 시스템 구축 완료 조건

다음을 모두 만족해야 한다.

### A. 환경

```text
[PASS] MPS available
[PASS] base model found
[PASS] CosyVoice3 loads locally
```

### B. Inference

```text
[PASS] base CosyVoice3 generates a non-empty WAV on Mac
```

### C. Dataset

```text
[PASS] WAV + transcript input accepted
[PASS] CosyVoice manifests generated
[PASS] train/dev split generated
```

### D. Features

```text
[PASS] speaker embedding generated locally
[PASS] speech tokens generated locally without CUDA
[PASS] parquet/data.list generated
```

### E. LoRA

```text
[PASS] base weights frozen
[PASS] LoRA weights trainable
[PASS] trainable parameter count verified
```

### F. MPS training

실제 voice dataset이 제공된 상태에서는:

```text
[PASS] real batch forward on MPS
[PASS] finite loss
[PASS] backward succeeds
[PASS] LoRA gradients are non-zero
[PASS] frozen weights receive no gradients
[PASS] optimizer.step succeeds
```

이 단계가 가장 중요하다.

### G. Checkpoint

```text
[PASS] adapter checkpoint saved
[PASS] adapter checkpoint reloads
[PASS] resume works
```

### H. TTS

```text
[PASS] trained adapter loads
[PASS] Korean text generates WAV
[PASS] generated WAV is non-empty/non-silent
```

### I. Package

```text
[PASS] reusable Voice Package created
```

---

# 36. 절대 “완료”라고 하면 안 되는 상황

다음 상태에서는 완료라고 보고하지 않는다.

```text
Zero-shot cloning만 됨
```

```text
dataset preprocessing만 됨
```

```text
model forward만 됨
```

```text
MPS에서 inference만 됨
```

```text
CUDA training script를 만들어 놓고 Mac에서는 실행 안 해봄
```

```text
LoRA 코드를 추가했지만 backward 테스트 안 함
```

```text
adapter 저장은 되지만 재로딩 테스트 안 함
```

```text
실제 WAV 생성 확인 안 함
```

---

# 37. 문제가 생겼을 때 원칙

Mac/MPS에서 unsupported op가 나오면:

1. 정확한 stack trace 확보
2. 해당 op가 training의 어느 stage인지 확인
3. 최신 PyTorch에서 지원 여부 확인
4. 작은 범위의 CPU fallback 가능한지 확인
5. dtype 문제인지 확인
6. patch 최소화
7. regression test 추가

**“MPS에서 안 되니까 CUDA를 쓰자”는 해결책은 허용하지 않는다.**

필요하면 느리더라도 CPU fallback을 사용한다.

목표는:

```text
속도보다 먼저
Mac 한 대에서 완주
```

이다.

---

# 38. 구현 순서

반드시 아래 순서대로 진행한다.

## Phase 1

```text
환경 조사
doctor
```

## Phase 2

```text
CosyVoice3 MPS baseline inference
```

## Phase 3

```text
dataset validation
dataset preparation
```

## Phase 4

```text
speaker embedding
speech token
parquet
```

## Phase 5

```text
native LoRA
freeze 검증
```

## Phase 6

```text
MPS 1-batch forward
```

## Phase 7

```text
MPS backward
optimizer step
```

## Phase 8

```text
checkpoint / resume
```

## Phase 9

```text
full training pipeline
```

## Phase 10

```text
Voice Package
LoRA inference
before/after comparison
```

앞 단계가 실패했는데 다음 단계로 넘어가지 않는다.

---

# 39. Codex 작업 방식

작업하면서 다음 파일을 유지한다.

```text
STATUS.md
```

예:

```markdown
# Status

## Completed
- [x] Environment doctor
- [x] MPS base inference
- [x] Dataset validation

## In progress
- [ ] Speech token CPU/CoreML extraction

## Blocked
- none

## Last verified command
...

## Last verified result
...
```

또한:

```text
DECISIONS.md
```

에 중요한 설계 결정을 기록한다.

예:

```text
Why DDP was removed
Why FP32 is the first training target
Why speech tokenizer uses CPU provider
Why base model is not copied into voice package
```

---

# 40. README에 반드시 포함할 사용법

최종 README는 사용자가 최소한 아래만 보고 실행할 수 있어야 한다.

## Setup

```bash
...
```

## Model path

```bash
export COSYVOICE_MODEL_DIR=/Users/jawoongku/Models/Fun-CosyVoice3-0.5B
```

## Prepare voice dataset

```text
data/my_voice/raw/*.wav
data/my_voice/transcripts.csv
```

## Validate

```bash
python -m mac_voice validate-data --dataset data/my_voice
```

## Prepare

```bash
python -m mac_voice prepare --dataset data/my_voice
```

## Extract features

```bash
python -m mac_voice features --dataset data/my_voice_prepared
```

## Smoke training

```bash
python -m mac_voice train --config configs/my_voice.yaml --max-steps 1
```

## Full training

```bash
python -m mac_voice train --config configs/my_voice.yaml
```

## Build voice package

```bash
python -m mac_voice package ...
```

## Generate TTS

```bash
python -m mac_voice synth ...
```

---

# 41. 성능 최적화는 2차

처음부터 다음에 시간을 쓰지 않는다.

- `torch.compile`
- MLX 재작성
- custom Metal kernel
- quantization
- vLLM
- TensorRT
- multi-device
- distributed training

먼저:

```text
correctness
→ end-to-end
→ reproducibility
→ quality
→ speed
```

순서다.

MPS 학습 전체가 정상 동작한 후에만 profiling을 수행한다.

---

# 42. MLX에 대한 정책

MLX가 Apple Silicon에서 매력적인 것은 사실이지만, 이 프로젝트의 1차 목표는:

```text
CosyVoice3 upstream 구조를 최대한 유지한 PyTorch MPS LoRA training
```

이다.

CosyVoice3의 전체 학습 pipeline을 MLX로 새로 포팅하지 않는다.

이는 완전히 다른 프로젝트가 될 수 있다.

추후 inference 성능 개선용 MLX 포트는 별도 검토할 수 있다.

---

# 43. 실제 사용자 음성 데이터가 아직 없을 경우

시스템 구현을 계속 진행하되 **가짜로 “내 목소리 모델 완성”이라고 하지 않는다.**

아래 상태까지 완성한다.

```text
environment
baseline MPS inference
dataset tooling
feature pipeline
LoRA injection
MPS trainer
checkpoint system
voice package system
tests
```

실제 음성이 들어오면 다음 검증만 수행하면 되도록 만든다.

```text
prepare
→ features
→ max-steps 1
→ full train
→ package
→ synth
```

---

# 44. 사용자 데이터 보호

내 음성 원본과 학습 결과는 git에 commit하지 않는다.

`.gitignore`:

```gitignore
data/**/raw/
data/**/prepared/
artifacts/
output/
*.wav
*.flac
*.m4a
```

단 test fixture용 소형 공개/생성 데이터는 별도 관리 가능하다.

사용자의 실제 음성 파일을 외부 서버에 업로드하지 않는다.

---

# 45. 작업 종료 시 보고 형식

작업이 끝나면 장황한 설명보다 아래 형식으로 보고한다.

```markdown
# Result

## Working
- ...
- ...

## Verified
- MPS inference: PASS
- Dataset preparation: PASS
- Speech token extraction without CUDA: PASS
- LoRA injection: PASS
- MPS forward: PASS/NOT TESTED
- MPS backward: PASS/NOT TESTED
- Adapter save/reload: PASS/NOT TESTED
- TTS with adapter: PASS/NOT TESTED

## Commands
...

## Remaining issue
...

## Next action for user
...
```

`NOT TESTED`를 `PASS`처럼 표현하지 않는다.

---

# 46. 참고 소스

작업 전에 최신 상태를 다시 확인하라.

### CosyVoice

- Repository  
  https://github.com/QwenAudio/CosyVoice

- CosyVoice3 LibriTTS training recipe  
  https://github.com/QwenAudio/CosyVoice/blob/main/examples/libritts/cosyvoice3/run.sh

- CosyVoice3 config  
  https://github.com/QwenAudio/CosyVoice/blob/main/examples/libritts/cosyvoice3/conf/cosyvoice3.yaml

- Apple Silicon MPS PR #1869  
  https://github.com/QwenAudio/CosyVoice/pull/1869

- Native LoRA PR #1924  
  https://github.com/QwenAudio/CosyVoice/pull/1924

- Speaker embedding extractor  
  https://github.com/QwenAudio/CosyVoice/blob/main/tools/extract_embedding.py

- Speech token extractor  
  https://github.com/QwenAudio/CosyVoice/blob/main/tools/extract_speech_token.py

### PyTorch

- MPS backend  
  https://docs.pytorch.org/docs/stable/notes/mps.html

- torch.mps  
  https://docs.pytorch.org/docs/main/mps.html

- MPS environment variables  
  https://docs.pytorch.org/docs/stable/mps_environment_variables.html

---

# 47. 최종 지시

이 작업에서 가장 중요한 것은 **Mac에서 돌아가는 것처럼 보이는 코드**가 아니라 실제로:

```text
내 음성
→ 전처리
→ CosyVoice3 dataset
→ LoRA
→ MPS backward
→ optimizer step
→ adapter
→ 내 목소리 TTS WAV
```

가 연결되는 것이다.

공식 upstream이 CUDA 중심이라는 이유로 목표를 CUDA 학습으로 변경하지 마라.

필요하다면 upstream training loop에서 CUDA/DDP 부분을 제거하고 **single-process MPS trainer를 별도로 구현하라.**

기존 Fun-CosyVoice3-0.5B base weights는 보존한다.

첫 번째 성공 기준은 속도가 아니다.

> **Apple Silicon Mac 한 대에서 실제 LoRA optimizer step을 성공시키는 것.**

그다음:

> **생성된 adapter를 다시 로딩하여 한국어 WAV를 생성하는 것.**

여기까지 실제 검증된 후에만 시스템 구축 완료로 판단한다.
