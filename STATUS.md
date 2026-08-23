# Status

## Completed

- [x] Attached instruction analyzed and converted into `docs/COSYVOICE3_MAC_VOICE_TRAINING_PLAN.md`.
- [x] Original instruction moved into `docs/`.
- [x] Phase 0 environment survey completed.
- [x] Initial project directories created.
- [x] Initial `python -m mac_voice doctor` command implemented.
- [x] WAV/transcript dataset validation implemented.
- [x] Local ffmpeg-based preparation and deterministic train/dev manifests implemented.
- [x] Dataset unit tests added and passing.
- [x] ONNX provider selection and feature-stage prerequisite checks implemented.
- [x] Upstream-compatible speaker embedding, speech-token, and parquet wrapper paths implemented.
- [x] Generic LoRA injection, base-freeze, parameter statistics, and gradient invariant helpers implemented.
- [x] Single-device trainer primitives implemented without CUDA/DDP dependencies.
- [x] Adapter-only checkpoint save/load helpers implemented.
- [x] Voice Package builder implemented without copying base model weights.
- [x] YAML training config loader/validator and safe `train --max-steps` entrypoint implemented.
- [x] Upstream CosyVoice3 baseline adapter and `baseline` CLI implemented.
- [x] Zero-shot reference adapter and `clone` CLI implemented with prompt suffix normalization.
- [x] Voice Package validation and guarded `synth` CLI implemented.
- [x] CosyVoice3 LLM adapter injection and adapter-backed synth path implemented.
- [x] README usage guide and upstream checkout diagnostics added.
- [x] `compare` and long-text `narrate` orchestration utilities added.
- [x] Resume optimizer/scheduler state and JSONL metrics logger added.
- [x] Generic full training loop with validation and latest/best checkpoints added.
- [x] CosyVoice parquet `data.list` validation CLI added.
- [x] Generic training loop resume integration added.
- [x] Real MPS forward/backward/optimizer probe added (`mps-smoke`).
- [x] Core CosyVoice3 model assets prepared and CPU zero-shot clone verified with a temporary reference.
- [x] Real feature pipeline verified on a two-item temporary WAV fixture: manifests, CampPlus embeddings, speech tokens, parquet, and feature-column validation.
- [x] Feature artifact completeness and finite-value validation added; adapter synth WAV output uses the standard-library PCM writer for torchcodec compatibility.
- [x] Actual CosyVoice3 LLM introspection verified 24 `q_proj`, `k_proj`, `v_proj`, and `o_proj` matches each (170 Linear modules total); zero-match and invalid trainable-ratio guards added.
- [x] Actual CosyVoice3 CPU model accepted 96 LoRA targets (24 per projection), saved a 270,336-parameter adapter-only checkpoint, and reloaded it into a fresh model with matching metadata.
- [x] Voice Package and `synth` verified end-to-end on CPU with a temporary reference and initial (untrained) adapter: 4.84-second 24 kHz mono PCM WAV generated; synth now uses the standard-library WAV loader/writer around torchcodec incompatibility.
- [x] `compare` generated before/after WAVs and comparison JSON on CPU (4.44s/4.20s, both non-silent); `narrate` split a three-sentence script into chunks and generated a 1.40-second joined WAV.
- [x] Removed the stale adapter-inference blocker; prerequisite validation now matches the connected CPU `synth` path.
- [x] Doctor output verified for Apple Silicon, macOS, Python/PyTorch/torchaudio, MPS state, ONNX providers, model assets, ffmpeg, and upstream commit.
- [x] Added and verified `inspect-model`, which prints actual runtime LoRA target counts and representative module names.
- [x] Added and verified `model-forward-smoke`: real CosyVoice3 CPU LLM forward/loss with a synthetic valid token batch (`loss=5.716359`, 96 LoRA matches, 270,336 trainable parameters); backward/optimizer and user-data batch remain unverified.
- [x] Added and verified `model-backward-smoke`: real CosyVoice3 CPU synthetic batch backward, finite/non-zero LoRA gradients, frozen-base gradient absence, and optimizer step (`loss=5.716359`); this is not MPS or user-data training.
- [x] Added and verified `parquet-backward-smoke` on a real feature-bearing parquet row (`utt=a`, loss `2.469638`): tokenizer conversion, CPU forward/backward, gradient checks, and optimizer step all passed.
- [x] Added and verified `parquet-train-smoke` with one real train row and one dev row: 2 CPU steps, validation loss, and adapter checkpoint save (`train_loss=2.446580`, `dev_loss=1.962479`).
- [x] Repeated the real parquet train/dev smoke for 3 CPU steps successfully (`train_loss=2.421854`, `dev_loss=1.955798`), confirming multi-step stability on the fixture.
- [x] Extended parquet train smoke to emit optimizer resume state and `metrics.jsonl`; verified 2 train records plus validation record and checkpoint/state paths.
- [x] Verified fresh-process parquet resume: reloaded adapter and optimizer state at step 2, performed one additional CPU step, and saved a resumed adapter checkpoint (`resumed_loss=2.421854`).
- [x] Added a regression test for nested list-valued parquet feature columns; all 28 tests pass.
- [x] User-selected 10-file voice set (0002–0011) validated and prepared: 8 train / 2 dev, real embeddings/speech tokens/parquet generated and validated.
- [x] Real user-data CPU 2-step parquet smoke completed (`train_loss=3.883042`, `dev_loss=4.202456`); Voice Package built and synth generated a 3.48-second 24 kHz mono WAV.
- [x] Extended real user-data CPU LoRA training to 20 steps (`train_loss=3.684267`, `dev_loss=4.185104`), built a new Voice Package, and generated a 3.72-second user-reference WAV.
- [x] Generated a before/after CPU comparison for the 20-step adapter; both outputs are valid non-silent 24 kHz mono WAVs (before 3.68s, after 3.72s). Objective speaker-similarity scoring remains unavailable.
- [x] Synth reads LoRA rank/alpha/dropout from adapter checkpoint metadata; rank-2 user package inference is now compatible.
- [x] Added the first Gradio UI prototype: recommended sentence queue, microphone input, transcript field, and WAV quality report; 3 UI core tests pass.
- [x] Verified the UI can construct a Gradio `Blocks` app in the `cosyvoice` environment with Gradio 3.43.2; pinned compatible UI dependencies in `requirements-ui.txt`.
- [x] Added dataset registration from the UI: quality-approved recordings are numbered, copied to `raw/`, and appended to `transcripts.csv`; 32 tests pass.
- [x] Added `init-project` to create a safe starter layout with raw audio, transcript template, artifacts, and training YAML.
- [x] Added filesystem-backed job metadata (`job.json`) with queued/running/completed/failed/cancelled states and `job-status` CLI.
- [x] Added Voice Package discovery and validation with `list-voices` CLI.
- [x] Added Gradio TTS panel with Voice Package selection, text input, output path, and synth-engine error reporting; UI construction and 38 tests pass.
- [x] Connected generated WAV playback to the TTS panel; failed synthesis clears the audio output.
- [x] Added append-only TTS history and a UI refresh control for recent generations; 39 tests pass.
- [x] Added UI-wide dataset validation for WAV/transcript completeness; 40 tests pass.
- [x] Added UI action to normalize recordings and create train/dev manifests via `prepare`; 41 tests pass.
- [x] Added UI training preflight for config, model directory, and feature parquet readiness; 42 tests pass.
- [x] Added UI job-status refresh for filesystem-backed training jobs; 43 tests pass.
- [x] Added `job-create` CLI for queued training metadata; verified create/status lifecycle.
- [x] Added `job-update` CLI for status, step, and error transitions.
- [x] Added UI metrics viewer for recent train/validation loss and learning rate; 44 tests pass.
- [x] Added optional recognized-text comparison with normalized transcript similarity; 45 tests pass. Full ASR inference remains a later integration.
- [x] Added optional Whisper `tiny`/`base` ASR button with lazy model loading and error reporting; 46 tests pass.
- [x] Added isolated optional ASR dependency manifest (`requirements-asr.txt`) so the base environment remains unchanged.
- [x] Added Gradio 3-compatible separate microphone/upload inputs with unified inspection and registration; 47 tests pass.
- [x] Extended the recording quality gate to flag non-24kHz or non-mono input; 48 tests pass.
- [x] Connected the existing chunked `narrate` engine to a long-text UI panel with playback; 49 tests pass.
- [x] Added configurable long-text chunk size (60–400 chars) to the narration UI; 50 tests pass.
- [x] Added UI cancellation marker for jobs via `job.json`; actual process interruption remains a future runner integration; 51 tests pass.
- [x] Added `cancel_path` cooperative cancellation to the reusable training loop; 52 tests pass.
- [x] Added job state transition validation to prevent terminal jobs from restarting; 53 tests pass.
- [x] Added `parquet-train-job`, connecting real CPU parquet smoke training to queued/running/completed/failed job states.
- [x] Ran the job command on the real user parquet for 1 CPU step: train_loss=3.892549, dev_loss=3.897105; fixed job step reporting (`steps` fallback).
- [x] Added and ran `parquet-resume-job` on the generated adapter/state, confirming a fresh-model resume step completes.
- [x] Enhanced `job-status` to show train/dev loss and checkpoint/state paths when available.
- [x] Added `package-job` to build and record a Voice Package from a run with job state transitions.
- [x] Ran `package-job` on the real CPU train20 run; generated and validated `my_voice_cpu_train20_verified`.
- [x] Synthesized Korean text with the verified package: `artifacts/my_voice_cpu_train20_verified.wav` (24kHz mono, 5.84s).
- [x] Ran base zero-shot vs verified adapter comparison: both outputs non-silent 24kHz mono; report saved at `artifacts/compare_verified/comparison.json`.
- [x] Added UI button to start background CPU parquet training with job metadata; status/metrics panels monitor the run; 54 tests pass.
- [x] Background UI training now persists combined stdout/stderr to `job.log` beside `job.json`.
- [x] Added UI log refresh for the latest `job.log` lines; 55 tests pass.
- [x] UI-started training now records the spawned process PID in `job.json`.
- [x] Job cancellation now sends a best-effort SIGTERM to the recorded running PID before marking cancelled.
- [x] `job-status` now reports whether a recorded PID is alive.
- [x] Added `job-status --json` for machine-readable UI/SwiftUI bridge consumption.
- [x] Added dataset-aware next-sentence recommendation to the recording UI; 57 tests pass.
- [x] Comparison reports now explicitly distinguish audio statistics from unavailable speaker-similarity scoring.
- [x] Added `list-voices --json` for machine-readable Voice Package catalogs.
- [x] Added read-only Python bridge primitives for job snapshots and Voice Package catalogs; 59 tests pass.
- [x] Added `bridge-status` CLI for one-call combined job/catalog JSON snapshots.
- [x] Updated roadmap to reflect the implemented Python subprocess/JSON bridge; SwiftUI shell remains separate.
- [x] Added parse-checked SwiftUI shell sources in `mac_app/` for bridge status and Voice Package listing.
- [x] Added and validated a Swift Package manifest for opening the shell in Xcode or SwiftPM.
- [x] Built the macOS SwiftUI executable successfully with `swift build`.
- [x] Added AVAudioEngine microphone recording with WAV output to the SwiftUI shell; SwiftPM build remains successful.
- [x] Added macOS microphone permission request before AVAudioEngine start; filesystem permission packaging remains pending.
- [x] Added `mac_app/Info.plist` with `NSMicrophoneUsageDescription` for app-bundle permission prompts.
- [x] Added and ran `scripts/build_macos_app.sh` to create a release `.app` bundle; signing/notarization remains pending.
- [x] Added bundle identifier/version metadata and optional `SIGNING_IDENTITY` codesign path; notarization remains pending.
- [x] Signed builds now run `codesign --verify --deep --strict` after signing.
- [x] Added app bundle smoke-check script and `make check-app` target.
- [x] Integrated release `.app` build and bundle smoke-check into `scripts/verify_all.sh`.
- [x] Extended Swift CI to build and smoke-check the `.app` bundle.
- [x] Added `list-runs` for local training run, job status, checkpoint, and metrics discovery; 62 tests pass.
- [x] Extended the Python/Swift bridge snapshot with training run catalog data.
- [x] SwiftUI run list now shows checkpoint paths for model management handoff.
- [x] SwiftUI run list also shows metrics file paths.
- [x] Run catalog now includes and displays the latest train loss when available.
- [x] SwiftUI run list also displays validation loss and learning rate when available.
- [x] SwiftUI status screen auto-refreshes bridge data every 10 seconds.
- [x] Added guarded `scripts/notarize_macos_app.sh`; execution requires a configured Apple notarytool profile.
- [x] Added GitHub Actions CI for Python unit tests and macOS Swift release build.
- [x] Added `scripts/verify_all.sh` to run both local checks from the correct project directories.
- [x] Added Makefile shortcuts for verification, app bundling, and UI launch.
- [x] Added Makefile notarization target with the existing profile guard.
- [x] Synchronized roadmap: unsigned `.app` packaging is complete; Developer ID signing/notarization remains pending.
- [x] SwiftUI shell accepts `PVS_PROJECT_DIR` for reliable bridge paths when launched from an `.app` bundle.
- [x] SwiftUI bridge accepts `PVS_PYTHON` to select the Python/Conda runtime.
- [x] SwiftUI app accepts `PVS_JOB_PATH` and `PVS_VOICES_PATH` for project-specific state.
- [x] Final local verification after SwiftPM ignore update: 59 Python tests and Swift release build pass; worktree clean.
- [x] Added pure-Python cosine similarity primitive for future validated speaker embeddings; actual embedding extraction/scoring remains pending.
- [x] Exposed the similarity primitive through the `similarity` CLI command.
- [x] Similarity CLI now accepts JSON embedding files as well as inline vectors.
- [x] Marked the product specification-to-code-structure milestone complete for the implemented MVP layers.
- [x] Added AVAudioPlayer playback controls for the latest SwiftUI recording.
- [x] Connected SwiftUI text input to the Python `synth` CLI and generated-WAV playback; release build passes.
- [x] SwiftUI TTS now runs off the main thread with a disabled-in-progress button and completion/error updates.

## In progress

- [ ] Establish an MPS-capable Python environment.
- [x] Locate/provision the core `Fun-CosyVoice3-0.5B` model assets without duplicating an existing copy.
- [ ] Implement and verify baseline MPS inference.
- [ ] Verify baseline with a real local CosyVoice3 model and non-silent WAV.
- [x] Verify zero-shot clone with a temporary reference WAV and local model (CPU only).
- [ ] Connect the trained adapter to the CosyVoice3 inference graph for `synth`.
- [x] Verify adapter injection and synth with a real base model and CPU-smoke adapter (quality evaluation remains pending).
- [x] Add speaker embedding, speech-token, and parquet feature extraction.
- [x] Wire model-specific feature extraction to a pinned/available CosyVoice checkout.
- [x] Run feature extraction against real model assets and verify generated tensors/parquet (temporary fixture; user voice not tested).
- [ ] Connect package builder to a real trained adapter and reference audio.
- [ ] Apply LoRA helpers to the actual CosyVoice3 model and verify target modules.
- [ ] Run the trainer on real MPS with a real CosyVoice batch.
- [ ] Connect the validated train entrypoint to the actual CosyVoice3 model loader.
- [ ] Integrate metrics/resume utilities into the full CosyVoice3 trainer loop.
- [ ] Connect the generic loop to real CosyVoice3 parquet batches and MPS device.
- [ ] Validate real generated parquet schema and feature columns.
- [ ] Install/provide an MPS-capable runtime and local model assets.
- [ ] Verify compare/narrate with real model outputs.

## Blocked

- MPS remains unavailable after upgrading `cosyvoice` to PyTorch 2.13.0 (`mps_built=True`, `mps_available=False`).
- MPS probe reports the runtime rejection: `The MPS backend is supported on macOS 14.0+. Current OS version can be queried using sw_vers`; macOS reports 26.5.2, indicating this PyTorch build does not recognize the current OS version.
- `/Users/jawoongku/Models/Fun-CosyVoice3-0.5B` core assets are present; optional TensorRT estimator remains incomplete.
- `sox` is not installed; `ffmpeg` is available.

## Last verified command

```bash
python -m mac_voice prepare --dataset <dataset> --output <prepared>
```

## Last verified result

The doctor command reports model assets and the clean upstream commit in a human-readable format. Temporary WAV fixtures passed `validate-data` and `prepare`, including 24 kHz mono conversion and deterministic train/dev manifests. Core model assets are present, a CPU zero-shot clone generated a 13.6-second 24 kHz mono WAV, the real feature pipeline generated embeddings, speech tokens, and feature-bearing parquet validated through `data.list`, the Voice Package/synth path generated a 4.84-second 24 kHz mono PCM WAV from an initial adapter, and compare/narrate generated valid CPU outputs. Upstream-compatible feature wrappers, feature artifact finite-value checks, actual CosyVoice3 LLM target introspection and CPU adapter checkpoint reload, a base-model-free Voice Package builder, validated training config/entrypoint, baseline/zero-shot adapters, actual CosyVoice3 LLM adapter injection, README usage guide, compare/narrate utilities, resume/metrics utilities, a resumable generic validation/checkpoint training loop, parquet data-list validation, and MPS smoke probe are implemented without modifying `/Users/jawoongku/CosyVoice`; all 28 tests pass under the `cosyvoice` environment. MPS remains unavailable, and the adapter was untrained, so no trained voice quality claim is made.
