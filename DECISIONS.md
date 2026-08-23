# Decisions

## 2026-08-23

- Keep the upstream CosyVoice checkout untouched where possible; add local wrappers under `mac_voice/`.
- Start with an environment doctor that degrades gracefully when optional ML dependencies are absent.
- Treat MPS availability and the local base-model path as explicit gates before inference or training.
- Keep the first training correctness target FP32 single-process MPS; do not add mixed-precision or distributed complexity early.
- Do not report training or voice-package completion until backward, optimizer step, adapter reload, and generated WAV checks pass.
- Use the existing clean `/Users/jawoongku/CosyVoice` checkout at commit `074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc` as the current upstream reference; wrappers call it without modifying its files.
