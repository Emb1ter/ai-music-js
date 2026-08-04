# Changelog

All notable changes to `ai-music-js` are documented here.

## 0.4.0 — 2026-08-04

This is the first npm release containing the complete browser-local vocal and
high-quality semantic-planning pipeline developed after 0.1.0.

### Added

- Optional ACE-Step 5 Hz 4B high-quality planner using an INT8-weight,
  FP32-compute ONNX transformer body and a sharded FP32 WebGPU audio-code head.
- ACE metadata reasoning for BPM, duration, key scale, and time signature.
- Conditioned/unconditional semantic-code sampling with CFG 2.0 and exact
  five-code-per-second duration constraints.
- Local Qwen3.5 0.8B lyric writing, lyric repair, backend-aligned word budgets,
  lyric-fit assessment, and automatic duration recommendations.
- Selectable standard INT4 and high-precision INT8 XL Turbo audio graphs.
- FP32 WebGPU Oobleck VAE decoding with chunked long-output execution and a
  correctness-preserving WASM fallback.
- Euler, Heun, deterministic Euler SDE, Haar DCW, and sequential multi-seed
  generation.
- Cache inspection and removal APIs, browser quota reporting, per-file planner
  cache policy, and persistent sparse FP32 planner-embedding rows.
- Exact generated-token embedding reuse from the already-loaded FP32 head,
  removing autoregressive embedding network requests.
- Detailed pipeline/planner timing telemetry, tensor diagnostics, model
  download progress, and a normalized monotonic progress API from 0 through 1.
- A full Vite demo for generation, playback, WAV download, model-cache
  management, storage reporting, progress, and timing inspection.

### Fixed

- Corrected early browser outputs that collapsed to tones or silence by
  restoring the proper XL Turbo conditioning, scheduling, and VAE path.
- Avoided Cache API quota failures by caching planner assets independently and
  streaming files that do not fit while reserving space for audio models.
- Avoided ONNX Runtime Web integer-overflow and shape-merge failures by splitting
  the 4B planner body/head and exporting the correct headless body contract.
- Rejected numerically broken INT4 planner output and qualified the
  INT8-weight/FP32-compute replacement against native FP32 rankings.
- Replaced the inaccurate FP16 WebGPU VAE with the directly upcast FP32 export.
- Removed the production planner's per-token remote sparse-embedding latency.

### Validation

- Browser-qualified on desktop Chromium using Apple Metal 3 on an Apple M4 Max
  with 64 GB unified memory.
- FP32 WebGPU VAE versus pthread/WASM: cosine `1.0000000000`, maximum absolute
  difference `2.07201e-5`, mean absolute difference `4.91866e-7`.
- INT8 audio reference, 10 seconds: condition cosine `0.999983`, final-latent
  cosine `0.999882`, waveform cosine `0.999490`.
- High-quality planner: native first-step centered-logit correlation
  `0.999778` conditional, `0.999755` unconditional, and `0.999837` after CFG.
- End-to-end 30-second browser vocal runs completed with coherent music and
  intelligible vocal content in listening validation.

## 0.1.0 — 2026-07-26

- Initial experimental npm package and standalone desktop-Chromium demo.
- Browser-local ACE-Step XL Turbo inference using ONNX Runtime Web and WebGPU.
- Instrumental generation, deterministic eight-step Euler sampling, WAV output,
  Worker execution, browser caching, and compatibility diagnostics.
