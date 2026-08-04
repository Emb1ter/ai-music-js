# Architecture

`ai-music-js` runs the full music-generation pipeline inside desktop Chromium.
The host page coordinates isolated module Workers; model inference and audio
creation never require a generation backend.

## End-to-end flow

1. Optional lyric writing runs Qwen3.5 0.8B through Transformers.js/WebGPU.
2. Optional high-quality planning runs the ACE-Step 5 Hz 4B planner:
   - Phase 1 reasons over BPM, duration, key scale, and time signature.
   - Phase 2 generates exactly `durationSeconds × 5` semantic codes with
     conditioned/unconditional CFG 2.0.
3. The semantic detokenizer expands 5 Hz codes to 25 Hz, 64-channel hints.
4. Qwen3 text encoding and lyric-token embedding create text conditions.
5. The XL condition encoder packs text, lyrics, semantic hints, silence source
   latents, and masks into the DiT conditioning contract.
6. Deterministic seeded Gaussian noise creates the initial 25 Hz latent.
7. ACE-Step XL Turbo performs eight shifted flow-matching steps. Heun performs
   fifteen DiT evaluations; Euler SDE adds deterministic secondary noise.
8. The FP32 Oobleck VAE decodes overlapping, context-padded latent chunks into
   48 kHz stereo samples.
9. The Worker creates channel PCM and a PCM16 WAV; the public wrapper creates
   the page-compatible `AudioBuffer` after receiving the channel buffers.

## Execution placement

| Component | Primary execution |
|---|---|
| Qwen3.5 lyric writer | Transformers.js + WebGPU |
| ACE planner transformer body | ONNX Runtime WebGPU |
| Planner FP32 audio-code head | Custom WebGPU compute shader |
| Planner CFG and top-p sampling | Worker CPU JavaScript |
| Text, condition, semantic and XL DiT graphs | ONNX Runtime WebGPU |
| FP32 VAE | Strict WebGPU; optional full WASM retry |
| Tokenization, scheduling, DCW, WAV packaging | Worker CPU JavaScript |

Shape-related ONNX nodes may be assigned to CPU by ONNX Runtime. Those warning
messages are expected and do not mean the main tensor computation left WebGPU.

## Worker lifecycle and memory

Language stages and audio inference use separate Workers. The high-quality
planner releases its ONNX session, KV cache, embedding rows, and FP32 head
before the XL audio Worker starts. This prevents the 4B planner and 4B DiT from
remaining live at the same time.

VAE decoding uses ten-second core chunks with two seconds of convolution
context around internal boundaries. Each chunk is cropped before joining, so
memory stays bounded without introducing independent no-context seams.

## Progress and telemetry

Every `generate()`, `generateBatch()`, `writeLyrics()`, and `planMusic()` call
emits public updates shaped like:

```ts
type ProgressUpdate = {
  type: "progress";
  progress: number; // finite, monotonic, 0 through 1
  operation: "generate" | "generate-batch" | "write-lyrics" | "plan-music";
  stage: string;
  detail?: string;
};
```

The number is stage-weighted. Planner semantic codes, sampler steps, VAE
chunks, and sequential batch items contribute exact fractional progress inside
their stage ranges. Downloads contribute within the currently active stage.
It is suitable for a progress bar but is not an estimated percentage of wall
clock time. `runtime.progress` exposes the most recently emitted value.

Detailed updates remain available independently: downloads, stages, timings,
planner profiles, tensor traces, diagnostics, compatibility, completion, and
errors.

## Numerical and quality policy

Loading a graph is not a quality gate. New paths are accepted only after shape,
finite-value, deterministic scheduler, numerical-reference, and browser
listening checks appropriate to the component. Known failed planner and FP16
VAE experiments are retained in scripts/diagnostics so future changes do not
repeat already-disqualified approaches.
