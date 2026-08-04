# ai-music-js

Generate instrumental or vocal music entirely inside a desktop browser with
**ACE-Step 1.5 XL Turbo**, ONNX Runtime Web, and WebGPU. Model loading,
tokenization, inference, VAE decoding, and WAV creation run in a Web Worker.
No generation server or API key is required.

Additional documentation:

- [Architecture and execution pipeline](docs/ARCHITECTURE.md)
- [Models, downloads, browser storage, and cache behavior](docs/MODELS_AND_STORAGE.md)
- [Troubleshooting browser, storage, memory, and quality issues](docs/TROUBLESHOOTING.md)
- [Release history](CHANGELOG.md)

> Experimental: start with 10-second generations and listen to the complete
> output before relying on longer tracks. Successful model loading alone does
> not prove that inference produced coherent music.

## Requirements

- Current desktop Chrome or Edge with WebGPU and hardware acceleration
- HTTPS in production, or `localhost` during development
- A capable desktop GPU; the tested development machine is an Apple M4 Max
  with 64 GB unified memory
- Approximately 5.63 GB for standard INT4 XL Turbo audio or 8.00 GB for the
  opt-in INT8 high-precision audio path
- 10.26 GB for the high-quality semantic planner plus standard audio, or
  12.64 GB when it is combined with high-precision audio; AI-written lyrics
  add 0.49 GB
- Enough time for the initial model download and shader/session compilation

Safari, Firefox, mobile browsers, and server-side Node.js are not supported.

## Install

```bash
npm install ai-music-js
```

## Quick start

```ts
import {
  AceStepWebGpu,
  DEFAULT_INSTRUMENTAL_PROMPT,
} from "ai-music-js";

const progressBar = document.querySelector<HTMLProgressElement>("progress")!;
const progressLabel = document.querySelector<HTMLElement>("#progress-label")!;

const music = new AceStepWebGpu({
  onUpdate(update) {
    if (update.type === "progress") {
      // Monotonic overall operation progress from 0 through 1.
      progressBar.value = update.progress;
      progressLabel.textContent = `${Math.round(update.progress * 100)}%`;
    }

    if (update.type === "download") {
      console.log(update.label, update.loaded, update.total);
    }

    if (update.type === "stage") {
      console.log(update.stage, update.detail);
    }
  },
});

const result = await music.generate({
  prompt: DEFAULT_INSTRUMENTAL_PROMPT,
  audioQuality: "standard",
  plannerQuality: "turbo",
  seed: 42,
  durationSeconds: 10,
  sampler: "euler",
});

const url = URL.createObjectURL(result.wav);
const audio = new Audio(url);
await audio.play();

// result.audioBuffer: 48 kHz stereo AudioBuffer
// result.wav: audio/wav Blob
// result.timings: stage timings in milliseconds
// result.trace: intermediate tensor summaries
```

For the numerically qualified INT8 condition encoder and DiT:

```ts
const result = await music.generate({
  prompt: DEFAULT_INSTRUMENTAL_PROMPT,
  audioQuality: "high",
  plannerQuality: "turbo",
  seed: 42,
  durationSeconds: 10,
});
```

`audioQuality: "high"` downloads 8.00 GB instead of 5.63 GB. In the
10-second Python reference test, the INT8 chain reached 0.999983 condition
cosine, 0.999882 final-latent cosine, and 0.999490 waveform cosine. Browser
WebGPU listening qualification is still required before treating this as a
general quality guarantee.

To use the browser-qualified INT8-weight / FP32-compute planner, change only
the planner choice. The downstream XL Turbo DiT/VAE pipeline remains the same:

```ts
const result = await music.generate({
  prompt: DEFAULT_INSTRUMENTAL_PROMPT,
  plannerQuality: "high-quality",
  seed: 42,
  durationSeconds: 30,
});
```

`turbo` is the fast direct XL path and the default. `high-quality` adds about
4.63 GB of prompt-dependent planner assets. Transformer weights are INT8;
activations, residuals, normalization, KV cache, sparse embeddings, the
audio-code head, CFG, and sampling remain FP32.

The high-quality planner follows ACE's production two-phase layout. Phase 1
uses constrained autoregressive reasoning to choose BPM, duration, key and
time signature. Caption and language rewriting remain disabled, matching the
local backend configuration, so the supplied music prompt and vocal language
stay authoritative. Phase 2 places the generated metadata in the assistant
reasoning prefix and generates exactly five semantic codes per final second.

For vocals, either supply lyrics or set `writeLyrics: true` to create them
locally with the pinned Qwen3.5 0.8B model. The caption must also request a
singer or vocals; a caption that only asks for an instrumental track
contradicts the lyric conditioning and is rejected before model loading:

```ts
const vocal = await music.generate({
  prompt: "Bright electropop, expressive female lead, huge melodic chorus",
  lyrics: `[Verse]
We follow every streetlight

[Chorus]
Sing it into daylight`,
  vocalLanguage: "en",
  seed: 1234,
  durationSeconds: 30,
  sampler: "heun",
  dcw: {
    enabled: true,
    mode: "double",
    scaler: 0.05,
    highScaler: 0.02,
  },
});

const aiWrittenVocal = await music.generate({
  prompt: "Bright electropop, expressive female lead, huge melodic chorus",
  writeLyrics: true,
  vocalLanguage: "en",
  seed: 1234,
  durationSeconds: 30,
  autoDuration: true,
  plannerQuality: "high-quality",
});
```

The lyric writer uses the same normal-vocal budget as the backend:
`max(40, round(durationSeconds × 7 / 6))` words. Use the exported
`assessLyricDuration()`, `recommendDurationForLyrics()`, and
`defaultMaxLyricWords()` helpers before generation when accepting arbitrary
user lyrics:

```ts
const fit = assessLyricDuration(userLyrics, 30);
if (!fit.fits) {
  console.warn(
    `${fit.wordCount} words need about ${fit.recommendedDurationSeconds}s`,
  );
}
```

The XL-specific model files download by default from the pinned
[Hugging Face repository](https://huggingface.co/emb1ter/ACE-Step-v1.5-XL-Turbo-ONNX-WebGPU).
The ACE-Step 5 Hz 4B planner uses a separately pinned
[split Q6/WebGPU-Q8 export](https://huggingface.co/emb1ter/ACE-Step-v1.5-5Hz-LM-4B-ONNX-WebGPU).
The opt-in high-quality path uses the separately versioned
[browser-qualified INT8-weight / FP32-compute body and sharded FP32 head](https://huggingface.co/emb1ter/ACE-Step-v1.5-5Hz-LM-4B-FP32-ONNX-WebGPU).
The same repository retains the original unquantized FP32 body as a numerical
reference and fallback artifact.
The validated FP32 VAE is hosted with the XL-specific files. Shared tokenizer,
text encoder, lyric embedding, and silence-conditioning assets use pinned
upstream revisions. Large files stream to the browser's
Origin Private File System; smaller assets use Cache Storage. Transformers.js
language files use an isolated `ai-music-js-transformers-v1` Cache Storage
bucket.

## Run the demo

```bash
git clone https://github.com/Emb1ter/ai-music-js.git
cd ai-music-js
npm install
npm run demo
```

Open the displayed `localhost` URL in desktop Chrome or Edge. The demo exposes
instrumental/vocal mode, lyrics, language, seed, duration, sampler, DCW, and
one-to-four sequential results. It displays download and inference progress
and provides audio playback plus WAV downloads.

To test a production build:

```bash
npm run build
npm run demo:build
npm exec vite preview -- --config examples/vite/vite.config.ts
```

## API

### `new AceStepWebGpu(options?)`

Useful options:

- `onUpdate(update)` receives normalized progress, compatibility, download,
  stage, timing, diagnostic, tensor-trace, completion, and error updates.
  `progress` updates contain a monotonic `progress` number from `0` through
  `1`, plus `operation`, `stage`, and optional `detail`. The same latest value
  is available synchronously as `music.progress`. Each new `generate()`,
  `generateBatch()`, `writeLyrics()`, or `planMusic()` operation resets it to
  `0`; successful completion sets it to `1`. Progress is stage-weighted and
  monotonic, but it is not an elapsed-time estimate.
- `modelBaseUrl` overrides the default Hugging Face directory for the selected
  XL condition, semantic-detokenizer, and DiT files.
- `allAssetsBaseUrl` points every required model/support file to one
  self-hosted directory.
- `allowWasmFallback` defaults to `true`.
- `workerFactory`, `workerUrl`, `wasmUrl`, and `wasmModuleUrl` support custom
  bundler/deployment setups.

### `generate(options)`

```ts
const result = await music.generate({
  prompt: "Cinematic orchestral instrumental with a memorable string melody",
  seed: 1234,
  durationSeconds: 20,
  signal: abortController.signal,
});
```

- `prompt` is required.
- `audioQuality` selects `standard` INT4 audio (default, 5.63 GB) or `high`
  INT8 condition/DiT audio (8.00 GB). Both use the same XL Turbo checkpoint,
  eight-step scheduler, semantic detokenizer, and validated FP32 WebGPU VAE.
- `lyrics` enables vocal generation. Omit it or pass `[Instrumental]` for an
  instrumental result.
- `writeLyrics: true` runs the pinned Qwen3.5 writer before planning. It is
  mutually exclusive with supplying `lyrics`.
- `plannerQuality` selects the end-to-end preset: `turbo` (default) runs
  direct XL Turbo text/lyric conditioning with no 4B planner; `high-quality`
  runs the verified 4.63 GB INT8-weight / FP32-compute 5 Hz planner first,
  expands its duration-constrained codes to 25 Hz hints, and then runs the
  same XL Turbo audio model. The high-quality conditional and training-aligned
  `NO USER INPUT` branches use CFG 2.0 and receive the same sampled code at
  every autoregressive step.
- `semanticPlanning` defaults to `true` for the high-quality preset. Set it to
  `false` to bypass planning explicitly.
- `semanticCodeIds` and `plannerMetadata` can be supplied together to hand a
  separately qualified ACE planner result to the audio pipeline. The code
  array must contain exactly `durationSeconds × 5` integer codebook indices
  from 0 through 63999. This path skips the language Worker, making it possible
  to release a large planner session before loading XL DiT:

  ```ts
  const result = await music.generate({
    prompt,
    lyrics,
    durationSeconds: 30,
    semanticCodeIds: qualifiedPlan.semanticCodeIds,
    plannerMetadata: qualifiedPlan.plannerMetadata,
  });
  ```
- `vocalLanguage` is `unknown` or a language code such as `en`, `es`, or
  `zh-Hans`. It defaults to `unknown`.
- `seed` is a deterministic unsigned 32-bit seed and defaults to `42`.
- `durationSeconds` must be a whole number from 10 through 120.
- `autoDuration: true` treats `durationSeconds` as the user's minimum
  preference. For vocals, the runtime first calculates the shortest duration
  that satisfies the backend lyric budget. The high-quality ACE metadata
  phase may select a longer duration, but never a shorter one. The direct
  Turbo path uses the deterministic lyric recommendation because it skips the
  4B planner.
- `sampler` is `euler` (default), `heun`, or `euler-sde`.
  `euler-sde` is currently restricted to instrumental generation.
- `dcw` configures optional sampler-side correction. Browser DCW currently
  implements the official one-level Haar behavior for `low`, `high`, `double`,
  and `pix` modes. It is disabled by default to preserve results from versions
  before 0.3.
- `signal` can cancel an active generation.

Only one generation can run on an instance at a time.

### Language stages

```ts
const lyrics = await music.writeLyrics({
  prompt: "A hopeful synth-pop song about finding your way home",
  durationSeconds: 30,
  seed: 42,
});

const plan = await music.planMusic({
  prompt: "Hopeful synth-pop with a clear female vocal",
  plannerQuality: "high-quality",
  lyrics: lyrics.lyrics,
  vocalLanguage: "en",
  durationSeconds: 30,
  autoDuration: true,
  seed: 42,
});
```

`writeLyrics()` and `planMusic()` can be used independently. `planMusic()`
returns enriched metadata plus exactly five ACE codebook indices per output
second. Its `metadata.durationSource` is `requested`, `ace`, or `recommended`.
Normal `generate()` calls the planner automatically.

### `generateBatch(options)`

```ts
const alternatives = await music.generateBatch({
  prompt: "Dreamy instrumental shoegaze with evolving guitars and drums",
  seeds: [100, 101, 102],
  durationSeconds: 20,
  sampler: "euler-sde",
});
```

`generateBatch()` accepts one through eight seeds and generates each result
sequentially. It deliberately does not create a larger GPU tensor batch, so
peak inference memory stays close to one generation. `batch-progress` updates
report the current seed. Each result gets its own deterministic initial latent;
Euler SDE also derives stable, independent secondary-noise streams from that
seed.

### Samplers and DCW

| Setting | DiT evaluations | Behavior |
|---|---:|---|
| `euler` | 8 | Original XL Turbo browser path |
| `heun` | 15 | Predictor/corrector ODE sampling; roughly doubles DiT time |
| `euler-sde` | 8 | Experimental instrumental-only mode; re-noises after each non-final step |

Heun and Euler SDE are separate modes. DCW can be combined with any sampler
and is applied after every step using the raw velocity at the current
timestep. `double` uses `t × scaler` for the Haar low band and
`(1 − t) × highScaler` for the high band. For vocals, use Euler first; Heun is
available when the additional generation time is acceptable.

### Lifecycle and cache

```ts
music.cancel();

const cache = await music.listCachedModels();
for (const model of cache.models) {
  console.log(model.id, model.complete, model.storedBytes);
}

await music.removeCachedModel("dit");
await music.removeCachedModel("dit-int8");
await music.removeCachedModel("condition-encoder-int8");
await music.removeCachedModel("music-planner");
await music.removeCachedModel("music-planner-high-quality");
await music.removeCachedModel("lyrics-writer");
await music.clearCache();
music.dispose();
```

`listCachedModels()` reports every graph/support component, its individual
files, whether each file is complete, where it is stored (`opfs` or
`cache-api`), and the origin's current usage/quota estimate.
`removeCachedModel(id)` removes one component and returns a refreshed
inventory. The planner and lyric writer appear beside the audio graphs.
`clearCache()` removes all of this library's isolated Cache Storage and OPFS
entries. Removed files are downloaded again when the pipeline next needs them.

The high-quality planner caches files independently. Before each planner file
is stored, the library checks Cache API and OPFS for the selected audio profile
and reserves space only for audio assets that are still missing. A large
planner shard can therefore be streamed while smaller shards are retained;
one skipped shard no longer disables caching for the entire planner.

The planner's complete tied FP32 embedding tensor is 2.22 GB, so the library
does not duplicate it beside the already-cached 655 MB audio-code head.
Instead it preloads the normal-text rows needed by the prompt and constrained
metadata vocabulary, persists each used row as a 10 KiB Cache API entry, and
reuses the selected audio token's exact row directly from the loaded FP32 head.
After warm-up, autoregressive semantic generation performs no embedding HTTP
range reads. `listCachedModels()` includes the persisted-row count and bytes
under `music-planner-high-quality`; removing that component removes the rows.

Browser storage is scoped to the exact origin. `localhost:3001`, a production
domain, and each new ngrok hostname all have separate caches. JavaScript running
on one origin cannot enumerate or remove data belonging to another; remove old
origins through Chromium's site-data settings.

Before a cold download, the runtime requests persistent storage and verifies
that the model plus temporary-write headroom fits the reported quota. Large
external-data shards are downloaded sequentially. If the browser denies
persistence or the quota is too small, free disk space, remove unneeded cached
components, use a stable non-Incognito origin, and retry.

`getRequiredAssets()` returns the default profile. Pass
`{ audioQuality: "high" }` to receive the exact INT8 file list and byte sizes
for deployment checks or self-hosting.

## Current pipeline

- ACE-Step 1.5 XL Turbo, 4.169B-parameter DiT
- Selectable INT4 standard or INT8 high-precision XL Turbo audio
- Selectable direct XL Turbo or INT8-weight / FP32-compute-planned XL path
- ACE Phase 1 constrained BPM/duration/key/time-signature reasoning
- Automatic lyric-fit duration with the backend vocal word budget
- Duration-constrained semantic-code planning
- Conditioned/unconditional semantic-code sampling with CFG 2.0
- 5 Hz codebook detokenization into 25 Hz XL semantic hints
- Instrumental text-to-music or vocals from supplied or Qwen-written lyrics
- Deterministic initial and Euler-SDE secondary noise
- Eight-step shift-3 Euler, Heun, or Euler-SDE flow matching
- Optional one-level Haar DCW
- Sequential multi-seed generation
- 10–120-second dynamic duration
- 48 kHz stereo `AudioBuffer` and PCM16 WAV output
- Heavy XL condition and DiT inference on WebGPU
- Browser-qualified FP32 VAE on strict WebGPU with a full WASM retry
- Memory-bounded, context-overlapped VAE chunk decoding
- Monotonic overall progress from 0 through 1 for generation and language jobs

## Download size and numerical validation

The standard audio model/support download is **5,626,494,229 bytes**
(5.626 GB / 5.240 GiB). The INT8 high-precision profile is
**8,004,092,572 bytes** (8.004 GB / 7.454 GiB). The high-quality hybrid planner
is **4,633,150,982 bytes** (4.315 GiB), making planner plus standard audio
**10,259,645,211 bytes** (9.555 GiB), or planner plus high-precision audio
**12,637,243,554 bytes** (11.769 GiB). The optional experimental split
Q6-body/WebGPU-Q8 planner cache is **3,628,429,574 bytes**; it is exposed by
`planMusic()` but is not part of the reliable Turbo preset because its ONNX
body can exhaust the browser WASM heap during `OrtRun`. The optional Qwen3.5
lyric writer is 489,166,749 bytes.

Verification status for this planner release:

- The default high-quality body uses 252 asymmetric INT8 `MatMulNBits`
  projections with block size 64. Every declared activation, residual,
  normalization, and KV-cache value remains FP32; the sparse embedding rows,
  sharded 64,000-row output head, CFG, and sampling also remain FP32.
- The hybrid path was qualified in desktop Chromium on Apple Metal 3. Its
  3.67 GiB transformer-body session was created in 2.5 seconds; 150 cached
  semantic steps produced 148 unique codes in 9.0 seconds. First-step
  unquantized-FP32 top-20 overlap was 100% conditional, 80% unconditional,
  and 60% after CFG 2.0, clearing the 50% ranking gate for every branch.
- Native first-step comparison measured centered logit correlations of
  0.999778 conditional, 0.999755 unconditional, and 0.999837 after CFG.
  Native top-20 overlap was 100%, 100%, and 90%, respectively. A separate-row
  ten-step KV-cache test kept all 64,000 logits finite and produced ten unique
  codes.

- The unquantized FP32 high-quality path was run end to end in desktop
  Chromium on Apple Metal 3. Its 13.56 GiB transformer-body session was
  created in 12.9 seconds; 150 cached semantic steps produced 144 unique codes
  in 16.0 seconds; conditional, unconditional, and CFG first-step top-20
  overlap was 100%, 80%, and 55%. The released plan then completed all eight
  XL Turbo DiT evaluations and three VAE chunks, producing a coherent
  30-second WAV that passed listening validation.

- The former INT4 planner export is rejected. Although it produced finite
  tensors, its first-step conditional, unconditional, and CFG top-20
  audio-code rankings had zero overlap with the official MLX checkpoint. Its
  resulting audio was a broadband pulse/noise texture with no intelligible
  vocals.
- A direct FP16 conversion initially overflowed at
  `model.layers.6.mlp.down_proj.MatMul`. The current graph scales the entire
  residual stream and all attention/MLP output projections by `1/256`, while
  scaling residual RMSNorm epsilon by the square of that factor. Normalized
  activations are mathematically unchanged and all first-step logits are
  finite.
- The full FP16 export preserved ranking but its 8.43 GB download caused a
  Chromium renderer reload. A 4.50 GB blockwise-Q8 build and a 3.59 GB
  Q6/Q8 build also preserved ranking but ONNX Runtime Web failed during
  session creation with `std::bad_alloc`. The retained experimental split-Q6
  design keeps its ONNX body at approximately 3.00 GB and evaluates the shared
  591 MB Q8 token embedding and output head in dedicated WebGPU kernels; the
  production high-quality path instead uses the qualified INT8/FP32 body and
  sharded FP32 head.
- The Q6 body contains ONNX unpack/dequantization subgraphs. Its Transformers.js
  session is created with graph optimization, the CPU memory arena, and memory
  patterns disabled. This prevents ONNX Runtime from constant-folding the
  packed Q6 weights into a temporary full FP16 copy during session creation;
  dequantization remains runtime WebGPU work.
- On the exact 30-second vocal test prompt, the complete split body plus Q8
  embedding/head versus official MLX has first-step top-20 overlap of 40%
  conditional, 85% unconditional, and 50% after CFG 2.0. Centered logit
  correlations are 0.982154, 0.974738, and 0.986143 respectively. The custom
  Q8 browser result still requires end-to-end validation.
- A native ONNX Runtime cached-step smoke test completed all 150 semantic
  steps with finite logits: 124 unique codes, 149 transitions, longest
  identical run 1, and dominant-code count 17.
- The codes phase restores ACE-Step's official CFG 2.0 behavior: a
  conditioned row and a training-aligned `NO USER INPUT` row, CFG calculated
  in float32 over only the valid 64,000 audio-code logits, and the same sampled
  token appended to both KV caches. It also reports unique-code count,
  transition count, adjacent-repeat ratio, longest run, and dominant-code
  ratio.
- The semantic detokenizer completes a finite 10-second contract pass from
  `[1, 50, 1]` codes to `[1, 250, 64]` hints and was separately compared with
  the official XL PyTorch module (maximum absolute difference `0.001953125`,
  mean absolute difference approximately `0.000288`).
- Unit, type, package, Worker, and demo builds pass.
- The packaged high-quality planner-to-XL-Turbo path completed 30-second vocal
  browser runs on the tested Apple M4 Max and produced coherent music with
  intelligible vocal content in listening validation. This qualifies the
  browser pipeline as functional; it does not claim sample-identical output or
  universal quality parity with the Python backend on other prompts/machines.

Python and ONNX were compared with the same prompt, seed, learned silence
latent, and deterministic browser-compatible noise. The ten-second INT8 audio
chain produced:

| Profile | Condition cosine | Final latent cosine | Waveform cosine |
|---|---:|---:|---:|
| INT8 high precision | 0.999983 | 0.999882 | 0.999490 |
| INT4 standard | 0.994970 | 0.960302 | 0.867693 |

The existing INT4 duration comparison remains:

| Duration | Condition cosine | Final latent cosine | Waveform cosine |
|---:|---:|---:|---:|
| 10 seconds | 0.994970 | 0.960302 | 0.867693 |
| 20 seconds | 0.994971 | 0.877853 | 0.718767 |

The 20-second tensors remain finite and non-collapsed, but INT4 divergence
increases with duration. Long generations are supported by shape contracts,
not guaranteed to match the Python output at the 10-second quality level.

The VAE decodes ten-second cores with two seconds of convolution context on
each internal boundary, then crops and joins the cores. The official BF16
checkpoint was upcast directly to an FP32 ONNX decoder; it was not derived
from the former FP16 conversion. In the isolated Chromium qualification, FP32
WebGPU was 8.12 times faster than the matching pthread/WASM reference and
produced cosine `1.0000000000`, maximum absolute difference `2.07201e-5`, and
mean absolute difference `4.91866e-7`. The former FP16 WebGPU decoder is not
used: it produced cosine `0.1748364475` against its WASM reference. If FP32
WebGPU session creation or inference fails and `allowWasmFallback` is enabled,
the Worker restarts every VAE chunk using the same FP32 graph on WASM.

In earlier standalone WASM validation, a one-shot 60-second decode failed at
approximately 4.13 GB RSS with `std::bad_alloc`; the six-chunk decode completed
at approximately 3.01 GB peak RSS. A 20-second full decode and its two-chunk
equivalent were bit-identical, including the join.

The Heun, Euler-SDE, and DCW scheduler equations are implemented from the
official ACE-Step source at commit
[`6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0`](https://github.com/ace-step/ACE-Step-1.5/commit/6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0).
Automated float32 snapshot tests cover the shifted schedule, Heun trapezoidal
update, SDE clean-prediction/re-noising calculation, deterministic secondary
streams, and native Haar DCW including odd-length padding.

The numerical table above describes the eight-step Euler audio path before
the newly added semantic planner is applied.
The vocal Euler path was also compared using explicitly vocal captions and
supplied English lyrics:

| Duration | Condition cosine | Final latent cosine | Waveform cosine |
|---:|---:|---:|---:|
| 10 seconds | 0.995112 | 0.846388 | 0.495369 |
| 30 seconds | 0.995092 | 0.786107 | 0.506491 |

The dynamic lyric conditioning is present and a small ASR probe recovered
supplied lyric content from both Euler outputs, but the INT4 audio is not
numerically close enough to claim Python parity. Euler SDE lost vocal lyric
intelligibility in the same probe and is therefore rejected for vocal
requests. Heun and vocal DCW still require broader listening tests on target
browser/GPU combinations. Euler SDE remains mathematically aligned with
Python for instrumental use, but its deterministic XorShift32 noise sequence
intentionally differs from PyTorch's RNG sequence.

## Unsupported

Reference audio, cover/repaint/lego/extract, user-supplied audio-code hints,
DiT CFG, base/SFT checkpoints, non-Haar DCW wavelets, combined Heun+SDE,
durations beyond 120 seconds, tiled long-form generation, Euler-SDE vocals,
true GPU tensor batching, LoRA, mobile, Safari, and Firefox. The official
planner's caption, genre, and language rewriting branches remain disabled;
the supplied caption and vocal language are authoritative. Semantic-code
range, count, and deterministic sampling constraints are enforced.

## Development

```bash
npm install
npm test
npm run typecheck
npm run build
npm run demo:build
npm run check
```

`npm run check` runs unit tests, TypeScript checking, the library/Worker build,
the demo build, and an npm package dry run.

## Model attribution

- Original XL Turbo model:
  [ACE-Step/acestep-v15-xl-turbo](https://huggingface.co/ACE-Step/acestep-v15-xl-turbo)
- Browser XL INT4/INT8 ONNX exports:
  [emb1ter/ACE-Step-v1.5-XL-Turbo-ONNX-WebGPU](https://huggingface.co/emb1ter/ACE-Step-v1.5-XL-Turbo-ONNX-WebGPU)
- Browser ACE 5 Hz planner split-Q6/WebGPU-Q8 export:
  [emb1ter/ACE-Step-v1.5-5Hz-LM-4B-ONNX-WebGPU](https://huggingface.co/emb1ter/ACE-Step-v1.5-5Hz-LM-4B-ONNX-WebGPU)
- Browser ACE 5 Hz planner hybrid INT8/FP32 and reference FP32 exports:
  [emb1ter/ACE-Step-v1.5-5Hz-LM-4B-FP32-ONNX-WebGPU](https://huggingface.co/emb1ter/ACE-Step-v1.5-5Hz-LM-4B-FP32-ONNX-WebGPU)
- Experimental ONNX starting point and shared assets:
  [shreyask/ACE-Step-v1.5-ONNX](https://huggingface.co/shreyask/ACE-Step-v1.5-ONNX)

This is an unofficial experimental project and is not affiliated with or
endorsed by the ACE-Step authors.

## License

The original `ai-music-js` source is available under the [MIT License](LICENSE).
The npm package also redistributes third-party Worker/WASM code, and the
browser downloads separately licensed model artifacts at runtime. See
[Third-party notices](THIRD_PARTY_NOTICES.md) for the applicable MIT and
Apache-2.0 license texts and attribution.
