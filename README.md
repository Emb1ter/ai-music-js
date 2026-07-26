# ai-music-js

Generate instrumental or vocal music entirely inside a desktop browser with
**ACE-Step 1.5 XL Turbo**, ONNX Runtime Web, and WebGPU. Model loading,
tokenization, inference, VAE decoding, and WAV creation run in a Web Worker.
No generation server or API key is required.

> Experimental: start with 10-second generations and listen to the complete
> output before relying on longer tracks. Successful model loading alone does
> not prove that inference produced coherent music.

## Requirements

- Current desktop Chrome or Edge with WebGPU and hardware acceleration
- HTTPS in production, or `localhost` during development
- A capable desktop GPU; the tested development machine is an Apple M4 Max
  with 64 GB unified memory
- Approximately 5.25 GB of browser storage for the first model download
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

const music = new AceStepWebGpu({
  onUpdate(update) {
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

For vocals, supply the lyrics yourself. This no-planning-LM build does not
write lyrics:

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
```

The XL-specific model files download by default from the pinned
[Hugging Face repository](https://huggingface.co/emb1ter/ACE-Step-v1.5-XL-Turbo-ONNX-WebGPU).
Shared tokenizer, text encoder, lyric embedding, VAE, and silence-conditioning
assets use pinned upstream revisions. Large files stream to the browser's
Origin Private File System; smaller assets use Cache Storage.

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

- `onUpdate(update)` receives compatibility, download, stage, timing,
  diagnostic, tensor-trace, completion, and error updates.
- `modelBaseUrl` overrides the default Hugging Face directory for the six XL
  ONNX files.
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
- `lyrics` enables vocal generation. Omit it or pass `[Instrumental]` for an
  instrumental result.
- `vocalLanguage` is `unknown` or a language code such as `en`, `es`, or
  `zh-Hans`. It defaults to `unknown`.
- `seed` is a deterministic unsigned 32-bit seed and defaults to `42`.
- `durationSeconds` must be a whole number from 10 through 120.
- `sampler` is `euler` (default), `heun`, or `euler-sde`.
- `dcw` configures optional sampler-side correction. Browser DCW currently
  implements the official one-level Haar behavior for `low`, `high`, `double`,
  and `pix` modes. It is disabled by default to preserve results from versions
  before 0.3.
- `signal` can cancel an active generation.

Only one generation can run on an instance at a time.

### `generateBatch(options)`

```ts
const alternatives = await music.generateBatch({
  prompt: "Dreamy shoegaze with a soaring chorus",
  lyrics: "[Chorus]\nStay inside this sound",
  vocalLanguage: "en",
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
| `euler-sde` | 8 | Predicts clean audio and re-noises at each non-final step |

Heun and Euler SDE are separate modes. DCW can be combined with any sampler
and is applied after every step using the raw velocity at the current
timestep. `double` uses `t × scaler` for the Haar low band and
`(1 − t) × highScaler` for the high band.

### Lifecycle and cache

```ts
music.cancel();

const cache = await music.listCachedModels();
for (const model of cache.models) {
  console.log(model.id, model.complete, model.storedBytes);
}

await music.removeCachedModel("dit");
await music.clearCache();
music.dispose();
```

`listCachedModels()` reports every graph/support component, its individual
files, whether each file is complete, where it is stored (`opfs` or
`cache-api`), and the origin's current usage/quota estimate.
`removeCachedModel(id)` removes one component and returns a refreshed
inventory. `clearCache()` removes all of this library's Cache Storage and OPFS
entries. Removed files are downloaded again when the pipeline next needs them.

Browser storage is scoped to the exact origin. `localhost:3001`, a production
domain, and each new ngrok hostname all have separate caches. JavaScript running
on one origin cannot enumerate or remove data belonging to another; remove old
origins through Chromium's site-data settings.

Before a cold download, the runtime requests persistent storage and verifies
that the model plus temporary-write headroom fits the reported quota. Large
external-data shards are downloaded sequentially. If the browser denies
persistence or the quota is too small, free disk space, remove unneeded cached
components, use a stable non-Incognito origin, and retry.

`getRequiredAssets()` returns the resolved URL and byte size of every required
file for deployment checks or self-hosting.

## Current pipeline

- ACE-Step 1.5 XL Turbo, 4.169B-parameter DiT
- Instrumental text-to-music or vocals from user-supplied lyrics
- Deterministic initial and Euler-SDE secondary noise
- No planning LM, reference audio, or CFG pass
- Eight-step shift-3 Euler, Heun, or Euler-SDE flow matching
- Optional one-level Haar DCW
- Sequential multi-seed generation
- 10–120-second dynamic duration
- 48 kHz stereo `AudioBuffer` and PCM16 WAV output
- Heavy XL condition and DiT inference on WebGPU
- WASM compatibility fallback and memory-bounded VAE correctness path

## Download size and numerical validation

The complete cold model/support download is **5,245,621,594 bytes** (5.246 GB /
4.885 GiB). The six XL-specific files account for 3.062 GB.

Python and ONNX were compared with the same prompt, seed, learned silence
latent, and deterministic browser-compatible noise:

| Duration | Condition cosine | Final latent cosine | Waveform cosine |
|---:|---:|---:|---:|
| 10 seconds | 0.994970 | 0.960302 | 0.867693 |
| 20 seconds | 0.994971 | 0.877853 | 0.718767 |

The 20-second tensors remain finite and non-collapsed, but INT4 divergence
increases with duration. Long generations are supported by shape contracts,
not guaranteed to match the Python output at the 10-second quality level.

The VAE decodes ten-second cores with two seconds of convolution context on
each internal boundary, then crops and joins the cores. In standalone
validation, a one-shot 60-second WASM decode failed at approximately 4.13 GB
RSS with `std::bad_alloc`; the six-chunk decode completed at approximately
3.01 GB peak RSS. A 20-second full decode and its two-chunk equivalent were
bit-identical, including the join.

The Heun, Euler-SDE, and DCW scheduler equations are implemented from the
official ACE-Step source at commit
[`6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0`](https://github.com/ace-step/ACE-Step-1.5/commit/6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0).
Automated float32 snapshot tests cover the shifted schedule, Heun trapezoidal
update, SDE clean-prediction/re-noising calculation, deterministic secondary
streams, and native Haar DCW including odd-length padding.

The numerical table above describes the eight-step Euler instrumental path.
Vocal quality and final-audio parity for Heun, Euler SDE, and DCW still require
listening tests on the target browser/GPU before a release should claim
reference-level quality. Euler SDE is mathematically aligned with Python, but
its deterministic XorShift32 noise sequence intentionally differs from
PyTorch's RNG sequence.

## Unsupported

Automatic lyric writing/planning LM, reference audio,
cover/repaint/lego/extract, audio-code hints, CFG, base/SFT checkpoints,
non-Haar DCW wavelets, combined Heun+SDE, durations beyond 120 seconds, tiled
long-form generation, true GPU tensor batching, LoRA, mobile, Safari, and
Firefox.

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
- Browser XL INT4 ONNX export:
  [emb1ter/ACE-Step-v1.5-XL-Turbo-ONNX-WebGPU](https://huggingface.co/emb1ter/ACE-Step-v1.5-XL-Turbo-ONNX-WebGPU)
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
