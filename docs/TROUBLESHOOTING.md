# Troubleshooting

## Browser compatibility

Use current desktop Chrome or Edge with hardware acceleration. Production pages
must use HTTPS; `localhost` is accepted during development. Safari, Firefox,
mobile browsers, private/incognito storage, and server-side Node.js inference
are not supported.

If compatibility fails, inspect the `compatibility` and `error` updates. The
adapter must support WebGPU and sufficiently large storage buffers for the
selected graph.

## Worker module could not be loaded

- Build the package before serving a demo that imports `dist`:

  ```bash
  npm run build
  npm run demo
  ```

- Serve the ONNX Runtime `.mjs` and `.wasm` files from the same package build.
- When self-hosting, pass correct `workerUrl`, `languageWorkerUrl`, `wasmUrl`,
  and `wasmModuleUrl` values.
- Configure the server to allow the hostname used by an ngrok tunnel.
- Do not mix an old Worker bundle with a newly built package entry point.

## Cache or quota failure

Read `listCachedModels()` and inspect `availableBytes`, `usageBytes`, and
`quotaBytes`. Free disk space is not the same as origin quota; Chromium decides
how much of the disk one origin may use.

Use one stable origin, request persistent storage, avoid Incognito, remove
unused model profiles, and retry. The planner can stream a shard that does not
fit, but ONNX Runtime still needs enough transient memory to create the session.

```ts
const inventory = await runtime.listCachedModels();
await runtime.removeCachedModel("music-planner-high-quality");
await runtime.clearCache();
```

## `std::bad_alloc`

This is an allocation failure, not a generic model-quality error. Reduce the
duration, close other GPU-heavy tabs, use the standard audio profile, or remove
stale origin data. The production planner and audio pipeline run in separate
Workers so their largest sessions are not resident together.

## ONNX execution-provider warnings

Warnings that some nodes were assigned to CPU are expected for shape/control
operators. Evaluate the measured stage timing before treating these warnings
as a performance defect. The transformer, DiT, and FP32 VAE tensor workloads
remain configured for WebGPU.

## Progress appears stationary

Normalized progress advances at stage boundaries, model-download fractions,
semantic-code counts, sampler steps, VAE chunks, and batch items. Session
creation or shader compilation may take time without an internal callback, so
the number can temporarily plateau. It remains monotonic and is not an ETA.

## Lyrics or chorus are skipped

The lyric word budget is tied to duration. Use `assessLyricDuration()` or set
`autoDuration: true`; a 30-second output cannot reliably sing an arbitrarily
long verse and chorus. Keep the music prompt explicitly vocal and start quality
testing with Euler. Euler SDE is intentionally rejected for vocals.

For the closest browser path to the backend conditioning structure, use:

```ts
await runtime.generate({
  prompt: "Expressive lead vocal, clear phrasing, polished full-band mix",
  writeLyrics: true,
  vocalLanguage: "en",
  durationSeconds: 30,
  autoDuration: true,
  plannerQuality: "high-quality",
  sampler: "euler",
});
```

## Slow high-quality planning

Inspect `planner-profile` updates. Warm semantic generation should not issue one
remote embedding request per audio token. The expected profile shows persistent
normal-text embedding hits and generated-token `head-row reuses`. If remote
range requests recur, confirm the page is using the same origin and that Cache
Storage was not cleared.

## VAE fallback

The qualified path uses the FP32 VAE on strict WebGPU. If session creation or a
chunk fails and `allowWasmFallback` is enabled, every VAE chunk is restarted on
WASM for consistency. This fallback is much slower. Check the `VAE execution
provider` diagnostic before comparing performance.
