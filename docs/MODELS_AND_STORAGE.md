# Models and browser storage

The npm package contains JavaScript, type declarations, Workers, ONNX Runtime
WebAssembly, licenses, and documentation. Model weights are downloaded at
runtime from immutable Hugging Face revisions.

## Selectable model paths

| Option | Download size | Purpose |
|---|---:|---|
| `audioQuality: "standard"` | 5,626,494,229 bytes | INT4 XL condition encoder and DiT plus shared support models and FP32 VAE |
| `audioQuality: "high"` | 8,004,092,572 bytes | INT8 XL condition encoder and DiT plus the same support models and FP32 VAE |
| `plannerQuality: "high-quality"` | +4,633,150,982 bytes | INT8-weight/FP32-compute ACE 5 Hz planner and FP32 WebGPU head |
| `writeLyrics: true` | +489,166,749 bytes | Qwen3.5 0.8B INT4 lyric writer |

The normal high-quality combination is therefore 10,259,645,211 bytes with
standard audio or 12,637,243,554 bytes with high-precision audio. Browser quota
and temporary session allocations require additional headroom.

## Pinned repositories

- XL Turbo audio and FP32 VAE:
  `emb1ter/ACE-Step-v1.5-XL-Turbo-ONNX-WebGPU` at
  `cf185389395b3a725d948a59262f3ab4be4b0ad8`
- High-quality 5 Hz planner:
  `emb1ter/ACE-Step-v1.5-5Hz-LM-4B-FP32-ONNX-WebGPU` at
  `ad1eba6d99ea99d7cd2db7f7fb14275634016777`
- Qwen3.5 lyric writer:
  `onnx-community/Qwen3.5-0.8B-Text-ONNX` at
  `1e45daba048899e7f771657ada617ec49350aa91`
- Shared experimental ONNX assets:
  `shreyask/ACE-Step-v1.5-ONNX` at
  `bdabfb5684fd70fcc76f98cbb51bb9ebc47ee342`

## Storage layout

- Large audio graph weights prefer the Origin Private File System (OPFS).
- Smaller audio/support files use Cache Storage when practical.
- Transformers.js and planner files use the isolated
  `ai-music-js-transformers-v1` Cache Storage bucket.
- Storage is scoped to the exact origin. Different ports, production domains,
  and ngrok hostnames do not share cached data.

The high-quality planner evaluates each file independently against current
quota and reserves room for missing audio assets. A shard may be streamed for
the current run when it cannot safely be persisted; other fitting shards can
still be cached.

## Sparse planner embeddings

The planner ties a 217,204-row FP32 input embedding to its output head. Storing
the complete 2.22 GB embedding file would duplicate the 64,000 audio-code rows
already present in the 655 MB FP32 head.

The library instead:

1. batch-prefetches prompt and constrained-metadata rows;
2. stores used normal-text rows as individual 10 KiB Cache API responses;
3. copies each selected audio token's exact FP32 row from the resident WebGPU
   head; and
4. injects that row into the next autoregressive body call.

This removes the former one-remote-range-request-per-semantic-token bottleneck.
Warm runs use persistent normal-text rows, while generated audio-token rows do
not consume duplicate persistent storage.

## Inspection and removal

```ts
const inventory = await runtime.listCachedModels();
console.log(inventory.usageBytes, inventory.quotaBytes, inventory.availableBytes);

await runtime.removeCachedModel("music-planner-high-quality");
await runtime.removeCachedModel("lyrics-writer");
await runtime.removeCachedModel("dit");
await runtime.clearCache();
```

`listCachedModels()` reports individual assets, storage backend, completeness,
origin usage/quota, and sparse planner-row storage. Removing the high-quality
planner also removes its synthetic sparse-row cache entries.

Chromium determines per-origin quota from browser policy, free disk space,
engagement, persistence, and browsing mode. The library requests persistent
storage but cannot force Chromium to grant a larger quota. Use a stable HTTPS
or localhost origin and avoid Incognito for multi-gigabyte model caches.
