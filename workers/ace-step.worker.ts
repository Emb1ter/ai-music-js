/// <reference lib="webworker" />

import { PreTrainedTokenizer } from "@huggingface/transformers";
import { unzipSync } from "fflate";
import * as ort from "onnxruntime-web/webgpu";
import {
  ALL_ASSETS,
  CACHE_NAME,
  DIT_ATTENTION_HEADS,
  INFERENCE_STEPS,
  INSTRUMENTAL_LYRIC_PROMPT,
  LATENT_CHANNELS,
  LATENT_FRAME_RATE,
  MAX_DURATION_SECONDS,
  MIN_DURATION_SECONDS,
  MODEL_DOWNLOAD_BYTES,
  MODEL_NAME,
  ORT_WASM_FILE,
  ORT_WASM_MODULE_FILE,
  SAMPLE_RATE,
  SUPPORT_ASSETS,
  TOTAL_DOWNLOAD_BYTES,
  TURBO_SHIFT,
  buildCaptionPrompt,
  durationToAudioFrames,
  durationToLatentFrames,
  graphById,
  type DownloadAsset,
  type GraphId,
} from "@/lib/model-manifest";
import { deterministicNormal } from "@/lib/prng";
import { createTurboSchedule, eulerFlowStep } from "@/lib/scheduler";
import {
  assertShape,
  tensorSummary,
  type TensorSummary,
} from "@/lib/tensor-diagnostics";
import type {
  WorkerAssetConfig,
  WorkerRequest,
  WorkerUpdate,
} from "@/lib/worker-protocol";
import { encodeStereoWav } from "@/lib/wav";

declare const self: DedicatedWorkerGlobalScope;

type GpuAdapterInfo = {
  vendor?: string;
  architecture?: string;
  device?: string;
  description?: string;
};

type GpuAdapter = {
  info?: GpuAdapterInfo;
  limits: {
    maxBufferSize: number;
    maxStorageBufferBindingSize: number;
  };
};

type GpuNavigator = WorkerNavigator & {
  gpu?: {
    requestAdapter: (options?: {
      powerPreference?: "low-power" | "high-performance";
    }) => Promise<GpuAdapter | null>;
  };
  storage?: StorageManager & {
    getDirectory?: () => Promise<FileSystemDirectoryHandle>;
  };
};

type NumericTensor = ort.Tensor;

const post = (message: WorkerUpdate, transfer: Transferable[] = []) => {
  self.postMessage(message, transfer);
};

const stageStarted = new Map<string, number>();
const timings: Record<string, number> = {};
const trace: TensorSummary[] = [];
let estimatedPeakBytes = 0;
let runtimeAssets: WorkerAssetConfig = {};

const updateEstimatedPeak = (bytes: number) => {
  estimatedPeakBytes = Math.max(estimatedPeakBytes, bytes);
};

const startStage = (stage: string, detail: string) => {
  const startedAt = performance.now();
  stageStarted.set(stage, startedAt);
  post({ type: "stage", stage, detail, startedAt });
};

const endStage = (stage: string) => {
  const startedAt = stageStarted.get(stage);
  if (startedAt === undefined) {
    return;
  }
  const milliseconds = performance.now() - startedAt;
  timings[stage] = (timings[stage] ?? 0) + milliseconds;
  stageStarted.delete(stage);
  post({ type: "timing", stage, milliseconds });
};

const recordTensor = (
  name: string,
  dims: readonly number[],
  data: ArrayLike<number>,
) => {
  const summary = tensorSummary(name, dims, data);
  trace.push(summary);
  post({ type: "trace", summary });
};

const asFloat32 = (
  tensor: NumericTensor,
  name: string,
): Float32Array => {
  const data = tensor.data;
  if (data instanceof Float32Array) {
    return new Float32Array(data);
  }
  if (
    data instanceof BigInt64Array ||
    data instanceof BigUint64Array
  ) {
    throw new TypeError(`${name} unexpectedly returned an int64 tensor.`);
  }
  if (Array.isArray(data)) {
    throw new TypeError(`${name} unexpectedly returned a non-numeric tensor.`);
  }
  return Float32Array.from(data as ArrayLike<number>, Number);
};

const postDownload = (
  asset: DownloadAsset,
  loaded: number,
  total: number,
  cached: boolean,
) => {
  post({
    type: "download",
    assetId: asset.id,
    group: asset.group,
    label: asset.label,
    loaded,
    total,
    cached,
  });
};

const directoryUrl = (value: string) =>
  value.endsWith("/") ? value : `${value}/`;

const resolvedAssetUrl = (asset: DownloadAsset) => {
  const baseUrl =
    runtimeAssets.allAssetsBaseUrl ??
    (asset.url.startsWith("/models/")
      ? runtimeAssets.modelBaseUrl
      : undefined);
  if (!baseUrl) {
    return new URL(asset.url, self.location.origin).href;
  }
  const configured = new URL(
    asset.fileName,
    new URL(directoryUrl(baseUrl), self.location.href),
  );
  const pinned = new URL(asset.url, self.location.origin);
  configured.search = pinned.search;
  return configured.href;
};

const fetchAsset = async (asset: DownloadAsset) => {
  const response = await fetch(
    new Request(resolvedAssetUrl(asset), {
      mode: "cors",
      credentials: "omit",
    }),
  );
  if (!response.ok) {
    throw new Error(
      `Download failed for ${asset.fileName}: HTTP ${response.status} ${response.statusText}`,
    );
  }
  return response;
};

const opfsDirectory = async () => {
  const storage = (navigator as GpuNavigator).storage;
  if (!storage?.getDirectory) {
    throw new Error(
      "Origin Private File System storage is unavailable. Large ACE-Step weight files cannot be cached safely in this browser.",
    );
  }
  const root = await storage.getDirectory();
  return root.getDirectoryHandle(CACHE_NAME, { create: true });
};

const opfsResponseBlob = async (asset: DownloadAsset) => {
  const directory = await opfsDirectory();
  try {
    const existingHandle = await directory.getFileHandle(asset.fileName);
    const existingFile = await existingHandle.getFile();
    if (existingFile.size === asset.bytes) {
      postDownload(asset, asset.bytes, asset.bytes, true);
      return existingFile;
    }
    await directory.removeEntry(asset.fileName);
  } catch (error) {
    if (!(error instanceof DOMException && error.name === "NotFoundError")) {
      throw new Error(
        `Could not inspect persistent storage for ${asset.fileName}: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
  }

  const response = await fetchAsset(asset);
  if (!response.body) {
    throw new Error(
      `Streaming download is unavailable for ${asset.fileName}; refusing to buffer a multi-gigabyte model file in JavaScript memory.`,
    );
  }
  const reportedSize = Number(response.headers.get("content-length"));
  const total =
    Number.isFinite(reportedSize) && reportedSize > 0
      ? reportedSize
      : asset.bytes;
  const fileHandle = await directory.getFileHandle(asset.fileName, {
    create: true,
  });
  const writable = await fileHandle.createWritable({
    keepExistingData: false,
  });
  const reader = response.body.getReader();
  let loaded = 0;
  let lastReport = 0;

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }
      await writable.write(value);
      loaded += value.byteLength;
      const now = performance.now();
      if (now - lastReport >= 100 || loaded >= total) {
        postDownload(asset, loaded, total, false);
        lastReport = now;
      }
    }
    await writable.close();
  } catch (error) {
    await writable.abort(error).catch(() => undefined);
    await directory.removeEntry(asset.fileName).catch(() => undefined);
    throw new Error(
      `Persistent file write failed for ${asset.fileName} after ${(loaded / 1e9).toFixed(2)} GB. ${error instanceof Error ? error.message : String(error)}`,
    );
  }

  const storedFile = await fileHandle.getFile();
  if (storedFile.size !== asset.bytes) {
    await directory.removeEntry(asset.fileName).catch(() => undefined);
    throw new Error(
      `Stored size mismatch for ${asset.fileName}: received ${storedFile.size} bytes, expected ${asset.bytes}.`,
    );
  }
  postDownload(asset, storedFile.size, storedFile.size, false);
  return storedFile;
};

const cacheResponseBlob = async (asset: DownloadAsset) => {
  if (!("caches" in self)) {
    return opfsResponseBlob(asset);
  }
  const cache = await caches.open(CACHE_NAME);
  const request = new Request(resolvedAssetUrl(asset), {
    mode: "cors",
    credentials: "omit",
  });
  let cachedResponse = await cache.match(request);
  if (cachedResponse) {
    postDownload(asset, asset.bytes, asset.bytes, true);
    return cachedResponse.blob();
  }

  const response = await fetchAsset(asset);

  const reportedSize = Number(response.headers.get("content-length"));
  const total =
    Number.isFinite(reportedSize) && reportedSize > 0
      ? reportedSize
      : asset.bytes;

  if (!response.body) {
    const blob = await response.blob();
    try {
      await cache.put(request, new Response(blob, { headers: response.headers }));
    } catch {
      return opfsResponseBlob(asset);
    }
    postDownload(asset, blob.size, total, false);
    return blob;
  }

  const [progressBody, cacheBody] = response.body.tee();
  const cacheWrite = cache.put(
    request,
    new Response(cacheBody, {
      headers: response.headers,
      status: response.status,
      statusText: response.statusText,
    }),
  );
  const reader = progressBody.getReader();
  let loaded = 0;
  let lastReport = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }
    loaded += value.byteLength;
    const now = performance.now();
    if (now - lastReport >= 100 || loaded >= total) {
      postDownload(asset, loaded, total, false);
      lastReport = now;
    }
  }

  try {
    await cacheWrite;
  } catch {
    post({
      type: "diagnostic",
      key: "storage fallback",
      value: `${asset.fileName}: Cache API rejected the entry; using OPFS`,
    });
    return opfsResponseBlob(asset);
  }
  cachedResponse = await cache.match(request);
  if (!cachedResponse) {
    throw new Error(`Cache verification failed for ${asset.fileName}.`);
  }
  postDownload(asset, total, total, false);
  return cachedResponse.blob();
};

const responseBlob = (asset: DownloadAsset) =>
  asset.role === "weights"
    ? opfsResponseBlob(asset)
    : cacheResponseBlob(asset);

const removeOpfsAssets = async (assets: DownloadAsset[]) => {
  const directory = await opfsDirectory();
  for (const asset of assets) {
    await directory.removeEntry(asset.fileName).catch((error) => {
      if (!(error instanceof DOMException && error.name === "NotFoundError")) {
        throw error;
      }
    });
  }
};

const pruneSupersededWeights = async () => {
  const directory = await opfsDirectory();
  const obsoleteFiles = [
    "condition_encoder_q4v2.onnx.data",
    "dit_decoder_q4v2.onnx.data",
    "condition_encoder_q4_fixed.onnx.data",
    "dit_decoder_q4_verified.onnx.data",
    "dit_decoder_xl_turbo_q4.onnx.data",
  ];
  let removed = 0;
  for (const fileName of obsoleteFiles) {
    try {
      await directory.removeEntry(fileName);
      removed += 1;
    } catch (error) {
      if (!(error instanceof DOMException && error.name === "NotFoundError")) {
        throw error;
      }
    }
  }
  if (removed > 0) {
    post({
      type: "diagnostic",
      key: "obsolete weights removed",
      value: removed,
    });
  }
};

const clearPersistentAssets = async () => {
  if ("caches" in self) {
    await caches.delete(CACHE_NAME);
  }
  const storage = (navigator as GpuNavigator).storage;
  if (!storage?.getDirectory) {
    return;
  }
  const root = await storage.getDirectory();
  await root.removeEntry(CACHE_NAME, { recursive: true }).catch((error) => {
    if (!(error instanceof DOMException && error.name === "NotFoundError")) {
      throw error;
    }
  });
};

const missingAssetBytes = async () => {
  let cachedBytes = 0;
  const directory = await opfsDirectory();
  const cache = "caches" in self ? await caches.open(CACHE_NAME) : null;
  for (const asset of ALL_ASSETS) {
    try {
      if (asset.role === "weights") {
        const handle = await directory.getFileHandle(asset.fileName);
        const file = await handle.getFile();
        if (file.size === asset.bytes) {
          cachedBytes += asset.bytes;
        }
        continue;
      }
      const response = await cache?.match(
        new Request(resolvedAssetUrl(asset), {
          mode: "cors",
          credentials: "omit",
        }),
      );
      if (response) {
        const blob = await response.blob();
        if (blob.size === asset.bytes) {
          cachedBytes += asset.bytes;
        }
      }
    } catch (error) {
      if (!(error instanceof DOMException && error.name === "NotFoundError")) {
        throw error;
      }
    }
  }
  return Math.max(0, TOTAL_DOWNLOAD_BYTES - cachedBytes);
};

const sessionFor = async (
  graphId: GraphId,
  allowWasmFallback: boolean,
) => {
  const graph = graphById(graphId);
  const weightBytes = graph.weights.reduce(
    (sum, asset) => sum + asset.bytes,
    0,
  );
  const forceWasmForCorrectness = graphId === "vae";
  startStage(
    `load:${graphId}`,
    `Loading ${graph.label} from persistent browser storage`,
  );

  try {
    const executionProviders: ort.InferenceSession.ExecutionProviderConfig[] =
      forceWasmForCorrectness
        ? ["wasm"]
        : allowWasmFallback
          ? ["webgpu", "wasm"]
          : ["webgpu"];
    post({
      type: "diagnostic",
      key: `${graphId} provider`,
      value: forceWasmForCorrectness
        ? "WASM correctness mode"
        : allowWasmFallback
          ? "WebGPU + WASM fallback"
          : "strict WebGPU",
    });
    for (let attempt = 0; attempt < 2; attempt += 1) {
      try {
        const [graphBlob, ...weightBlobs] = await Promise.all([
          responseBlob(graph.graph),
          ...graph.weights.map(responseBlob),
        ]);
        // ORT Web materializes each Blob before mounting it in WASM.
        updateEstimatedPeak(graph.graph.bytes + weightBytes * 2);
        const graphBytes = await graphBlob.arrayBuffer();
        const session = await ort.InferenceSession.create(graphBytes, {
          executionProviders,
          externalData: graph.weights.map((asset, index) => ({
            path: asset.fileName,
            data: weightBlobs[index],
          })),
          graphOptimizationLevel: "all",
          enableCpuMemArena: true,
          enableMemPattern: true,
          preferredOutputLocation: "cpu",
        });
        endStage(`load:${graphId}`);
        return session;
      } catch (error) {
        const message =
          error instanceof Error ? error.message : String(error);
        const unreadableFile =
          (error instanceof DOMException &&
            error.name === "NotReadableError") ||
          /requested file could not be read|permission problems|notreadable/i.test(
            message,
          );
        if (attempt === 0 && unreadableFile) {
          post({
            type: "diagnostic",
            key: `${graphId} cache recovery`,
            value:
              "Unreadable external-data file removed; downloading fresh chunks",
          });
          await removeOpfsAssets(graph.weights);
          continue;
        }
        throw error;
      }
    }
    throw new Error(`${graph.label} session creation exhausted its retries.`);
  } catch (error) {
    endStage(`load:${graphId}`);
    const operatorHint =
      graph.webGpuOnlyBlockers.length > 0
        ? `Known WebGPU-only blockers: ${graph.webGpuOnlyBlockers.join(", ")}.`
        : undefined;
    const wrapped = new Error(
      `${graph.label} session creation failed. ${operatorHint ?? ""} ${error instanceof Error ? error.message : String(error)}`,
    );
    Object.assign(wrapped, {
      graph: graphId,
      operatorHint,
    });
    throw wrapped;
  }
};

const parseSilenceLatent = async () => {
  const asset = SUPPORT_ASSETS.find(
    (item) => item.id === "conditioning:silence",
  );
  if (!asset) {
    throw new Error("Silence latent asset is missing from the manifest.");
  }
  const archiveBlob = await responseBlob(asset);
  const archive = unzipSync(new Uint8Array(await archiveBlob.arrayBuffer()));
  const tensorEntry = Object.entries(archive).find(([name]) =>
    name.endsWith("/data/0"),
  );
  if (!tensorEntry) {
    throw new Error(
      "silence_latent.pt does not contain the expected PyTorch data/0 tensor.",
    );
  }
  const [, raw] = tensorEntry;
  if (raw.byteLength !== 960_000 * Float32Array.BYTES_PER_ELEMENT) {
    throw new Error(
      `Unexpected silence latent payload: ${raw.byteLength} bytes; expected 3,840,000.`,
    );
  }
  const aligned = raw.byteOffset % Float32Array.BYTES_PER_ELEMENT === 0
    ? raw
    : raw.slice();
  const channelMajor = new Float32Array(
    aligned.buffer,
    aligned.byteOffset,
    aligned.byteLength / Float32Array.BYTES_PER_ELEMENT,
  );

  const slice = (frames: number) => {
    if (frames > 15_000) {
      throw new RangeError("Requested silence latent exceeds the stored 600 seconds.");
    }
    const timeMajor = new Float32Array(frames * LATENT_CHANNELS);
    for (let frame = 0; frame < frames; frame += 1) {
      for (let channel = 0; channel < LATENT_CHANNELS; channel += 1) {
        timeMajor[frame * LATENT_CHANNELS + channel] =
          channelMajor[channel * 15_000 + frame];
      }
    }
    return timeMajor;
  };
  return { slice };
};

const loadTokenizer = async () => {
  const tokenizerAsset = SUPPORT_ASSETS.find(
    (item) => item.id === "tokenizer:json",
  );
  const configAsset = SUPPORT_ASSETS.find(
    (item) => item.id === "tokenizer:config",
  );
  if (!tokenizerAsset || !configAsset) {
    throw new Error("Tokenizer assets are missing from the manifest.");
  }
  const [tokenizerBlob, configBlob] = await Promise.all([
    responseBlob(tokenizerAsset),
    responseBlob(configAsset),
  ]);
  const [tokenizerJson, tokenizerConfig] = await Promise.all([
    tokenizerBlob.text().then((value) => JSON.parse(value)),
    configBlob.text().then((value) => JSON.parse(value)),
  ]);
  return new PreTrainedTokenizer(tokenizerJson, tokenizerConfig);
};

const tokenize = (
  tokenizer: PreTrainedTokenizer,
  text: string,
  maxLength: number,
) => {
  const encoded = tokenizer(text, {
    add_special_tokens: true,
    truncation: true,
    max_length: maxLength,
    padding: false,
  });
  const inputData = encoded.input_ids.data;
  const ids =
    inputData instanceof BigInt64Array
      ? inputData.slice()
      : BigInt64Array.from(inputData as ArrayLike<number>, BigInt);
  if (ids.length === 0) {
    throw new Error("Tokenizer returned an empty sequence.");
  }
  return {
    ids,
    dims: [1, ids.length] as const,
    attentionMask: new Float32Array(ids.length).fill(1),
  };
};

const checkCompatibility = async (latentFrames: number) => {
  const gpuNavigator = navigator as GpuNavigator;
  if (!gpuNavigator.gpu) {
    post({
      type: "compatibility",
      ok: false,
      message:
        "WebGPU is not exposed to this worker. Use current desktop Chrome or Edge over HTTPS or localhost, and check that hardware acceleration is enabled.",
    });
    throw new Error("WebGPU is unavailable in the inference worker.");
  }
  const adapter = await gpuNavigator.gpu.requestAdapter({
    powerPreference: "high-performance",
  });
  if (!adapter) {
    post({
      type: "compatibility",
      ok: false,
      message:
        "A WebGPU adapter could not be created. The browser may have blocklisted this GPU or hardware acceleration may be disabled.",
    });
    throw new Error("WebGPU adapter request returned null.");
  }

  const info = adapter.info ?? {};
  const adapterSummary = {
    vendor: info.vendor ?? "unreported",
    architecture: info.architecture ?? "unreported",
    device: info.device ?? "unreported",
    description: info.description ?? "unreported",
    maxBufferSize: Number(adapter.limits.maxBufferSize),
    maxStorageBufferBindingSize: Number(
      adapter.limits.maxStorageBufferBindingSize,
    ),
  };
  post({
    type: "compatibility",
    ok: true,
    message: "WebGPU adapter acquired inside the inference worker.",
    adapter: adapterSummary,
  });
  const attentionTokens = Math.ceil(latentFrames / 2);
  const largestAttentionBytes =
    DIT_ATTENTION_HEADS *
    attentionTokens *
    attentionTokens *
    Float32Array.BYTES_PER_ELEMENT;
  post({
    type: "diagnostic",
    key: "estimated full-attention tensor",
    value: largestAttentionBytes,
  });
  if (largestAttentionBytes > adapter.limits.maxBufferSize) {
    throw new Error(
      `The selected duration needs an estimated ${(largestAttentionBytes / 1e9).toFixed(2)} GB attention buffer, above this adapter's ${(Number(adapter.limits.maxBufferSize) / 1e9).toFixed(2)} GB WebGPU buffer limit.`,
    );
  }

  try {
    await pruneSupersededWeights();
    const estimate = await gpuNavigator.storage?.estimate();
    if (estimate) {
      const missingBytes = await missingAssetBytes();
      post({
        type: "diagnostic",
        key: "storage quota",
        value: estimate.quota ?? 0,
      });
      post({
        type: "diagnostic",
        key: "storage used",
        value: estimate.usage ?? 0,
      });
      post({
        type: "diagnostic",
        key: "uncached model data",
        value: missingBytes,
      });
      if (
        estimate.quota &&
        estimate.usage !== undefined &&
        estimate.quota - estimate.usage < missingBytes
      ) {
        throw new Error(
          `Insufficient browser storage quota. Need ${(missingBytes / 1e9).toFixed(2)} GB for uncached XL assets, but only ${((estimate.quota - estimate.usage) / 1e9).toFixed(2)} GB is available.`,
        );
      }
    }
    await gpuNavigator.storage?.persist?.();
  } catch (error) {
    if (error instanceof Error && error.message.startsWith("Insufficient")) {
      throw error;
    }
  }
};

const runGeneration = async (
  prompt: string,
  seed: number,
  durationSeconds: number,
  allowWasmFallback: boolean,
  assets: WorkerAssetConfig = {},
) => {
  if (
    !Number.isInteger(durationSeconds) ||
    durationSeconds < MIN_DURATION_SECONDS ||
    durationSeconds > MAX_DURATION_SECONDS
  ) {
    throw new RangeError(
      `Duration must be a whole number from ${MIN_DURATION_SECONDS} to ${MAX_DURATION_SECONDS} seconds.`,
    );
  }
  const latentFrames = durationToLatentFrames(durationSeconds);
  const audioFrames = durationToAudioFrames(durationSeconds);
  trace.length = 0;
  runtimeAssets = assets;
  Object.keys(timings).forEach((key) => delete timings[key]);
  estimatedPeakBytes = 0;
  ort.env.logLevel = "warning";
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.proxy = false;
  ort.env.wasm.wasmPaths = {
    mjs:
      runtimeAssets.wasmModuleUrl ??
      new URL(ORT_WASM_MODULE_FILE, self.location.origin).href,
    wasm:
      runtimeAssets.wasmUrl ??
      new URL(ORT_WASM_FILE, self.location.origin).href,
  };
  ort.env.webgpu.powerPreference = "high-performance";

  startStage("compatibility", "Checking worker-side WebGPU and storage limits");
  await checkCompatibility(latentFrames);
  endStage("compatibility");
  post({
    type: "diagnostic",
    key: "model",
    value: MODEL_NAME,
  });
  post({
    type: "diagnostic",
    key: "model download",
    value: MODEL_DOWNLOAD_BYTES,
  });
  post({
    type: "diagnostic",
    key: "total download",
    value: TOTAL_DOWNLOAD_BYTES,
  });
  post({
    type: "diagnostic",
    key: "execution policy",
    value: allowWasmFallback
      ? "WebGPU text/DiT + WASM fallback; VAE forced to WASM"
      : "strict WebGPU except VAE correctness mode",
  });
  post({
    type: "diagnostic",
    key: "requested duration",
    value: `${durationSeconds} seconds`,
  });
  post({
    type: "diagnostic",
    key: "latent frames",
    value: latentFrames,
  });

  startStage("tokenization", "Loading tokenizer and building ACE-Step prompts");
  const [tokenizer, silence] = await Promise.all([
    loadTokenizer(),
    parseSilenceLatent(),
  ]);
  const textTokens = tokenize(
    tokenizer,
    buildCaptionPrompt(prompt, durationSeconds),
    256,
  );
  const lyricTokens = tokenize(tokenizer, INSTRUMENTAL_LYRIC_PROMPT, 2048);
  endStage("tokenization");
  post({
    type: "diagnostic",
    key: "caption tokens",
    value: textTokens.ids.length,
  });
  post({
    type: "diagnostic",
    key: "lyric tokens",
    value: lyricTokens.ids.length,
  });

  startStage("text-encoding", "Encoding the caption with Qwen3");
  const textSession = await sessionFor("text-encoder", allowWasmFallback);
  const textOutputs = await textSession.run({
    input_ids: new ort.Tensor("int64", textTokens.ids, textTokens.dims),
  });
  const textTensor = textOutputs.hidden_states as NumericTensor;
  const textHidden = asFloat32(textTensor, "text hidden states");
  const textDims = [...textTensor.dims];
  assertShape(
    textDims,
    [1, textTokens.ids.length, 1024],
    "text hidden states",
  );
  recordTensor("text_hidden_states", textDims, textHidden);
  textTensor.dispose();
  await textSession.release();
  endStage("text-encoding");

  startStage("lyric-embedding", "Embedding the instrumental lyric marker");
  const embeddingSession = await sessionFor(
    "lyric-embedding",
    allowWasmFallback,
  );
  const embeddingOutputs = await embeddingSession.run({
    input_ids: new ort.Tensor("int64", lyricTokens.ids, lyricTokens.dims),
  });
  const lyricTensor = embeddingOutputs.hidden_states as NumericTensor;
  const lyricHidden = asFloat32(lyricTensor, "lyric embeddings");
  const lyricDims = [...lyricTensor.dims];
  assertShape(
    lyricDims,
    [1, lyricTokens.ids.length, 1024],
    "lyric embeddings",
  );
  recordTensor("lyric_hidden_states", lyricDims, lyricHidden);
  lyricTensor.dispose();
  await embeddingSession.release();
  endStage("lyric-embedding");

  startStage(
    "condition-packing",
    "Projecting and packing caption, instrumental lyric, timbre, and source context",
  );
  const conditionSession = await sessionFor(
    "condition-encoder",
    allowWasmFallback,
  );
  const sourceLatents = silence.slice(latentFrames);
  const referenceLatents = silence.slice(750);
  const chunkMask = new Float32Array(
    latentFrames * LATENT_CHANNELS,
  ).fill(1);
  const emptyLmHints = new Float32Array(
    latentFrames * LATENT_CHANNELS,
  );
  recordTensor(
    "source_silence_latents",
    [1, latentFrames, LATENT_CHANNELS],
    sourceLatents,
  );

  const conditionOutputs = await conditionSession.run({
    text_hidden_states: new ort.Tensor("float32", textHidden, textDims),
    text_attention_mask: new ort.Tensor(
      "float32",
      textTokens.attentionMask,
      textTokens.dims,
    ),
    lyric_hidden_states: new ort.Tensor("float32", lyricHidden, lyricDims),
    lyric_attention_mask: new ort.Tensor(
      "float32",
      lyricTokens.attentionMask,
      lyricTokens.dims,
    ),
    refer_audio_acoustic_hidden_states_packed: new ort.Tensor(
      "float32",
      referenceLatents,
      [1, 750, LATENT_CHANNELS],
    ),
    refer_audio_order_mask: new ort.Tensor(
      "int64",
      new BigInt64Array([0n]),
      [1],
    ),
    src_latents: new ort.Tensor(
      "float32",
      sourceLatents,
      [1, latentFrames, LATENT_CHANNELS],
    ),
    chunk_masks: new ort.Tensor(
      "float32",
      chunkMask,
      [1, latentFrames, LATENT_CHANNELS],
    ),
    is_covers: new ort.Tensor("float32", new Float32Array([0]), [1]),
    precomputed_lm_hints_25hz: new ort.Tensor(
      "float32",
      emptyLmHints,
      [1, latentFrames, LATENT_CHANNELS],
    ),
  });
  const encoderTensor =
    conditionOutputs.encoder_hidden_states as NumericTensor;
  const contextTensor = conditionOutputs.context_latents as NumericTensor;
  const encoderHidden = asFloat32(
    encoderTensor,
    "packed encoder hidden states",
  );
  const contextLatents = asFloat32(contextTensor, "context latents");
  const encoderDims = [...encoderTensor.dims];
  const contextDims = [...contextTensor.dims];
  if (
    encoderDims[0] !== 1 ||
    encoderDims[2] !== 2048 ||
    encoderDims[1] < textTokens.ids.length
  ) {
    throw new Error(
      `Packed condition shape is invalid: [${encoderDims.join(", ")}].`,
    );
  }
  assertShape(
    contextDims,
    [1, latentFrames, 128],
    "context latents",
  );
  recordTensor("encoder_hidden_states", encoderDims, encoderHidden);
  recordTensor("context_latents", contextDims, contextLatents);
  encoderTensor.dispose();
  contextTensor.dispose();
  conditionOutputs.encoder_attention_mask?.dispose();
  await conditionSession.release();
  endStage("condition-packing");

  startStage("initial-latent", `Generating deterministic Gaussian noise seed ${seed}`);
  let latent = deterministicNormal(
    latentFrames * LATENT_CHANNELS,
    seed,
  );
  recordTensor(
    "initial_latent",
    [1, latentFrames, LATENT_CHANNELS],
    latent,
  );
  endStage("initial-latent");

  startStage(
    "flow-matching",
    "Running eight ACE-Step XL Turbo Euler evaluations",
  );
  const ditSession = await sessionFor("dit", allowWasmFallback);
  const schedule = createTurboSchedule(INFERENCE_STEPS, TURBO_SHIFT);
  for (const step of schedule) {
    const stepStage = `euler:${step.index + 1}`;
    startStage(
      stepStage,
      `DiT ${step.index + 1}/${schedule.length} · t=${step.current.toFixed(6)} → ${step.next.toFixed(6)}`,
    );
    const outputs = await ditSession.run({
      hidden_states: new ort.Tensor(
        "float32",
        latent,
        [1, latentFrames, LATENT_CHANNELS],
      ),
      timestep: new ort.Tensor(
        "float32",
        new Float32Array([step.current]),
        [1],
      ),
      encoder_hidden_states: new ort.Tensor(
        "float32",
        encoderHidden,
        encoderDims,
      ),
      context_latents: new ort.Tensor(
        "float32",
        contextLatents,
        contextDims,
      ),
    });
    const velocityTensor = outputs.velocity as NumericTensor;
    const velocity = asFloat32(velocityTensor, "DiT velocity");
    assertShape(
      velocityTensor.dims,
      [1, latentFrames, LATENT_CHANNELS],
      `velocity step ${step.index + 1}`,
    );
    recordTensor(
      `velocity_${step.index + 1}`,
      velocityTensor.dims,
      velocity,
    );
    latent = eulerFlowStep(latent, velocity, step.delta);
    recordTensor(
      `latent_${step.index + 1}`,
      [1, latentFrames, LATENT_CHANNELS],
      latent,
    );
    velocityTensor.dispose();
    endStage(stepStage);
  }
  await ditSession.release();
  endStage("flow-matching");

  const rms = (
    values: ArrayLike<number>,
    start: number,
    end: number,
  ) => {
    let sumSquares = 0;
    for (let index = start; index < end; index += 1) {
      const value = Number(values[index]);
      if (!Number.isFinite(value)) {
        return Number.NaN;
      }
      sumSquares += value * value;
    }
    return Math.sqrt(sumSquares / Math.max(1, end - start));
  };
  const firstTwoLatentRms = rms(
    latent,
    0,
    Math.min(2, durationSeconds) * LATENT_FRAME_RATE * LATENT_CHANNELS,
  );
  const remainingLatentRms = rms(
    latent,
    Math.min(2, durationSeconds) * LATENT_FRAME_RATE * LATENT_CHANNELS,
    latent.length,
  );
  post({
    type: "diagnostic",
    key: "final latent RMS 0-2s",
    value: firstTwoLatentRms.toFixed(6),
  });
  post({
    type: "diagnostic",
    key: `final latent RMS 2-${durationSeconds}s`,
    value: remainingLatentRms.toFixed(6),
  });
  if (
    !Number.isFinite(firstTwoLatentRms) ||
    !Number.isFinite(remainingLatentRms) ||
    remainingLatentRms < 0.05
  ) {
    throw new Error(
      `DiT output collapsed before VAE decoding: latent RMS is ${firstTwoLatentRms.toFixed(6)} for 0-2s and ${remainingLatentRms.toFixed(6)} for 2-${durationSeconds}s.`,
    );
  }

  startStage(
    "vae-decode",
    `Decoding ${latentFrames} latent frames to ${durationSeconds} seconds of stereo PCM`,
  );
  const channelMajorLatent = new Float32Array(latent.length);
  for (let frame = 0; frame < latentFrames; frame += 1) {
    for (let channel = 0; channel < LATENT_CHANNELS; channel += 1) {
      channelMajorLatent[channel * latentFrames + frame] =
        latent[frame * LATENT_CHANNELS + channel];
    }
  }
  const vaeSession = await sessionFor("vae", allowWasmFallback);
  const vaeOutputs = await vaeSession.run({
    latents: new ort.Tensor(
      "float32",
      channelMajorLatent,
      [1, LATENT_CHANNELS, latentFrames],
    ),
  });
  const waveformTensor = vaeOutputs.waveform as NumericTensor;
  const waveform = asFloat32(waveformTensor, "decoded waveform");
  assertShape(waveformTensor.dims, [1, 2, audioFrames], "decoded waveform");
  recordTensor("waveform", waveformTensor.dims, waveform);
  const channelWindowRms = (
    channel: number,
    startSecond: number,
    endSecond: number,
  ) =>
    rms(
      waveform,
      channel * audioFrames + startSecond * SAMPLE_RATE,
      channel * audioFrames + endSecond * SAMPLE_RATE,
    );
  const tailStartSecond = Math.max(2, durationSeconds - 2);
  const waveformWindows = [
    [0, 2],
    [2, tailStartSecond],
    [tailStartSecond, durationSeconds],
  ] as const;
  for (const [startSecond, endSecond] of waveformWindows) {
    const leftRms = channelWindowRms(0, startSecond, endSecond);
    const rightRms = channelWindowRms(1, startSecond, endSecond);
    post({
      type: "diagnostic",
      key: `waveform RMS ${startSecond}-${endSecond}s`,
      value: `${leftRms.toFixed(6)} L / ${rightRms.toFixed(6)} R`,
    });
    if (
      startSecond === 2 &&
      endSecond > startSecond &&
      (
        !Number.isFinite(leftRms) ||
        !Number.isFinite(rightRms) ||
        Math.max(leftRms, rightRms) < 0.002
      )
    ) {
      waveformTensor.dispose();
      await vaeSession.release();
      throw new Error(
        `VAE output collapsed after two seconds: ${startSecond}-${endSecond}s RMS is ${leftRms.toFixed(6)} L / ${rightRms.toFixed(6)} R.`,
      );
    }
  }
  waveformTensor.dispose();
  await vaeSession.release();
  endStage("vae-decode");

  startStage("audio-packaging", "Encoding stereo PCM as a playable WAV");
  const left = waveform.slice(0, audioFrames);
  const right = waveform.slice(audioFrames, audioFrames * 2);
  const wav = encodeStereoWav(left, right, SAMPLE_RATE);
  updateEstimatedPeak(
    graphById("vae").weights.reduce(
      (sum, asset) => sum + asset.bytes,
      0,
    ) +
      waveform.byteLength +
      wav.byteLength +
      encoderHidden.byteLength +
      contextLatents.byteLength,
  );
  endStage("audio-packaging");

  post(
    {
      type: "complete",
      wav,
      left: left.buffer,
      right: right.buffer,
      sampleRate: SAMPLE_RATE,
      durationSeconds,
      latentFrames,
      trace: [...trace],
      timings: { ...timings },
      estimatedPeakBytes,
    },
    [wav, left.buffer, right.buffer],
  );
};

self.onmessage = async (event: MessageEvent<WorkerRequest>) => {
  if (event.data.type === "clear-cache") {
    await clearPersistentAssets();
    post({ type: "cache-cleared" });
    return;
  }

  try {
    await runGeneration(
      event.data.prompt,
      event.data.seed,
      event.data.durationSeconds,
      event.data.allowWasmFallback,
      event.data.assets,
    );
  } catch (error) {
    const details = error as Error & {
      graph?: string;
      operatorHint?: string;
    };
    post({
      type: "error",
      stage: [...stageStarted.keys()].at(-1) ?? "unknown",
      message: details.message ?? String(error),
      graph: details.graph,
      operatorHint: details.operatorHint,
      stack: details.stack,
    });
  }
};
