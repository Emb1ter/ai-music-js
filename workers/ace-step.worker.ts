/// <reference lib="webworker" />

import { PreTrainedTokenizer } from "@huggingface/transformers";
import { unzipSync } from "fflate";
import * as ort from "onnxruntime-web/webgpu";
import {
  ALL_ASSETS,
  CACHE_NAME,
  DIT_ATTENTION_HEADS,
  INFERENCE_STEPS,
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
  VAE_UPSAMPLE_FACTOR,
  buildCaptionPrompt,
  buildLyricPrompt,
  durationToAudioFrames,
  durationToLatentFrames,
  graphById,
  hasVocalPromptConflict,
  isInstrumentalLyrics,
  type DownloadAsset,
  type GraphId,
} from "@/lib/model-manifest";
import { applyDcw } from "@/lib/dcw";
import {
  deterministicNormal,
  deterministicNormalStream,
} from "@/lib/prng";
import {
  createTurboSchedule,
  eulerFlowStep,
  eulerSdeFlowStep,
  heunFlowStep,
  predictCleanSample,
} from "@/lib/scheduler";
import type {
  ResolvedDcwOptions,
  SamplerMode,
} from "@/lib/generation-options";
import {
  assertShape,
  tensorSummary,
  type TensorSummary,
} from "@/lib/tensor-diagnostics";
import { createVaeDecodePlan } from "@/lib/vae-chunking";
import type {
  CacheInventory,
  CachedAssetInfo,
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
const MIN_STORAGE_HEADROOM_BYTES = 512_000_000;
const PER_FILE_STORAGE_HEADROOM_BYTES = 128_000_000;

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

const storageEstimate = async () => {
  try {
    return await (navigator as GpuNavigator).storage?.estimate();
  } catch {
    return undefined;
  }
};

const storageNumbers = (estimate?: StorageEstimate) => {
  const quota = estimate?.quota;
  const usage = estimate?.usage;
  const available =
    quota !== undefined && usage !== undefined
      ? Math.max(0, quota - usage)
      : undefined;
  return { quota, usage, available };
};

const storageFailureDetail = async () => {
  const { quota, usage, available } = storageNumbers(
    await storageEstimate(),
  );
  if (
    quota === undefined ||
    usage === undefined ||
    available === undefined
  ) {
    return "";
  }
  return ` Browser storage reports ${(usage / 1e9).toFixed(2)} GB used, ${(quota / 1e9).toFixed(2)} GB quota, and ${(available / 1e9).toFixed(2)} GB available for ${self.location.origin}.`;
};

const assertFileStorageCapacity = async (asset: DownloadAsset) => {
  const { available } = storageNumbers(await storageEstimate());
  const required = asset.bytes + PER_FILE_STORAGE_HEADROOM_BYTES;
  if (available !== undefined && available < required) {
    throw new Error(
      `Insufficient browser storage for ${asset.fileName}. This file needs ${(asset.bytes / 1e9).toFixed(2)} GB plus temporary-write headroom, but only ${(available / 1e9).toFixed(2)} GB is available. Remove cached model components or all model data, free disk space, and avoid Incognito mode. Browser storage is separate for every hostname, including each ngrok URL.`,
    );
  }
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

  await assertFileStorageCapacity(asset);
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
    await reader.cancel(error).catch(() => undefined);
    await writable.abort(error).catch(() => undefined);
    await directory.removeEntry(asset.fileName).catch(() => undefined);
    const storageDetail = await storageFailureDetail();
    throw new Error(
      `Persistent file write failed for ${asset.fileName} after ${(loaded / 1e9).toFixed(2)} GB. ${error instanceof Error ? error.message : String(error)}${storageDetail} Remove cached model components or all model data, free disk space, and avoid Incognito mode before retrying.`,
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
    const cachedBlob = await cachedResponse.blob();
    if (cachedBlob.size === asset.bytes) {
      postDownload(asset, asset.bytes, asset.bytes, true);
      return cachedBlob;
    }
    await cache.delete(request);
    post({
      type: "diagnostic",
      key: "partial cache entry removed",
      value: `${asset.fileName}: ${cachedBlob.size} of ${asset.bytes} bytes`,
    });
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

const assetCacheRequest = (asset: DownloadAsset) =>
  new Request(resolvedAssetUrl(asset), {
    mode: "cors",
    credentials: "omit",
  });

const inspectCachedAsset = async (
  asset: DownloadAsset,
  directory: FileSystemDirectoryHandle,
  cache: Cache | null,
): Promise<CachedAssetInfo> => {
  let storedBytes = 0;
  let storage: CachedAssetInfo["storage"] = null;

  try {
    const handle = await directory.getFileHandle(asset.fileName);
    const file = await handle.getFile();
    storedBytes = file.size;
    storage = "opfs";
    if (file.size === asset.bytes) {
      return {
        id: asset.id,
        group: asset.group,
        label: asset.label,
        fileName: asset.fileName,
        role: asset.role,
        expectedBytes: asset.bytes,
        storedBytes: file.size,
        cached: true,
        storage,
      };
    }
  } catch (error) {
    if (!(error instanceof DOMException && error.name === "NotFoundError")) {
      throw error;
    }
  }

  const response = await cache?.match(assetCacheRequest(asset));
  if (response) {
    const cacheBytes = (await response.blob()).size;
    if (cacheBytes >= storedBytes) {
      storedBytes = cacheBytes;
      storage = "cache-api";
    }
    if (cacheBytes === asset.bytes) {
      return {
        id: asset.id,
        group: asset.group,
        label: asset.label,
        fileName: asset.fileName,
        role: asset.role,
        expectedBytes: asset.bytes,
        storedBytes: cacheBytes,
        cached: true,
        storage: "cache-api",
      };
    }
  }

  return {
    id: asset.id,
    group: asset.group,
    label: asset.label,
    fileName: asset.fileName,
    role: asset.role,
    expectedBytes: asset.bytes,
    storedBytes,
    cached: false,
    storage,
  };
};

const cacheInventory = async (): Promise<CacheInventory> => {
  const directory = await opfsDirectory();
  const cache = "caches" in self ? await caches.open(CACHE_NAME) : null;
  const assets: CachedAssetInfo[] = [];
  for (const asset of ALL_ASSETS) {
    assets.push(await inspectCachedAsset(asset, directory, cache));
  }

  const grouped = new Map<string, CachedAssetInfo[]>();
  for (const asset of assets) {
    const group = grouped.get(asset.group) ?? [];
    group.push(asset);
    grouped.set(asset.group, group);
  }
  const models = [...grouped.entries()].map(([id, groupAssets]) => {
    const expectedBytes = groupAssets.reduce(
      (sum, asset) => sum + asset.expectedBytes,
      0,
    );
    const storedBytes = groupAssets.reduce(
      (sum, asset) => sum + Math.min(asset.storedBytes, asset.expectedBytes),
      0,
    );
    const complete = groupAssets.every((asset) => asset.cached);
    return {
      id,
      label: groupAssets[0]?.label ?? id,
      expectedBytes,
      storedBytes,
      complete,
      partial: !complete && storedBytes > 0,
      assets: groupAssets,
    };
  });
  const expectedBytes = assets.reduce(
    (sum, asset) => sum + asset.expectedBytes,
    0,
  );
  const storedBytes = assets.reduce(
    (sum, asset) => sum + Math.min(asset.storedBytes, asset.expectedBytes),
    0,
  );
  const readyBytes = assets.reduce(
    (sum, asset) => sum + (asset.cached ? asset.expectedBytes : 0),
    0,
  );
  const estimate = await storageEstimate();
  const { quota, usage, available } = storageNumbers(estimate);
  let persisted: boolean | undefined;
  try {
    persisted = await (navigator as GpuNavigator).storage?.persisted?.();
  } catch {
    persisted = undefined;
  }
  return {
    origin: self.location.origin,
    cacheName: CACHE_NAME,
    expectedBytes,
    storedBytes,
    readyBytes,
    missingBytes: Math.max(0, expectedBytes - readyBytes),
    usageBytes: usage,
    quotaBytes: quota,
    availableBytes: available,
    persisted,
    models,
  };
};

const removeCachedAssets = async (assets: DownloadAsset[]) => {
  const directory = await opfsDirectory();
  const cache = "caches" in self ? await caches.open(CACHE_NAME) : null;
  for (const asset of assets) {
    await directory.removeEntry(asset.fileName).catch((error) => {
      if (!(error instanceof DOMException && error.name === "NotFoundError")) {
        throw error;
      }
    });
    await cache?.delete(assetCacheRequest(asset));
  }
};

const removeCachedModel = async (modelId: string) => {
  const assets = ALL_ASSETS.filter((asset) => asset.group === modelId);
  if (assets.length === 0) {
    throw new RangeError(`Unknown cached model component: ${modelId}`);
  }
  const before = await cacheInventory();
  const beforeModel = before.models.find((model) => model.id === modelId);
  await removeCachedAssets(assets);
  return beforeModel?.storedBytes ?? 0;
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
        const graphBlob = await responseBlob(graph.graph);
        const weightBlobs: Blob[] = [];
        for (const asset of graph.weights) {
          weightBlobs.push(await responseBlob(asset));
        }
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

  let persisted: boolean | undefined;
  try {
    persisted = await gpuNavigator.storage?.persisted?.();
    if (!persisted) {
      persisted = await gpuNavigator.storage?.persist?.();
    }
  } catch {
    persisted = undefined;
  }
  post({
    type: "diagnostic",
    key: "persistent storage",
    value:
      persisted === undefined
        ? "unsupported"
        : persisted
          ? "granted"
          : "not granted",
  });

  await pruneSupersededWeights();
  const estimate = await gpuNavigator.storage?.estimate();
  if (estimate) {
    const inventory = await cacheInventory();
    const missingBytes = inventory.missingBytes;
    const reclaimablePartialBytes = inventory.models.reduce(
      (modelSum, model) =>
        modelSum +
        model.assets.reduce(
          (assetSum, asset) =>
            assetSum + (asset.cached ? 0 : asset.storedBytes),
          0,
        ),
      0,
    );
    const { quota, usage, available } = storageNumbers(estimate);
    const effectiveAvailable =
      available === undefined
        ? undefined
        : available + reclaimablePartialBytes;
    post({
      type: "diagnostic",
      key: "storage quota",
      value: quota ?? 0,
    });
    post({
      type: "diagnostic",
      key: "storage used",
      value: usage ?? 0,
    });
    post({
      type: "diagnostic",
      key: "uncached model data",
      value: missingBytes,
    });
    const headroom =
      missingBytes > 0
        ? Math.max(
            MIN_STORAGE_HEADROOM_BYTES,
            Math.ceil(missingBytes * 0.05),
          )
        : 0;
    post({
      type: "diagnostic",
      key: "storage write headroom",
      value: headroom,
    });
    if (reclaimablePartialBytes > 0) {
      post({
        type: "diagnostic",
        key: "partial model data reclaimable",
        value: reclaimablePartialBytes,
      });
    }
    if (
      effectiveAvailable !== undefined &&
      effectiveAvailable < missingBytes + headroom
    ) {
      throw new Error(
        `Insufficient browser storage quota for ${self.location.origin}. The uncached XL assets need ${(missingBytes / 1e9).toFixed(2)} GB plus ${(headroom / 1e9).toFixed(2)} GB temporary-write headroom, but only ${(effectiveAvailable / 1e9).toFixed(2)} GB is available after reclaiming partial files. Use the demo cache manager or listCachedModels()/removeCachedModel()/clearCache(), free disk space, and avoid Incognito mode. Each hostname, including each ngrok URL, has separate browser storage.`,
      );
    }
  }
};

const runGeneration = async (
  prompt: string,
  lyrics: string,
  vocalLanguage: string,
  seed: number,
  durationSeconds: number,
  sampler: SamplerMode,
  dcw: ResolvedDcwOptions,
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
  const instrumental = isInstrumentalLyrics(lyrics);
  if (hasVocalPromptConflict(prompt, lyrics)) {
    throw new Error(
      "Vocal lyrics were supplied, but the caption requests an instrumental track without asking for a singer or vocals.",
    );
  }
  if (!instrumental && sampler === "euler-sde") {
    throw new Error(
      "Euler SDE is currently limited to instrumental generation because the XL INT4 vocal quality gate fails. Use Euler or Heun for vocals.",
    );
  }
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
  post({
    type: "diagnostic",
    key: "generation mode",
    value: instrumental
      ? "instrumental"
      : `vocals (${vocalLanguage})`,
  });
  post({
    type: "diagnostic",
    key: "sampler",
    value: sampler,
  });
  post({
    type: "diagnostic",
    key: "DCW",
    value: dcw.enabled
      ? `${dcw.mode} · ${dcw.scaler}/${dcw.highScaler}`
      : "disabled",
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
  const lyricTokens = tokenize(
    tokenizer,
    buildLyricPrompt(lyrics, vocalLanguage),
    2048,
  );
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

  startStage(
    "lyric-embedding",
    instrumental
      ? "Embedding the instrumental lyric marker"
      : "Embedding user-supplied lyrics",
  );
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
    `Projecting and packing caption, ${
      instrumental ? "instrumental marker" : "lyrics"
    }, timbre, and source context`,
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
    `Running ACE-Step XL Turbo ${sampler} sampling (${
      sampler === "heun"
        ? INFERENCE_STEPS * 2 - 1
        : INFERENCE_STEPS
    } DiT evaluations)`,
  );
  const ditSession = await sessionFor("dit", allowWasmFallback);
  const schedule = createTurboSchedule(INFERENCE_STEPS, TURBO_SHIFT);

  const evaluateVelocity = async (
    inputLatent: Float32Array,
    timestep: number,
    tensorName: string,
  ) => {
    const hiddenStatesTensor = new ort.Tensor(
      "float32",
      inputLatent,
      [1, latentFrames, LATENT_CHANNELS],
    );
    const timestepTensor = new ort.Tensor(
      "float32",
      new Float32Array([timestep]),
      [1],
    );
    const encoderHiddenTensor = new ort.Tensor(
      "float32",
      encoderHidden,
      encoderDims,
    );
    const contextLatentsTensor = new ort.Tensor(
      "float32",
      contextLatents,
      contextDims,
    );
    try {
      const outputs = await ditSession.run({
        hidden_states: hiddenStatesTensor,
        timestep: timestepTensor,
        encoder_hidden_states: encoderHiddenTensor,
        context_latents: contextLatentsTensor,
      });
      const velocityTensor = outputs.velocity as NumericTensor;
      try {
        const velocity = asFloat32(velocityTensor, "DiT velocity");
        assertShape(
          velocityTensor.dims,
          [1, latentFrames, LATENT_CHANNELS],
          tensorName,
        );
        recordTensor(tensorName, velocityTensor.dims, velocity);
        return velocity;
      } finally {
        velocityTensor.dispose();
      }
    } finally {
      hiddenStatesTensor.dispose();
      timestepTensor.dispose();
      encoderHiddenTensor.dispose();
      contextLatentsTensor.dispose();
    }
  };

  for (const step of schedule) {
    const stepStage = `${sampler}:${step.index + 1}`;
    startStage(
      stepStage,
      `DiT ${step.index + 1}/${schedule.length} · t=${step.current.toFixed(6)} → ${step.next.toFixed(6)}`,
    );
    const latentBefore = latent;
    const velocity = await evaluateVelocity(
      latentBefore,
      step.current,
      `velocity_${step.index + 1}`,
    );
    const denoised = dcw.enabled
      ? predictCleanSample(latentBefore, velocity, step.current)
      : undefined;

    if (sampler === "heun" && step.next > 0) {
      const predictor = eulerFlowStep(
        latentBefore,
        velocity,
        step.delta,
      );
      const correctorVelocity = await evaluateVelocity(
        predictor,
        step.next,
        `corrector_velocity_${step.index + 1}`,
      );
      latent = heunFlowStep(
        latentBefore,
        velocity,
        correctorVelocity,
        step.delta,
      );
    } else if (sampler === "euler-sde") {
      const secondaryNoise =
        step.next > 0
          ? deterministicNormalStream(
              latentBefore.length,
              seed,
              step.index + 1,
            )
          : new Float32Array(latentBefore.length);
      latent = eulerSdeFlowStep(
        latentBefore,
        velocity,
        secondaryNoise,
        step.current,
        step.next,
      );
    } else {
      latent = eulerFlowStep(latentBefore, velocity, step.delta);
    }
    if (denoised) {
      latent = applyDcw(
        latent,
        denoised,
        latentFrames,
        LATENT_CHANNELS,
        step.current,
        dcw,
      );
    }
    recordTensor(
      `latent_${step.index + 1}`,
      [1, latentFrames, LATENT_CHANNELS],
      latent,
    );
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
    `Decoding ${latentFrames} latent frames to ${durationSeconds} seconds of stereo PCM in memory-bounded chunks`,
  );
  const vaePlan = createVaeDecodePlan(latentFrames);
  const waveform = new Float32Array(audioFrames * 2);
  post({
    type: "diagnostic",
    key: "VAE decode chunks",
    value: vaePlan.length,
  });
  const vaeSession = await sessionFor("vae", allowWasmFallback);
  try {
    for (const chunk of vaePlan) {
      const inputFrames =
        chunk.inputEndFrame - chunk.inputStartFrame;
      const coreFrames =
        chunk.coreEndFrame - chunk.coreStartFrame;
      const chunkLatent = new Float32Array(
        inputFrames * LATENT_CHANNELS,
      );
      for (
        let inputFrame = 0;
        inputFrame < inputFrames;
        inputFrame += 1
      ) {
        const sourceFrame = chunk.inputStartFrame + inputFrame;
        for (
          let channel = 0;
          channel < LATENT_CHANNELS;
          channel += 1
        ) {
          chunkLatent[channel * inputFrames + inputFrame] =
            latent[sourceFrame * LATENT_CHANNELS + channel];
        }
      }

      const chunkStage = `vae-decode:${chunk.index + 1}`;
      startStage(
        chunkStage,
        `VAE chunk ${chunk.index + 1}/${vaePlan.length} · latent frames ${chunk.coreStartFrame}-${chunk.coreEndFrame}`,
      );
      const inputTensor = new ort.Tensor(
        "float32",
        chunkLatent,
        [1, LATENT_CHANNELS, inputFrames],
      );
      let vaeOutputs: ort.InferenceSession.OnnxValueMapType;
      try {
        vaeOutputs = await vaeSession.run({
          latents: inputTensor,
        });
      } catch (error) {
        throw new Error(
          `VAE chunk ${chunk.index + 1}/${vaePlan.length} failed for ${coreFrames / LATENT_FRAME_RATE} seconds of output. ${error instanceof Error ? error.message : String(error)}`,
        );
      } finally {
        inputTensor.dispose();
      }

      const waveformTensor = vaeOutputs.waveform as NumericTensor;
      try {
        const chunkAudioFrames = inputFrames * VAE_UPSAMPLE_FACTOR;
        assertShape(
          waveformTensor.dims,
          [1, 2, chunkAudioFrames],
          `decoded waveform chunk ${chunk.index + 1}`,
        );
        const chunkWaveform = asFloat32(
          waveformTensor,
          `decoded waveform chunk ${chunk.index + 1}`,
        );
        const cropStart =
          chunk.cropStartFrame * VAE_UPSAMPLE_FACTOR;
        const coreAudioFrames = coreFrames * VAE_UPSAMPLE_FACTOR;
        const destinationStart =
          chunk.coreStartFrame * VAE_UPSAMPLE_FACTOR;
        for (let channel = 0; channel < 2; channel += 1) {
          const sourceStart =
            channel * chunkAudioFrames + cropStart;
          waveform.set(
            chunkWaveform.subarray(
              sourceStart,
              sourceStart + coreAudioFrames,
            ),
            channel * audioFrames + destinationStart,
          );
        }
      } finally {
        waveformTensor.dispose();
      }
      endStage(chunkStage);
    }
  } finally {
    await vaeSession.release();
  }
  recordTensor("waveform", [1, 2, audioFrames], waveform);
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
      throw new Error(
        `VAE output collapsed after two seconds: ${startSecond}-${endSecond}s RMS is ${leftRms.toFixed(6)} L / ${rightRms.toFixed(6)} R.`,
      );
    }
  }
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
      seed,
      sampler,
      instrumental,
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
  try {
    if (event.data.type === "clear-cache") {
      await clearPersistentAssets();
      post({ type: "cache-cleared" });
      return;
    }
    if (event.data.type === "list-cache") {
      runtimeAssets = event.data.assets ?? {};
      post({
        type: "cache-inventory",
        inventory: await cacheInventory(),
      });
      return;
    }
    if (event.data.type === "remove-cached-model") {
      runtimeAssets = event.data.assets ?? {};
      const removedBytes = await removeCachedModel(event.data.modelId);
      post({
        type: "cached-model-removed",
        modelId: event.data.modelId,
        removedBytes,
      });
      post({
        type: "cache-inventory",
        inventory: await cacheInventory(),
      });
      return;
    }
    await runGeneration(
      event.data.prompt,
      event.data.lyrics,
      event.data.vocalLanguage,
      event.data.seed,
      event.data.durationSeconds,
      event.data.sampler,
      event.data.dcw,
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
      stage:
        [...stageStarted.keys()].at(-1) ??
        (event.data.type === "start" ? "unknown" : "cache"),
      message: details.message ?? String(error),
      graph: details.graph,
      operatorHint: details.operatorHint,
      stack: details.stack,
    });
  }
};
