import type { TensorSummary } from "./tensor-diagnostics";
import type {
  ResolvedDcwOptions,
  SamplerMode,
} from "./generation-options";

export type WorkerAssetConfig = {
  /**
   * Overrides every model/support download URL. The directory must contain
   * every file from getRequiredAssets().
   */
  allAssetsBaseUrl?: string;
  /**
   * Overrides only the fresh XL condition/DiT files. Shared upstream assets
   * remain pinned to Hugging Face.
   */
  modelBaseUrl?: string;
  /** URL of the pinned ONNX Runtime Web asyncify WASM binary. */
  wasmUrl?: string;
  /** URL of the matching Emscripten module used to instantiate the WASM. */
  wasmModuleUrl?: string;
};

export type StartRequest = {
  type: "start";
  prompt: string;
  lyrics: string;
  vocalLanguage: string;
  seed: number;
  durationSeconds: number;
  sampler: SamplerMode;
  dcw: ResolvedDcwOptions;
  allowWasmFallback: boolean;
  assets?: WorkerAssetConfig;
};

export type ClearCacheRequest = {
  type: "clear-cache";
};

export type ListCacheRequest = {
  type: "list-cache";
  assets?: WorkerAssetConfig;
};

export type RemoveCachedModelRequest = {
  type: "remove-cached-model";
  modelId: string;
  assets?: WorkerAssetConfig;
};

export type WorkerRequest =
  | StartRequest
  | ClearCacheRequest
  | ListCacheRequest
  | RemoveCachedModelRequest;

export type DownloadUpdate = {
  type: "download";
  assetId: string;
  group: string;
  label: string;
  loaded: number;
  total: number;
  cached: boolean;
};

export type StageUpdate = {
  type: "stage";
  stage: string;
  detail: string;
  startedAt: number;
};

export type TimingUpdate = {
  type: "timing";
  stage: string;
  milliseconds: number;
};

export type CompatibilityUpdate = {
  type: "compatibility";
  ok: boolean;
  message: string;
  adapter?: {
    vendor: string;
    architecture: string;
    device: string;
    description: string;
    maxBufferSize: number;
    maxStorageBufferBindingSize: number;
  };
};

export type TraceUpdate = {
  type: "trace";
  summary: TensorSummary;
};

export type DiagnosticUpdate = {
  type: "diagnostic";
  key: string;
  value: string | number;
};

export type CompleteUpdate = {
  type: "complete";
  seed: number;
  sampler: SamplerMode;
  instrumental: boolean;
  wav: ArrayBuffer;
  left: ArrayBuffer;
  right: ArrayBuffer;
  sampleRate: number;
  durationSeconds: number;
  latentFrames: number;
  trace: TensorSummary[];
  timings: Record<string, number>;
  estimatedPeakBytes: number;
};

export type BatchProgressUpdate = {
  type: "batch-progress";
  index: number;
  total: number;
  seed: number;
  status: "started" | "complete";
};

export type ErrorUpdate = {
  type: "error";
  stage: string;
  message: string;
  graph?: string;
  operatorHint?: string;
  stack?: string;
};

export type CacheClearedUpdate = {
  type: "cache-cleared";
};

export type CachedAssetInfo = {
  id: string;
  group: string;
  label: string;
  fileName: string;
  role: "graph" | "weights" | "tokenizer" | "conditioning";
  expectedBytes: number;
  storedBytes: number;
  cached: boolean;
  storage: "cache-api" | "opfs" | null;
};

export type CachedModelInfo = {
  id: string;
  label: string;
  expectedBytes: number;
  storedBytes: number;
  complete: boolean;
  partial: boolean;
  assets: CachedAssetInfo[];
};

export type CacheInventory = {
  origin: string;
  cacheName: string;
  expectedBytes: number;
  storedBytes: number;
  readyBytes: number;
  missingBytes: number;
  usageBytes?: number;
  quotaBytes?: number;
  availableBytes?: number;
  persisted?: boolean;
  models: CachedModelInfo[];
};

export type CacheInventoryUpdate = {
  type: "cache-inventory";
  inventory: CacheInventory;
};

export type CachedModelRemovedUpdate = {
  type: "cached-model-removed";
  modelId: string;
  removedBytes: number;
};

export type WorkerUpdate =
  | DownloadUpdate
  | StageUpdate
  | TimingUpdate
  | CompatibilityUpdate
  | TraceUpdate
  | DiagnosticUpdate
  | CompleteUpdate
  | BatchProgressUpdate
  | ErrorUpdate
  | CacheClearedUpdate
  | CacheInventoryUpdate
  | CachedModelRemovedUpdate;
