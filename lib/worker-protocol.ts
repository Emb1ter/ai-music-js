import type { TensorSummary } from "./tensor-diagnostics";

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
  seed: number;
  durationSeconds: number;
  allowWasmFallback: boolean;
  assets?: WorkerAssetConfig;
};

export type ClearCacheRequest = {
  type: "clear-cache";
};

export type WorkerRequest = StartRequest | ClearCacheRequest;

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

export type WorkerUpdate =
  | DownloadUpdate
  | StageUpdate
  | TimingUpdate
  | CompatibilityUpdate
  | TraceUpdate
  | DiagnosticUpdate
  | CompleteUpdate
  | ErrorUpdate
  | CacheClearedUpdate;
