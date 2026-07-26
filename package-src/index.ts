import {
  ALL_ASSETS,
  DEFAULT_MODEL_BASE_URL,
  DEFAULT_DURATION_SECONDS,
  MAX_DURATION_SECONDS,
  MIN_DURATION_SECONDS,
  MODEL_NAME,
  MODEL_PARAMETER_COUNT,
  PIPELINE_BUILD,
  TOTAL_DOWNLOAD_BYTES,
  type DownloadAsset,
} from "../lib/model-manifest";
import type {
  CompleteUpdate,
  ErrorUpdate,
  WorkerAssetConfig,
  WorkerRequest,
  WorkerUpdate,
} from "../lib/worker-protocol";

export {
  DEFAULT_DURATION_SECONDS,
  DEFAULT_MODEL_BASE_URL,
  INFERENCE_STEPS,
  MAX_DURATION_SECONDS,
  MIN_DURATION_SECONDS,
  MODEL_NAME,
  MODEL_PARAMETER_COUNT,
  PIPELINE_BUILD,
  SAMPLE_RATE,
  TOTAL_DOWNLOAD_BYTES,
} from "../lib/model-manifest";
export type {
  CacheClearedUpdate,
  CompatibilityUpdate,
  CompleteUpdate,
  DiagnosticUpdate,
  DownloadUpdate,
  ErrorUpdate,
  StageUpdate,
  TimingUpdate,
  TraceUpdate,
  WorkerAssetConfig,
  WorkerUpdate,
} from "../lib/worker-protocol";
export type { TensorSummary } from "../lib/tensor-diagnostics";

export const DEFAULT_INSTRUMENTAL_PROMPT =
  "Warm analog synthwave instrumental, steady electronic drums, pulsing bass, cinematic pads, memorable lead melody, polished studio mix";

export type UrlValue = string | URL;

export type AceStepWebGpuOptions = {
  /**
   * Directory containing the fresh XL ONNX files. Defaults to the pinned
   * Hugging Face browser export.
   */
  modelBaseUrl?: UrlValue;
  /**
   * Directory containing every required model/support file. When set, this
   * takes precedence over modelBaseUrl and the pinned upstream URLs.
   */
  allAssetsBaseUrl?: UrlValue;
  /** Override the packaged ONNX Runtime Web WASM URL. */
  wasmUrl?: UrlValue;
  /** Override the matching packaged Emscripten module URL. */
  wasmModuleUrl?: UrlValue;
  /** Override the packaged module Worker URL. */
  workerUrl?: UrlValue;
  /** Integrate with a custom Worker loader or test harness. */
  workerFactory?: (url: URL, options: WorkerOptions) => Worker;
  /** Receive download, stage, timing, trace, diagnostic and result updates. */
  onUpdate?: AceStepUpdateListener;
  /** Defaults to WebGPU with WASM compatibility fallback enabled. */
  allowWasmFallback?: boolean;
};

export type GenerateOptions = {
  prompt: string;
  seed?: number;
  durationSeconds?: number;
  allowWasmFallback?: boolean;
  signal?: AbortSignal;
};

export type AceStepGenerationResult = {
  audioBuffer: AudioBuffer;
  wav: Blob;
  wavBytes: ArrayBuffer;
  channels: readonly [Float32Array, Float32Array];
  sampleRate: number;
  durationSeconds: number;
  latentFrames: number;
  trace: CompleteUpdate["trace"];
  timings: Record<string, number>;
  estimatedPeakBytes: number;
};

export type AceStepUpdateListener = (update: WorkerUpdate) => void;

export type RequiredAsset = Pick<
  DownloadAsset,
  "id" | "group" | "label" | "fileName" | "bytes" | "role"
> & {
  url: string;
};

export type AssetResolutionOptions = {
  origin?: UrlValue;
  modelBaseUrl?: UrlValue;
  allAssetsBaseUrl?: UrlValue;
};

const browserBaseUrl = () =>
  typeof location === "undefined" ? import.meta.url : location.href;

const absoluteUrl = (value: UrlValue, base = browserBaseUrl()) =>
  new URL(value.toString(), base);

const directoryUrl = (value: UrlValue, base = browserBaseUrl()) => {
  const resolved = absoluteUrl(value, base);
  if (!resolved.pathname.endsWith("/")) {
    resolved.pathname += "/";
  }
  return resolved;
};

const configuredAssetUrl = (
  asset: DownloadAsset,
  options: AssetResolutionOptions,
) => {
  const origin = options.origin
    ? absoluteUrl(options.origin).href
    : browserBaseUrl();
  const baseUrl =
    options.allAssetsBaseUrl ??
    (asset.url.startsWith("/models/")
      ? options.modelBaseUrl ?? DEFAULT_MODEL_BASE_URL
      : undefined);
  if (!baseUrl) {
    return new URL(asset.url, origin).href;
  }
  const configured = new URL(
    asset.fileName,
    directoryUrl(baseUrl, origin),
  );
  configured.search = new URL(asset.url, origin).search;
  return configured.href;
};

/**
 * Returns the exact files and resolved URLs used by the Worker. Use this to
 * provision a static bucket/CDN and to display cold-download requirements.
 */
export const getRequiredAssets = (
  options: AssetResolutionOptions = {},
): RequiredAsset[] =>
  ALL_ASSETS.map((asset) => ({
    id: asset.id,
    group: asset.group,
    label: asset.label,
    fileName: asset.fileName,
    bytes: asset.bytes,
    role: asset.role,
    url: configuredAssetUrl(asset, options),
  }));

/** Fresh XL files that the npm package deliberately does not embed. */
export const LOCAL_MODEL_FILES: readonly RequiredAsset[] = ALL_ASSETS.filter(
  (asset) => asset.url.startsWith("/models/"),
).map((asset) => ({
  id: asset.id,
  group: asset.group,
  label: asset.label,
  fileName: asset.fileName,
  bytes: asset.bytes,
  role: asset.role,
  url: asset.url,
}));

export class AceStepWebGpuError extends Error {
  readonly stage: string;
  readonly graph?: string;
  readonly operatorHint?: string;

  constructor(update: ErrorUpdate) {
    super(update.message);
    this.name = "AceStepWebGpuError";
    this.stage = update.stage;
    this.graph = update.graph;
    this.operatorHint = update.operatorHint;
    if (update.stack) {
      this.stack = update.stack;
    }
  }
}

type WorkerOutcome<T> =
  | { done: false }
  | { done: true; value: T };

export class AceStepWebGpu {
  readonly modelName = MODEL_NAME;
  readonly modelParameterCount = MODEL_PARAMETER_COUNT;
  readonly pipelineBuild = PIPELINE_BUILD;
  readonly totalDownloadBytes = TOTAL_DOWNLOAD_BYTES;

  private readonly workerUrl: URL;
  private readonly workerFactory: (
    url: URL,
    options: WorkerOptions,
  ) => Worker;
  private readonly assets: WorkerAssetConfig;
  private readonly defaultAllowWasmFallback: boolean;
  private readonly listeners = new Set<AceStepUpdateListener>();
  private worker: Worker | null = null;
  private cancelActive: ((reason: Error) => void) | null = null;
  private disposed = false;

  constructor(options: AceStepWebGpuOptions = {}) {
    this.workerUrl = options.workerUrl
      ? absoluteUrl(options.workerUrl)
      : new URL("./ace-step.worker.js", import.meta.url);
    this.workerFactory =
      options.workerFactory ??
      ((url, workerOptions) => new Worker(url, workerOptions));
    this.assets = {
      modelBaseUrl: options.modelBaseUrl
        ? directoryUrl(options.modelBaseUrl).href
        : DEFAULT_MODEL_BASE_URL,
      allAssetsBaseUrl: options.allAssetsBaseUrl
        ? directoryUrl(options.allAssetsBaseUrl).href
        : undefined,
      wasmUrl: options.wasmUrl
        ? absoluteUrl(options.wasmUrl).href
        : new URL(
            "./wasm/ort-wasm-simd-threaded.asyncify.wasm",
            import.meta.url,
          ).href,
      wasmModuleUrl: options.wasmModuleUrl
        ? absoluteUrl(options.wasmModuleUrl).href
        : new URL(
            "./wasm/ort-wasm-simd-threaded.asyncify.mjs",
            import.meta.url,
          ).href,
    };
    this.defaultAllowWasmFallback = options.allowWasmFallback ?? true;
    if (options.onUpdate) {
      this.listeners.add(options.onUpdate);
    }
  }

  get busy() {
    return this.worker !== null;
  }

  subscribe(listener: AceStepUpdateListener) {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  }

  private notify(update: WorkerUpdate) {
    for (const listener of this.listeners) {
      try {
        listener(update);
      } catch (error) {
        console.error("ACE-Step update listener failed", error);
      }
    }
  }

  private createWorker() {
    if (this.disposed) {
      throw new Error("This ACE-Step WebGPU instance has been disposed.");
    }
    if (this.worker) {
      throw new Error("An ACE-Step Worker operation is already active.");
    }
    const worker = this.workerFactory(this.workerUrl, {
      type: "module",
      name: "ai-music-js",
    });
    this.worker = worker;
    return worker;
  }

  private runWorker<T>(
    request: WorkerRequest,
    handleUpdate: (update: WorkerUpdate) => WorkerOutcome<T>,
    signal?: AbortSignal,
  ): Promise<T> {
    if (signal?.aborted) {
      return Promise.reject(
        new DOMException("The ACE-Step operation was aborted.", "AbortError"),
      );
    }

    let worker: Worker;
    try {
      worker = this.createWorker();
    } catch (error) {
      return Promise.reject(error);
    }

    return new Promise<T>((resolve, reject) => {
      let settled = false;

      const finish = (
        callback: (value: T | PromiseLike<T>) => void,
        value: T,
      ) => {
        if (settled) return;
        settled = true;
        signal?.removeEventListener("abort", abort);
        worker.terminate();
        if (this.worker === worker) {
          this.worker = null;
          this.cancelActive = null;
        }
        callback(value);
      };

      const fail = (error: Error) => {
        if (settled) return;
        settled = true;
        signal?.removeEventListener("abort", abort);
        worker.terminate();
        if (this.worker === worker) {
          this.worker = null;
          this.cancelActive = null;
        }
        reject(error);
      };

      const abort = () => {
        fail(
          new DOMException(
            "The ACE-Step operation was aborted.",
            "AbortError",
          ),
        );
      };

      this.cancelActive = fail;
      signal?.addEventListener("abort", abort, { once: true });

      worker.onmessage = (event: MessageEvent<WorkerUpdate>) => {
        const update = event.data;
        this.notify(update);
        if (update.type === "error") {
          fail(new AceStepWebGpuError(update));
          return;
        }
        try {
          const outcome = handleUpdate(update);
          if (outcome.done) {
            finish(resolve, outcome.value);
          }
        } catch (error) {
          fail(error instanceof Error ? error : new Error(String(error)));
        }
      };
      worker.onerror = (event) => {
        event.preventDefault();
        fail(
          new Error(
            event.message || "The ACE-Step inference Worker crashed.",
          ),
        );
      };
      worker.postMessage(request);
    });
  }

  generate(options: GenerateOptions): Promise<AceStepGenerationResult> {
    const prompt = options.prompt.trim();
    const seed = options.seed ?? 42;
    const durationSeconds =
      options.durationSeconds ?? DEFAULT_DURATION_SECONDS;
    if (!prompt) {
      return Promise.reject(new TypeError("A non-empty music prompt is required."));
    }
    if (!Number.isInteger(seed)) {
      return Promise.reject(new TypeError("Seed must be an integer."));
    }
    if (
      !Number.isInteger(durationSeconds) ||
      durationSeconds < MIN_DURATION_SECONDS ||
      durationSeconds > MAX_DURATION_SECONDS
    ) {
      return Promise.reject(
        new RangeError(
          `Duration must be a whole number from ${MIN_DURATION_SECONDS} to ${MAX_DURATION_SECONDS} seconds.`,
        ),
      );
    }

    return this.runWorker<AceStepGenerationResult>(
      {
        type: "start",
        prompt,
        seed,
        durationSeconds,
        allowWasmFallback:
          options.allowWasmFallback ??
          this.defaultAllowWasmFallback,
        assets: this.assets,
      },
      (update) => {
        if (update.type !== "complete") {
          return { done: false };
        }
        return {
          done: true,
          value: resultFromUpdate(update),
        };
      },
      options.signal,
    );
  }

  clearCache(signal?: AbortSignal): Promise<void> {
    return this.runWorker<void>(
      { type: "clear-cache" },
      (update) =>
        update.type === "cache-cleared"
          ? { done: true, value: undefined }
          : { done: false },
      signal,
    );
  }

  cancel() {
    if (!this.cancelActive) return false;
    this.cancelActive(
      new DOMException(
        "The ACE-Step operation was cancelled.",
        "AbortError",
      ),
    );
    return true;
  }

  dispose() {
    this.cancel();
    this.worker?.terminate();
    this.worker = null;
    this.cancelActive = null;
    this.listeners.clear();
    this.disposed = true;
  }
}

const resultFromUpdate = (
  update: CompleteUpdate,
): AceStepGenerationResult => {
  const left = new Float32Array(update.left);
  const right = new Float32Array(update.right);
  const audioBuffer = new AudioBuffer({
    length: left.length,
    numberOfChannels: 2,
    sampleRate: update.sampleRate,
  });
  audioBuffer.copyToChannel(left, 0);
  audioBuffer.copyToChannel(right, 1);
  return {
    audioBuffer,
    wav: new Blob([update.wav], { type: "audio/wav" }),
    wavBytes: update.wav,
    channels: [left, right],
    sampleRate: update.sampleRate,
    durationSeconds: update.durationSeconds,
    latentFrames: update.latentFrames,
    trace: update.trace,
    timings: update.timings,
    estimatedPeakBytes: update.estimatedPeakBytes,
  };
};
