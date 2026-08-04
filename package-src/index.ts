import {
  ALL_ASSETS,
  DEFAULT_AUDIO_QUALITY,
  DEFAULT_MODEL_BASE_URL,
  DEFAULT_DURATION_SECONDS,
  HIGH_QUALITY_TOTAL_DOWNLOAD_BYTES,
  MAX_DURATION_SECONDS,
  MIN_DURATION_SECONDS,
  MODEL_NAME,
  MODEL_PARAMETER_COUNT,
  PIPELINE_BUILD,
  TOTAL_DOWNLOAD_BYTES,
  assetsForAudioQuality,
  buildLyricPrompt,
  hasVocalPromptConflict,
  isAudioQuality,
  isInstrumentalLyrics,
  type AudioQuality,
  type DownloadAsset,
} from "../lib/model-manifest";
import {
  resolveDcwOptions,
  validateSamplerMode,
  type DcwOptions,
  type SamplerMode,
} from "../lib/generation-options";
import {
  DEFAULT_LYRICS_MODEL,
  DEFAULT_LYRICS_MODEL_REVISION,
  assessLyricDuration,
  defaultMaxLyricWords,
  recommendDurationForLyrics,
} from "../lib/lyrics";
import {
  DEFAULT_PLANNER_MODEL,
  DEFAULT_PLANNER_MODEL_REVISION,
  type PlannerMetadata,
} from "../lib/planner";
import {
  DEFAULT_HIGH_QUALITY_PLANNER_MODEL,
  DEFAULT_HIGH_QUALITY_PLANNER_MODEL_REVISION,
  DEFAULT_PLANNER_QUALITY,
  isPlannerQuality,
  type PlannerQuality,
} from "../lib/planner-quality";
import {
  LANGUAGE_MODEL_COMPONENTS,
  LANGUAGE_MODEL_DOWNLOAD_BYTES,
} from "../lib/language-model-manifest";
import type {
  CacheInventory,
  CompleteUpdate,
  ErrorUpdate,
  LanguageWorkerRequest,
  LyricsCompleteUpdate,
  PlanCompleteUpdate,
  PlannerProfileUpdate,
  ProgressUpdate,
  WorkerAssetConfig,
  WorkerRequest,
  WorkerUpdate,
} from "../lib/worker-protocol";

export {
  DEFAULT_DURATION_SECONDS,
  DEFAULT_AUDIO_QUALITY,
  DEFAULT_MODEL_BASE_URL,
  HIGH_QUALITY_TOTAL_DOWNLOAD_BYTES,
  INFERENCE_STEPS,
  MAX_LYRICS_CHARACTERS,
  MAX_DURATION_SECONDS,
  MIN_DURATION_SECONDS,
  MODEL_NAME,
  MODEL_PARAMETER_COUNT,
  PIPELINE_BUILD,
  SAMPLE_RATE,
  TOTAL_DOWNLOAD_BYTES,
  isAudioQuality,
  buildLyricPrompt,
  hasVocalPromptConflict,
  isInstrumentalLyrics,
} from "../lib/model-manifest";
export type { AudioQuality } from "../lib/model-manifest";
export {
  DCW_MODES,
  DEFAULT_DCW_OPTIONS,
  SAMPLER_MODES,
} from "../lib/generation-options";
export {
  BACKEND_VOCAL_MIN_WORDS,
  BACKEND_VOCAL_WORDS_PER_SECOND,
  DEFAULT_VOCAL_MIN_DURATION_SECONDS,
  DEFAULT_LYRICS_MODEL,
  DEFAULT_LYRICS_MODEL_REVISION,
  assessLyricDuration,
  cleanLyrics,
  compactLyrics,
  countLyricWords,
  defaultMaxLyricWords,
  lyricQualityIssues,
  recommendDurationForLyrics,
} from "../lib/lyrics";
export type {
  LyricDurationRecommendation,
  RecommendLyricDurationOptions,
} from "../lib/lyrics";
export {
  AUDIO_CODEBOOK_SIZE,
  AUDIO_CODE_TOKEN_END,
  AUDIO_CODE_TOKEN_START,
  DEFAULT_PLANNER_MODEL,
  DEFAULT_PLANNER_MODEL_REVISION,
  SEMANTIC_CODE_RATE,
  audioCodeToTokenId,
  fallbackPlannerMetadata,
  formatPlannerMetadata,
  parsePlannerMetadata,
  semanticCodeCount,
  tokenIdToAudioCode,
} from "../lib/planner";
export {
  DEFAULT_HIGH_QUALITY_PLANNER_MODEL,
  DEFAULT_HIGH_QUALITY_PLANNER_MODEL_REVISION,
  DEFAULT_PLANNER_QUALITY,
  isPlannerQuality,
} from "../lib/planner-quality";
export {
  HIGH_QUALITY_PLANNER_MODEL_ASSETS,
  HIGH_QUALITY_PLANNER_MODEL_DOWNLOAD_BYTES,
  LANGUAGE_CACHE_NAME,
  LANGUAGE_MODEL_COMPONENTS,
  LANGUAGE_MODEL_DOWNLOAD_BYTES,
  LYRICS_MODEL_ASSETS,
  LYRICS_MODEL_DOWNLOAD_BYTES,
  PLANNER_MODEL_ASSETS,
  PLANNER_MODEL_DOWNLOAD_BYTES,
} from "../lib/language-model-manifest";
export type {
  DcwMode,
  DcwOptions,
  ResolvedDcwOptions,
  SamplerMode,
} from "../lib/generation-options";
export type {
  BatchProgressUpdate,
  CachedAssetInfo,
  CachedModelInfo,
  CacheClearedUpdate,
  CacheInventory,
  CacheInventoryUpdate,
  CachedModelRemovedUpdate,
  CompatibilityUpdate,
  CompleteUpdate,
  DiagnosticUpdate,
  DownloadUpdate,
  ErrorUpdate,
  LyricsCompleteUpdate,
  PlanCompleteUpdate,
  PlannerProfileUpdate,
  ProgressUpdate,
  LanguageWorkerRequest,
  StageUpdate,
  TimingUpdate,
  TraceUpdate,
  WorkerAssetConfig,
  WorkerUpdate,
} from "../lib/worker-protocol";
export type {
  PlannerEmbeddingStats,
  PlannerInputFingerprint,
  PlannerProfilePhase,
  PlannerProfileReport,
  PlannerTimingMetric,
} from "../lib/planner-profile";
export type { PlannerMetadata } from "../lib/planner";
export type { PlannerQuality } from "../lib/planner-quality";
export type {
  LanguageModelAsset,
  LanguageModelComponent,
} from "../lib/language-model-manifest";
export type { TensorSummary } from "../lib/tensor-diagnostics";

/** Default Turbo preset: the shared XL audio graphs without a 4B planner. */
export const DEFAULT_GENERATION_DOWNLOAD_BYTES = TOTAL_DOWNLOAD_BYTES;
export const HIGH_PRECISION_GENERATION_DOWNLOAD_BYTES =
  HIGH_QUALITY_TOTAL_DOWNLOAD_BYTES;

/** Every audio and language component, including the optional lyric writer. */
export const FULL_MODEL_DOWNLOAD_BYTES =
  TOTAL_DOWNLOAD_BYTES + LANGUAGE_MODEL_DOWNLOAD_BYTES;

export const DEFAULT_INSTRUMENTAL_PROMPT =
  "Warm analog synthwave instrumental, steady electronic drums, pulsing bass, cinematic pads, memorable lead melody, polished studio mix";

export const DEFAULT_VOCAL_PROMPT =
  "Warm analog synthwave song, steady electronic drums, pulsing bass, cinematic pads, memorable chorus, clear expressive lead vocal singing every supplied lyric, polished studio mix";

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
  /** Override the packaged Qwen/Transformers.js language Worker URL. */
  languageWorkerUrl?: UrlValue;
  /** Integrate with a custom Worker loader or test harness. */
  workerFactory?: (url: URL, options: WorkerOptions) => Worker;
  /** Integrate the Qwen language stage with a custom Worker loader. */
  languageWorkerFactory?: (url: URL, options: WorkerOptions) => Worker;
  /** Qwen3.5 Transformers.js model repository override. */
  lyricsModelId?: string;
  /** Immutable Hugging Face revision used for the lyric model. */
  lyricsModelRevision?: string;
  /** Exact ACE-Step 5 Hz 4B planner repository override. */
  plannerModelId?: string;
  /** Immutable Hugging Face revision used for the ACE planner. */
  plannerModelRevision?: string;
  /** INT8-weight / FP32-compute ACE planner repository override. */
  highQualityPlannerModelId?: string;
  /** Immutable Hugging Face revision used for the high-quality planner. */
  highQualityPlannerModelRevision?: string;
  /** Receive download, stage, timing, trace, diagnostic and result updates. */
  onUpdate?: AceStepUpdateListener;
  /** Defaults to WebGPU with WASM compatibility fallback enabled. */
  allowWasmFallback?: boolean;
};

export type GenerateOptions = {
  prompt: string;
  /**
   * Audio-model precision. `standard` uses the 5.63 GB INT4 audio path;
   * `high` uses the 8.00 GB INT8 condition encoder and DiT. Both use the
   * browser-qualified FP32 WebGPU VAE.
   */
  audioQuality?: AudioQuality;
  /**
   * Semantic planner mode. `turbo` skips the 4B planner; `high-quality` uses
   * the verified ~4.63 GB INT8-weight / FP32-compute planner.
   */
  plannerQuality?: PlannerQuality;
  /** Omit or leave empty for instrumental generation. */
  lyrics?: string;
  /** Generate lyrics from the prompt with browser-local Qwen3.5 first. */
  writeLyrics?: boolean;
  /**
   * Generate ACE 5 Hz semantic codes before diffusion. Defaults to true and
   * is the browser path closest to the official backend.
   */
  semanticPlanning?: boolean;
  /**
   * Precomputed ACE 5 Hz codebook indices. When supplied, plannerMetadata is
   * also required and the language-model Worker is skipped. This is intended
   * for separately qualified planners that release their GPU session before
   * the audio pipeline starts.
   */
  semanticCodeIds?: readonly number[];
  /** Metadata paired with semanticCodeIds from the same ACE planner run. */
  plannerMetadata?: PlannerMetadata;
  /** Hard lyric budget; defaults to a duration-derived value. */
  maxLyricWords?: number;
  /** BCP-47-like language code placed in ACE-Step's lyric prompt. */
  vocalLanguage?: string;
  seed?: number;
  /** Requested duration. With autoDuration, this is the minimum preference. */
  durationSeconds?: number;
  /**
   * Let the ACE metadata phase choose the duration, never shorter than the
   * duration required by the backend-matched lyric budget.
   */
  autoDuration?: boolean;
  sampler?: SamplerMode;
  dcw?: DcwOptions;
  allowWasmFallback?: boolean;
  signal?: AbortSignal;
};

export type WriteLyricsOptions = {
  prompt: string;
  seed?: number;
  durationSeconds?: number;
  maxWords?: number;
  signal?: AbortSignal;
};

export type WriteLyricsResult = Omit<LyricsCompleteUpdate, "type">;

export type PlanMusicOptions = {
  prompt: string;
  plannerQuality?: PlannerQuality;
  /**
   * Audio profile whose missing browser assets must remain protected while
   * planner shards are cached. Defaults to the standard INT4 audio path.
   */
  audioQuality?: AudioQuality;
  lyrics?: string;
  vocalLanguage?: string;
  seed?: number;
  durationSeconds?: number;
  autoDuration?: boolean;
  signal?: AbortSignal;
};

export type PlanMusicResult = Omit<PlanCompleteUpdate, "type">;

export type GenerateBatchOptions = Omit<GenerateOptions, "seed"> & {
  /** One sequential generation per seed; accepted length is 1 through 8. */
  seeds: readonly number[];
};

export type AceStepGenerationResult = {
  seed: number;
  audioQuality: AudioQuality;
  sampler: SamplerMode;
  instrumental: boolean;
  /** Empty for instrumentals; includes supplied or Qwen-generated lyrics. */
  lyrics: string;
  /** Present when the exact ACE 5 Hz planner conditioned this generation. */
  plan?: {
    plannerQuality: PlannerQuality;
    metadata: PlannerMetadata;
    reasoning: string;
    model: string;
    revision: string;
    semanticCodeCount: number;
    timings: Record<string, number>;
  };
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
  audioQuality?: AudioQuality;
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
  assetsForAudioQuality(
    options.audioQuality ?? DEFAULT_AUDIO_QUALITY,
  ).map((asset) => ({
    id: asset.id,
    group: asset.group,
    label: asset.label,
    fileName: asset.fileName,
    bytes: asset.bytes,
    role: asset.role,
    url: configuredAssetUrl(asset, options),
  }));

/** Fresh XL files that the npm package deliberately does not embed. */
export const LOCAL_MODEL_FILES: readonly RequiredAsset[] =
  assetsForAudioQuality().filter(
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

/** Fresh XL INT8 files used by GenerateOptions.audioQuality = "high". */
export const HIGH_PRECISION_MODEL_FILES: readonly RequiredAsset[] =
  assetsForAudioQuality("high")
    .filter((asset) => asset.url.startsWith("/models/"))
    .map((asset) => ({
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

type ProgressRange = {
  start: number;
  end: number;
};

type WorkerProgressKind = "lyrics" | "planner" | "audio";

const clampProgress = (value: number) =>
  Math.min(1, Math.max(0, Number.isFinite(value) ? value : 0));

const progressSubrange = (
  parent: ProgressRange,
  start: number,
  end: number,
): ProgressRange => ({
  start: parent.start + (parent.end - parent.start) * start,
  end: parent.start + (parent.end - parent.start) * end,
});

class OperationProgressReporter {
  private value = 0;
  private lastStage = "";
  private lastDetail = "";

  constructor(
    private readonly operation: ProgressUpdate["operation"],
    private readonly emit: (update: ProgressUpdate) => void,
  ) {}

  report(
    progress: number,
    stage: string,
    detail?: string,
  ) {
    const next = Math.max(this.value, clampProgress(progress));
    const nextDetail = detail ?? "";
    if (
      next === this.value &&
      stage === this.lastStage &&
      nextDetail === this.lastDetail
    ) {
      return;
    }
    this.value = next;
    this.lastStage = stage;
    this.lastDetail = nextDetail;
    this.emit({
      type: "progress",
      progress: next,
      operation: this.operation,
      stage,
      ...(detail ? { detail } : {}),
    });
  }

  reportLocal(
    range: ProgressRange,
    progress: number,
    stage: string,
    detail?: string,
  ) {
    this.report(
      range.start + (range.end - range.start) * clampProgress(progress),
      stage,
      detail,
    );
  }

  current() {
    return this.value;
  }
}

const STATIC_WORKER_PROGRESS: Record<
  WorkerProgressKind,
  Record<string, ProgressRange>
> = {
  lyrics: {
    "lyrics-model": { start: 0.01, end: 0.48 },
    "lyrics-writing": { start: 0.5, end: 0.9 },
    "lyrics-repair": { start: 0.72, end: 0.97 },
  },
  planner: {
    "planner-model": { start: 0.01, end: 0.1 },
    "planner-body": { start: 0.1, end: 0.24 },
    "planner-metadata": { start: 0.25, end: 0.42 },
    "planner-head": { start: 0.43, end: 0.54 },
    "planner-semantic-codes": { start: 0.55, end: 0.99 },
  },
  audio: {
    compatibility: { start: 0.01, end: 0.03 },
    tokenization: { start: 0.03, end: 0.07 },
    "text-encoding": { start: 0.07, end: 0.14 },
    "lyric-embedding": { start: 0.14, end: 0.19 },
    "semantic-detokenizer": { start: 0.19, end: 0.24 },
    "condition-packing": { start: 0.24, end: 0.31 },
    "initial-latent": { start: 0.31, end: 0.32 },
    "flow-matching": { start: 0.32, end: 0.76 },
    "vae-decode": { start: 0.76, end: 0.97 },
    "audio-packaging": { start: 0.97, end: 0.995 },
  },
};

class WorkerProgressTracker {
  private localProgress = 0;
  private activeBand: ProgressRange = { start: 0, end: 0.01 };
  private readonly stageBands = new Map<string, ProgressRange>();
  private readonly downloads = new Map<
    string,
    { loaded: number; total: number }
  >();

  constructor(
    private readonly kind: WorkerProgressKind,
    private readonly reporter: OperationProgressReporter,
    private readonly range: ProgressRange,
  ) {}

  private report(progress: number, stage: string, detail?: string) {
    this.localProgress = Math.max(this.localProgress, clampProgress(progress));
    this.reporter.reportLocal(
      this.range,
      this.localProgress,
      stage,
      detail,
    );
  }

  private dynamicStageBand(update: Extract<WorkerUpdate, { type: "stage" }>) {
    if (this.kind !== "audio") return undefined;
    const sampler = update.stage.match(/^(?:euler|heun|euler-sde):(\d+)$/);
    const samplerDetail = update.detail.match(/DiT\s+(\d+)\/(\d+)/i);
    if (sampler && samplerDetail) {
      const index = Number(samplerDetail[1]);
      const total = Number(samplerDetail[2]);
      return {
        start: 0.32 + 0.44 * ((index - 1) / total),
        end: 0.32 + 0.44 * (index / total),
      };
    }
    const vae = update.stage.match(/^vae-decode:(\d+)$/);
    const vaeDetail = update.detail.match(/VAE chunk\s+(\d+)\/(\d+)/i);
    if (vae && vaeDetail) {
      const index = Number(vaeDetail[1]);
      const total = Number(vaeDetail[2]);
      return {
        start: 0.76 + 0.21 * ((index - 1) / total),
        end: 0.76 + 0.21 * (index / total),
      };
    }
    return undefined;
  }

  observe(update: WorkerUpdate) {
    if (update.type === "stage") {
      const band =
        this.dynamicStageBand(update) ??
        STATIC_WORKER_PROGRESS[this.kind][update.stage];
      if (band) {
        this.activeBand = band;
        this.stageBands.set(update.stage, band);
        this.downloads.clear();
        this.report(band.start, update.stage, update.detail);
      } else {
        this.report(this.localProgress, update.stage, update.detail);
      }
      return;
    }

    if (update.type === "download") {
      this.downloads.set(update.assetId, {
        loaded: update.loaded,
        total: update.total,
      });
      const totals = [...this.downloads.values()].reduce(
        (sum, item) => ({
          loaded: sum.loaded + item.loaded,
          total: sum.total + item.total,
        }),
        { loaded: 0, total: 0 },
      );
      if (totals.total > 0) {
        const fraction = Math.min(1, totals.loaded / totals.total);
        this.report(
          this.activeBand.start +
            (this.activeBand.end - this.activeBand.start) * fraction * 0.85,
          `download:${update.group}`,
          `${update.label} ${Math.round(fraction * 100)}%`,
        );
      }
      return;
    }

    if (update.type === "timing") {
      const band =
        this.stageBands.get(update.stage) ??
        STATIC_WORKER_PROGRESS[this.kind][update.stage];
      if (band) this.report(band.end, update.stage);
      return;
    }

    if (update.type === "planner-profile" && this.kind === "planner") {
      const { completedSemanticSteps, targetSemanticSteps } = update.report;
      const fraction =
        targetSemanticSteps > 0
          ? completedSemanticSteps / targetSemanticSteps
          : 0;
      this.report(
        0.55 + 0.44 * fraction,
        "planner-semantic-codes",
        `${completedSemanticSteps}/${targetSemanticSteps} semantic codes`,
      );
      return;
    }

    if (
      (this.kind === "lyrics" && update.type === "lyrics-complete") ||
      (this.kind === "planner" && update.type === "plan-complete") ||
      (this.kind === "audio" && update.type === "complete")
    ) {
      this.report(1, `${this.kind}-complete`);
    }
  }
}

export class AceStepWebGpu {
  readonly modelName = MODEL_NAME;
  readonly modelParameterCount = MODEL_PARAMETER_COUNT;
  readonly pipelineBuild = PIPELINE_BUILD;
  readonly audioDownloadBytes = TOTAL_DOWNLOAD_BYTES;
  readonly highPrecisionAudioDownloadBytes =
    HIGH_QUALITY_TOTAL_DOWNLOAD_BYTES;
  readonly totalDownloadBytes = DEFAULT_GENERATION_DOWNLOAD_BYTES;
  readonly totalLanguageModelBytes = LANGUAGE_MODEL_DOWNLOAD_BYTES;
  readonly totalBrowserModelBytes = FULL_MODEL_DOWNLOAD_BYTES;

  private readonly workerUrl: URL;
  private readonly languageWorkerUrl: URL;
  private readonly workerFactory: (
    url: URL,
    options: WorkerOptions,
  ) => Worker;
  private readonly languageWorkerFactory: (
    url: URL,
    options: WorkerOptions,
  ) => Worker;
  private readonly lyricsModelId: string;
  private readonly lyricsModelRevision: string;
  private readonly plannerModelId: string;
  private readonly plannerModelRevision: string;
  private readonly highQualityPlannerModelId: string;
  private readonly highQualityPlannerModelRevision: string;
  private readonly assets: WorkerAssetConfig;
  private readonly defaultAllowWasmFallback: boolean;
  private readonly listeners = new Set<AceStepUpdateListener>();
  private worker: Worker | null = null;
  private cancelActive: ((reason: Error) => void) | null = null;
  private disposed = false;
  private operationProgress = 0;

  constructor(options: AceStepWebGpuOptions = {}) {
    this.workerUrl = options.workerUrl
      ? absoluteUrl(options.workerUrl)
      : new URL("./ace-step.worker.js", import.meta.url);
    this.languageWorkerUrl = options.languageWorkerUrl
      ? absoluteUrl(options.languageWorkerUrl)
      : new URL("./language.worker.js", import.meta.url);
    this.workerFactory =
      options.workerFactory ??
      ((url, workerOptions) => new Worker(url, workerOptions));
    this.languageWorkerFactory =
      options.languageWorkerFactory ??
      options.workerFactory ??
      ((url, workerOptions) => new Worker(url, workerOptions));
    this.lyricsModelId =
      options.lyricsModelId?.trim() || DEFAULT_LYRICS_MODEL;
    this.lyricsModelRevision =
      options.lyricsModelRevision?.trim() ||
      DEFAULT_LYRICS_MODEL_REVISION;
    this.plannerModelId =
      options.plannerModelId?.trim() || DEFAULT_PLANNER_MODEL;
    this.plannerModelRevision =
      options.plannerModelRevision?.trim() ||
      DEFAULT_PLANNER_MODEL_REVISION;
    this.highQualityPlannerModelId =
      options.highQualityPlannerModelId?.trim() ||
      DEFAULT_HIGH_QUALITY_PLANNER_MODEL;
    this.highQualityPlannerModelRevision =
      options.highQualityPlannerModelRevision?.trim() ||
      DEFAULT_HIGH_QUALITY_PLANNER_MODEL_REVISION;
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

  /** Latest normalized progress value emitted by the active operation. */
  get progress() {
    return this.operationProgress;
  }

  subscribe(listener: AceStepUpdateListener) {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  }

  private notify(update: WorkerUpdate) {
    if (update.type === "progress") {
      this.operationProgress = update.progress;
    }
    for (const listener of this.listeners) {
      try {
        listener(update);
      } catch (error) {
        console.error("ACE-Step update listener failed", error);
      }
    }
  }

  private async requestPersistentStorage() {
    if (
      typeof navigator === "undefined" ||
      !navigator.storage?.persist
    ) {
      return;
    }
    try {
      if (!(await navigator.storage.persisted?.())) {
        await navigator.storage.persist();
      }
    } catch {
      // Compatibility and quota diagnostics are reported by the Worker.
    }
  }

  private createWorker(
    url = this.workerUrl,
    factory = this.workerFactory,
    name = "ai-music-js",
  ) {
    if (this.disposed) {
      throw new Error("This ACE-Step WebGPU instance has been disposed.");
    }
    if (this.worker) {
      throw new Error("An ACE-Step Worker operation is already active.");
    }
    const worker = factory(url, {
      type: "module",
      name,
    });
    this.worker = worker;
    return worker;
  }

  private runWorker<T>(
    request: WorkerRequest | LanguageWorkerRequest,
    handleUpdate: (update: WorkerUpdate) => WorkerOutcome<T>,
    signal?: AbortSignal,
    workerConfig?: {
      url: URL;
      factory: (url: URL, options: WorkerOptions) => Worker;
      name: string;
    },
    progressTracker?: WorkerProgressTracker,
  ): Promise<T> {
    if (signal?.aborted) {
      return Promise.reject(
        new DOMException("The ACE-Step operation was aborted.", "AbortError"),
      );
    }

    let worker: Worker;
    try {
      worker = this.createWorker(
        workerConfig?.url,
        workerConfig?.factory,
        workerConfig?.name,
      );
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
        progressTracker?.observe(update);
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

  async writeLyrics(options: WriteLyricsOptions): Promise<WriteLyricsResult> {
    const reporter = new OperationProgressReporter(
      "write-lyrics",
      (update) => this.notify(update),
    );
    reporter.report(0, "starting", "Preparing the lyric writer.");
    try {
      const result = await this.writeLyricsInternal(
        options,
        reporter,
        { start: 0, end: 1 },
      );
      reporter.report(1, "complete", "Lyrics are ready.");
      return result;
    } catch (error) {
      reporter.report(
        reporter.current(),
        "stopped",
        error instanceof Error ? error.message : String(error),
      );
      throw error;
    }
  }

  private async writeLyricsInternal(
    options: WriteLyricsOptions,
    reporter: OperationProgressReporter,
    range: ProgressRange,
  ): Promise<WriteLyricsResult> {
    const prompt = options.prompt.trim();
    const seed = options.seed ?? 42;
    const durationSeconds =
      options.durationSeconds ?? DEFAULT_DURATION_SECONDS;
    const maxWords =
      options.maxWords ?? defaultMaxLyricWords(durationSeconds);
    if (!prompt) {
      throw new TypeError("A non-empty song brief is required.");
    }
    if (
      !Number.isInteger(seed) ||
      seed < 0 ||
      seed > 0x7fff_ffff
    ) {
      throw new TypeError(
        "Lyric seed must be an integer from 0 through 2147483647.",
      );
    }
    if (
      !Number.isInteger(durationSeconds) ||
      durationSeconds < MIN_DURATION_SECONDS ||
      durationSeconds > MAX_DURATION_SECONDS
    ) {
      throw new RangeError(
        `Duration must be a whole number from ${MIN_DURATION_SECONDS} to ${MAX_DURATION_SECONDS} seconds.`,
      );
    }
    if (
      !Number.isInteger(maxWords) ||
      maxWords < 20 ||
      maxWords > 450
    ) {
      throw new RangeError(
        "Maximum lyric words must be a whole number from 20 through 450.",
      );
    }
    await this.requestPersistentStorage();
    return this.runWorker<WriteLyricsResult>(
      {
        type: "write-lyrics",
        prompt,
        seed,
        durationSeconds,
        maxWords,
        modelId: this.lyricsModelId,
        revision: this.lyricsModelRevision,
      },
      (update) =>
        update.type === "lyrics-complete"
          ? {
              done: true,
              value: {
                lyrics: update.lyrics,
                model: update.model,
                revision: update.revision,
                seed: update.seed,
                durationSeconds: update.durationSeconds,
                maxWords: update.maxWords,
                attempts: update.attempts,
                timings: update.timings,
              },
            }
          : { done: false },
      options.signal,
      {
        url: this.languageWorkerUrl,
        factory: this.languageWorkerFactory,
        name: "ai-music-js-language",
      },
      new WorkerProgressTracker("lyrics", reporter, range),
    );
  }

  async planMusic(options: PlanMusicOptions): Promise<PlanMusicResult> {
    const reporter = new OperationProgressReporter(
      "plan-music",
      (update) => this.notify(update),
    );
    reporter.report(0, "starting", "Preparing the music planner.");
    try {
      const result = await this.planMusicInternal(
        options,
        reporter,
        { start: 0, end: 1 },
      );
      reporter.report(1, "complete", "Music plan is ready.");
      return result;
    } catch (error) {
      reporter.report(
        reporter.current(),
        "stopped",
        error instanceof Error ? error.message : String(error),
      );
      throw error;
    }
  }

  private async planMusicInternal(
    options: PlanMusicOptions,
    reporter: OperationProgressReporter,
    range: ProgressRange,
  ): Promise<PlanMusicResult> {
    const prompt = options.prompt.trim();
    const plannerQuality =
      options.plannerQuality ?? DEFAULT_PLANNER_QUALITY;
    const audioQuality =
      options.audioQuality ?? DEFAULT_AUDIO_QUALITY;
    const lyrics = options.lyrics?.trim() ?? "";
    const vocalLanguage = options.vocalLanguage?.trim() || "unknown";
    const seed = options.seed ?? 42;
    const durationSeconds =
      options.durationSeconds ?? DEFAULT_DURATION_SECONDS;
    const autoDuration = options.autoDuration ?? false;
    if (!prompt) {
      throw new TypeError("A non-empty music prompt is required.");
    }
    if (!isPlannerQuality(plannerQuality)) {
      throw new RangeError(
        `Unknown planner quality: ${String(plannerQuality)}.`,
      );
    }
    if (!isAudioQuality(audioQuality)) {
      throw new RangeError(
        `Unknown audio quality: ${String(audioQuality)}.`,
      );
    }
    if (
      !Number.isInteger(seed) ||
      seed < 0 ||
      seed > 0xffff_ffff
    ) {
      throw new TypeError("Planner seed must be an unsigned 32-bit integer.");
    }
    if (
      !Number.isInteger(durationSeconds) ||
      durationSeconds < MIN_DURATION_SECONDS ||
      durationSeconds > MAX_DURATION_SECONDS
    ) {
      throw new RangeError(
        `Duration must be a whole number from ${MIN_DURATION_SECONDS} to ${MAX_DURATION_SECONDS} seconds.`,
      );
    }
    buildLyricPrompt(lyrics, vocalLanguage);
    const recommendedDurationSeconds = lyrics
      ? recommendDurationForLyrics(lyrics, {
          minimumDurationSeconds: Math.max(30, durationSeconds),
          maximumDurationSeconds: MAX_DURATION_SECONDS,
        })
      : durationSeconds;
    await this.requestPersistentStorage();
    return this.runWorker<PlanMusicResult>(
      {
        type: "plan-music",
        plannerQuality,
        audioQuality,
        prompt,
        lyrics,
        vocalLanguage,
        seed: seed % 2_147_483_648,
        durationSeconds,
        autoDuration,
        recommendedDurationSeconds,
        modelId: this.plannerModelId,
        revision: this.plannerModelRevision,
        highQualityModelId: this.highQualityPlannerModelId,
        highQualityRevision: this.highQualityPlannerModelRevision,
        assets: this.assets,
      },
      (update) =>
        update.type === "plan-complete"
          ? {
              done: true,
              value: {
                semanticCodeIds: update.semanticCodeIds,
                plannerQuality: update.plannerQuality,
                metadata: update.metadata,
                reasoning: update.reasoning,
                model: update.model,
                revision: update.revision,
                seed: update.seed,
                timings: update.timings,
                plannerProfile: update.plannerProfile,
              },
            }
          : { done: false },
      options.signal,
      {
        url: this.languageWorkerUrl,
        factory: this.languageWorkerFactory,
        name: "ai-music-js-planner",
      },
      new WorkerProgressTracker("planner", reporter, range),
    );
  }

  async generate(options: GenerateOptions): Promise<AceStepGenerationResult> {
    const reporter = new OperationProgressReporter(
      "generate",
      (update) => this.notify(update),
    );
    reporter.report(0, "starting", "Preparing music generation.");
    try {
      const result = await this.generateInternal(
        options,
        reporter,
        { start: 0, end: 1 },
      );
      reporter.report(1, "complete", "Audio is ready.");
      return result;
    } catch (error) {
      reporter.report(
        reporter.current(),
        "stopped",
        error instanceof Error ? error.message : String(error),
      );
      throw error;
    }
  }

  private async generateInternal(
    options: GenerateOptions,
    reporter: OperationProgressReporter,
    range: ProgressRange,
  ): Promise<AceStepGenerationResult> {
    const pipelineStartedAt = performance.now();
    const pipelineTimings: Record<string, number> = {};
    const finishPipelineStage = (
      stage: string,
      startedAt: number,
      notify = true,
    ) => {
      const milliseconds = performance.now() - startedAt;
      pipelineTimings[stage] = milliseconds;
      if (notify) {
        this.notify({ type: "timing", stage, milliseconds });
      }
      return milliseconds;
    };
    const prompt = options.prompt.trim();
    const audioQuality =
      options.audioQuality ?? DEFAULT_AUDIO_QUALITY;
    const plannerQuality =
      options.plannerQuality ?? DEFAULT_PLANNER_QUALITY;
    let lyrics = options.lyrics?.trim() ?? "";
    const vocalLanguage = options.vocalLanguage?.trim() || "unknown";
    const seed = options.seed ?? 42;
    const durationSeconds =
      options.durationSeconds ?? DEFAULT_DURATION_SECONDS;
    const autoDuration = options.autoDuration ?? false;
    const usesLyricsWriter = options.writeLyrics === true;
    const usesPlanner =
      (options.semanticPlanning ?? true) &&
      plannerQuality === "high-quality" &&
      options.semanticCodeIds === undefined;
    const lyricsShare = usesLyricsWriter
      ? usesPlanner
        ? 0.12
        : 0.15
      : 0;
    const plannerShare = usesPlanner
      ? usesLyricsWriter
        ? 0.4
        : 0.45
      : 0;
    const lyricsProgressRange = progressSubrange(
      range,
      0,
      lyricsShare,
    );
    const plannerProgressRange = progressSubrange(
      range,
      lyricsShare,
      lyricsShare + plannerShare,
    );
    const audioProgressRange = progressSubrange(
      range,
      lyricsShare + plannerShare,
      1,
    );
    let sampler: SamplerMode;
    let dcw;
    try {
      sampler = validateSamplerMode(options.sampler ?? "euler");
      dcw = resolveDcwOptions(options.dcw);
    } catch (error) {
      throw error;
    }
    if (!prompt) {
      throw new TypeError("A non-empty music prompt is required.");
    }
    if (!isAudioQuality(audioQuality)) {
      throw new RangeError(
        `Unknown audio quality: ${String(audioQuality)}.`,
      );
    }
    if (!isPlannerQuality(plannerQuality)) {
      throw new RangeError(
        `Unknown planner quality: ${String(plannerQuality)}.`,
      );
    }
    if (options.writeLyrics && lyrics) {
      throw new TypeError(
        "Choose either supplied lyrics or writeLyrics, not both.",
      );
    }
    if (options.writeLyrics && sampler === "euler-sde") {
      throw new RangeError(
        "Euler SDE is currently limited to instrumental generation. Use Euler or Heun when Qwen writes vocals.",
      );
    }
    if (options.writeLyrics) {
      const lyricsStartedAt = performance.now();
      const written = await this.writeLyricsInternal(
        {
          prompt,
          seed: seed % 2_147_483_648,
          durationSeconds,
          maxWords: options.maxLyricWords,
          signal: options.signal,
        },
        reporter,
        lyricsProgressRange,
      );
      lyrics = written.lyrics;
      finishPipelineStage("pipeline:lyrics", lyricsStartedAt);
    }
    buildLyricPrompt(lyrics, vocalLanguage);
    const instrumental = isInstrumentalLyrics(lyrics);
    if (hasVocalPromptConflict(prompt, lyrics)) {
      throw new TypeError(
        "Vocal lyrics were supplied, but the music prompt requests an instrumental track without asking for a singer or vocals.",
      );
    }
    if (!instrumental && sampler === "euler-sde") {
      throw new RangeError(
        "Euler SDE is currently limited to instrumental generation because the XL INT4 vocal quality gate fails. Use Euler or Heun for vocals.",
      );
    }
    if (
      !Number.isInteger(seed) ||
      seed < 0 ||
      seed > 0xffff_ffff
    ) {
      throw new TypeError("Seed must be an unsigned 32-bit integer.");
    }
    if (
      !Number.isInteger(durationSeconds) ||
      durationSeconds < MIN_DURATION_SECONDS ||
      durationSeconds > MAX_DURATION_SECONDS
    ) {
      throw new RangeError(
        `Duration must be a whole number from ${MIN_DURATION_SECONDS} to ${MAX_DURATION_SECONDS} seconds.`,
      );
    }
    const recommendedDurationSeconds = instrumental
      ? durationSeconds
      : recommendDurationForLyrics(lyrics, {
          minimumDurationSeconds: Math.max(30, durationSeconds),
          maximumDurationSeconds: MAX_DURATION_SECONDS,
        });

    let plan: PlanMusicResult | undefined;
    const hasSemanticCodes = options.semanticCodeIds !== undefined;
    const hasPlannerMetadata = options.plannerMetadata !== undefined;
    if (hasSemanticCodes !== hasPlannerMetadata) {
      throw new TypeError(
        "Precomputed semanticCodeIds and plannerMetadata must be supplied together.",
      );
    }
    if (hasSemanticCodes && hasPlannerMetadata) {
      const semanticCodeIds = [...options.semanticCodeIds!];
      const precomputedDuration =
        options.plannerMetadata!.durationSeconds;
      if (
        semanticCodeIds.length !== precomputedDuration * 5 ||
        semanticCodeIds.some(
          (code) =>
            !Number.isInteger(code) ||
            code < 0 ||
            code >= 64_000,
        )
      ) {
        throw new RangeError(
          `Precomputed semantic codes must contain exactly ${precomputedDuration * 5} integer values from 0 through 63999.`,
        );
      }
      plan = {
        semanticCodeIds,
        plannerQuality,
        metadata: options.plannerMetadata!,
        reasoning: "Precomputed ACE planner result.",
        model: "external-precomputed-planner",
        revision: "external",
        seed,
        timings: {},
      };
    } else if (
      (options.semanticPlanning ?? true) &&
      plannerQuality === "high-quality"
    ) {
      const plannerStartedAt = performance.now();
      plan = await this.planMusicInternal(
        {
          prompt,
          plannerQuality,
          audioQuality,
          lyrics,
          vocalLanguage,
          seed,
          durationSeconds,
          autoDuration,
          signal: options.signal,
        },
        reporter,
        plannerProgressRange,
      );
      finishPipelineStage("pipeline:planner", plannerStartedAt);
    } else if (
      (options.semanticPlanning ?? true) &&
      plannerQuality === "turbo"
    ) {
      this.notify({
        type: "diagnostic",
        key: "Turbo model path",
        value:
          "direct XL Turbo text/lyric conditioning · 4B semantic planner skipped",
      });
    }
    const effectivePrompt = plan?.metadata.caption || prompt;
    const effectiveLanguage =
      plan?.metadata.language || vocalLanguage;
    const effectiveDurationSeconds =
      plan?.metadata.durationSeconds ??
      (autoDuration
        ? recommendedDurationSeconds
        : durationSeconds);

    const storageStartedAt = performance.now();
    reporter.report(
      audioProgressRange.start,
      "audio-preparation",
      "Preparing the ACE-Step audio pipeline.",
    );
    await this.requestPersistentStorage();
    finishPipelineStage("pipeline:persistent-storage", storageStartedAt);
    const audioWorkerStartedAt = performance.now();
    const result = await this.runWorker<AceStepGenerationResult>(
        {
          type: "start",
          prompt: effectivePrompt,
          audioQuality,
          lyrics,
          vocalLanguage: effectiveLanguage,
          semanticCodeIds: plan?.semanticCodeIds,
          plannerMetadata: plan?.metadata,
          seed,
          durationSeconds: effectiveDurationSeconds,
          sampler,
          dcw,
          allowWasmFallback:
            options.allowWasmFallback ??
            this.defaultAllowWasmFallback,
          assets: this.assets,
        },
        (update) => {
          if (update.type !== "complete") {
            return { done: false };
          }
          const audioBufferStartedAt = performance.now();
          const value = resultFromUpdate(
            update,
            lyrics,
            audioQuality,
            plan,
          );
          const audioBufferMilliseconds = finishPipelineStage(
            "pipeline:audio-buffer",
            audioBufferStartedAt,
            false,
          );
          value.timings["pipeline:audio-buffer"] =
            audioBufferMilliseconds;
          return {
            done: true,
            value,
          };
        },
        options.signal,
        undefined,
        new WorkerProgressTracker(
          "audio",
          reporter,
          audioProgressRange,
        ),
      );
    finishPipelineStage("pipeline:audio-worker", audioWorkerStartedAt, false);
    finishPipelineStage("pipeline:total", pipelineStartedAt, false);
    Object.assign(result.timings, pipelineTimings);
    return result;
  }

  /**
   * Runs independent generations one at a time. This intentionally does not
   * create a larger GPU batch, so peak model memory remains equivalent to one
   * generation.
   */
  async generateBatch(
    options: GenerateBatchOptions,
  ): Promise<AceStepGenerationResult[]> {
    if (
      !Array.isArray(options.seeds) ||
      options.seeds.length < 1 ||
      options.seeds.length > 8
    ) {
      throw new RangeError(
        "Sequential batch seeds must contain from 1 through 8 values.",
      );
    }
    if (
      options.seeds.some(
        (seed) =>
          !Number.isInteger(seed) ||
          seed < 0 ||
          seed > 0xffff_ffff,
      )
    ) {
      throw new TypeError(
        "Every sequential batch seed must be an unsigned 32-bit integer.",
      );
    }
    const { seeds, ...generationOptions } = options;
    const reporter = new OperationProgressReporter(
      "generate-batch",
      (update) => this.notify(update),
    );
    reporter.report(0, "starting", `Preparing ${seeds.length} results.`);
    const results: AceStepGenerationResult[] = [];
    try {
      for (const [index, seed] of seeds.entries()) {
        this.notify({
          type: "batch-progress",
          index,
          total: seeds.length,
          seed,
          status: "started",
        });
        const itemRange = {
          start: index / seeds.length,
          end: (index + 1) / seeds.length,
        };
        reporter.report(
          itemRange.start,
          "batch-item",
          `Generating result ${index + 1}/${seeds.length} with seed ${seed}.`,
        );
        const result = await this.generateInternal(
          {
            ...generationOptions,
            seed,
          },
          reporter,
          itemRange,
        );
        results.push(result);
        this.notify({
          type: "batch-progress",
          index,
          total: seeds.length,
          seed,
          status: "complete",
        });
      }
    } catch (error) {
      reporter.report(
        reporter.current(),
        "stopped",
        error instanceof Error ? error.message : String(error),
      );
      throw error;
    }
    reporter.report(1, "complete", `${results.length} results are ready.`);
    return results;
  }

  /**
   * Lists the current origin's ACE-Step files, grouped by model component.
   * Browser storage is origin-scoped, so a different hostname has a separate
   * inventory.
   */
  listCachedModels(signal?: AbortSignal): Promise<CacheInventory> {
    return this.runWorker<CacheInventory>(
      { type: "list-cache", assets: this.assets },
      (update) =>
        update.type === "cache-inventory"
          ? { done: true, value: update.inventory }
          : { done: false },
      signal,
    );
  }

  /**
   * Removes one component returned by listCachedModels(), including all of its
   * graph and external-data files.
   */
  removeCachedModel(
    modelId: string,
    signal?: AbortSignal,
  ): Promise<CacheInventory> {
    const normalizedId = modelId.trim();
    if (
      !ALL_ASSETS.some((asset) => asset.group === normalizedId) &&
      !LANGUAGE_MODEL_COMPONENTS.some(
        (component) => component.id === normalizedId,
      )
    ) {
      return Promise.reject(
        new RangeError(`Unknown cached model component: ${modelId}`),
      );
    }
    return this.runWorker<CacheInventory>(
      {
        type: "remove-cached-model",
        modelId: normalizedId,
        assets: this.assets,
      },
      (update) =>
        update.type === "cache-inventory"
          ? { done: true, value: update.inventory }
          : { done: false },
      signal,
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
  lyrics: string,
  audioQuality: AudioQuality,
  plan?: PlanMusicResult,
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
    seed: update.seed,
    audioQuality,
    sampler: update.sampler,
    instrumental: update.instrumental,
    lyrics,
    plan: plan
      ? {
          plannerQuality: plan.plannerQuality,
          metadata: plan.metadata,
          reasoning: plan.reasoning,
          model: plan.model,
          revision: plan.revision,
          semanticCodeCount: plan.semanticCodeIds.length,
          timings: plan.timings,
        }
      : undefined,
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
