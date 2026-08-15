import {
  DEFAULT_LYRICS_MODEL,
  DEFAULT_LYRICS_MODEL_REVISION,
} from "./lyrics";

export type LyricsModelFamily = "auto" | "qwen3.5" | "gemma4";

export type LyricsModelDtype =
  | "q4"
  | "q4f16"
  | "q8"
  | "fp16"
  | "fp32";

export type LyricsModelConfig = {
  /** Stable application-facing identifier. Defaults to `custom:<modelId>`. */
  id?: string;
  /** Human-readable name used in progress messages. */
  label?: string;
  /** Supported Transformers.js architecture loader. `auto` uses AutoModelForCausalLM. */
  family: LyricsModelFamily;
  /** Hugging Face repository ID, or the directory below modelBaseUrl. */
  modelId: string;
  /** Immutable Hub revision when remotely hosted. Defaults to `main`. */
  revision?: string;
  /** ONNX variant suffix selected by Transformers.js. */
  dtype: LyricsModelDtype;
  /**
   * Optional parent URL for self-hosted Transformers.js files. The loader
   * reads `<modelBaseUrl>/<modelId>/config.json`, tokenizer files and ONNX
   * graphs without contacting Hugging Face.
   */
  modelBaseUrl?: string | URL;
  /** Optional application-supplied model-specific system prompt. */
  systemPrompt?: string;
};

export const DEFAULT_LYRICS_MODEL_PROFILE_ID =
  "qwen3.5-0.8b-q4" as const;

export const QWEN35_2B_MODEL =
  "onnx-community/Qwen3.5-2B-ONNX-OPT";

export const QWEN35_2B_MODEL_REVISION =
  "2ea7886f48b926aca97de8b0e041ffca7e3ebaa9";

export const RHYMEAI_GEMMA4_E2B_MODEL =
  "Shayde182/rhymeai-gemma-4-gguf";

export const RHYMEAI_GEMMA4_E2B_MODEL_REVISION =
  "27492c6696f2f868bbbe3aeafb633009ef932598";

/**
 * The E2B RhymeAI adapter is distributed as GGUF. ai-music-js uses the
 * locally converted, text-only ONNX export below this same-origin path.
 */
export const RHYMEAI_GEMMA4_E2B_MODEL_BASE_URL =
  "/local-lyrics-models/";

export const RHYMEAI_GEMMA4_E4B_V3_MODEL =
  "emb1ter/RhymeAI-Gemma-4-E4B-v3-ONNX-WebGPU";

export const BUILT_IN_LYRICS_MODELS = {
  [DEFAULT_LYRICS_MODEL_PROFILE_ID]: {
    id: DEFAULT_LYRICS_MODEL_PROFILE_ID,
    label: "Qwen3.5 0.8B · Q4 · 0.49 GB",
    family: "qwen3.5",
    modelId: DEFAULT_LYRICS_MODEL,
    revision: DEFAULT_LYRICS_MODEL_REVISION,
    dtype: "q4",
  },
  "qwen3.5-2b-q4f16": {
    id: "qwen3.5-2b-q4f16",
    label: "Qwen3.5 2B · Q4F16 · 1.40 GB",
    family: "qwen3.5",
    modelId: QWEN35_2B_MODEL,
    revision: QWEN35_2B_MODEL_REVISION,
    dtype: "q4f16",
  },
  "rhymeai-gemma4-e2b-q4f16": {
    id: "rhymeai-gemma4-e2b-q4f16",
    label: "RhymeAI Gemma 4 E2B · Q4F16 · 3.13 GB",
    family: "gemma4",
    modelId: RHYMEAI_GEMMA4_E2B_MODEL,
    revision: RHYMEAI_GEMMA4_E2B_MODEL_REVISION,
    dtype: "q4f16",
    modelBaseUrl: RHYMEAI_GEMMA4_E2B_MODEL_BASE_URL,
  },
  "rhymeai-gemma4-e4b-v3-q4f16": {
    id: "rhymeai-gemma4-e4b-v3-q4f16",
    label: "RhymeAI Gemma 4 E4B v3 · Q4F16 · 4.92 GB",
    family: "gemma4",
    modelId: RHYMEAI_GEMMA4_E4B_V3_MODEL,
    revision: "12495b6e18df4ae1738e5bb9415d610a7a9a3d92",
    dtype: "q4f16",
  },
} as const satisfies Record<
  string,
  LyricsModelConfig & {
    id: string;
    label: string;
    revision: string;
  }
>;

export type BuiltInLyricsModelId =
  keyof typeof BUILT_IN_LYRICS_MODELS;

export type LyricsModelSelection =
  | BuiltInLyricsModelId
  | LyricsModelConfig;

export type ResolvedLyricsModel = {
  id: string;
  label: string;
  family: LyricsModelFamily;
  modelId: string;
  revision: string;
  dtype: LyricsModelDtype;
  modelBaseUrl?: string;
  systemPrompt?: string;
};

const MODEL_FAMILIES = new Set<LyricsModelFamily>([
  "auto",
  "qwen3.5",
  "gemma4",
]);

const MODEL_DTYPES = new Set<LyricsModelDtype>([
  "q4",
  "q4f16",
  "q8",
  "fp16",
  "fp32",
]);

const nonEmpty = (value: string | undefined, name: string) => {
  const normalized = value?.trim() ?? "";
  if (!normalized) {
    throw new TypeError(`${name} must be a non-empty string.`);
  }
  return normalized;
};

export const resolveLyricsModel = (
  selection: LyricsModelSelection = DEFAULT_LYRICS_MODEL_PROFILE_ID,
): ResolvedLyricsModel => {
  const selected: LyricsModelConfig =
    typeof selection === "string"
      ? BUILT_IN_LYRICS_MODELS[selection]
      : selection;
  if (!selected) {
    throw new RangeError(`Unknown built-in lyric model: ${selection}.`);
  }

  const family = selected.family;
  const dtype = selected.dtype;
  if (!MODEL_FAMILIES.has(family)) {
    throw new RangeError(`Unsupported lyric-model family: ${String(family)}.`);
  }
  if (!MODEL_DTYPES.has(dtype)) {
    throw new RangeError(`Unsupported lyric-model dtype: ${String(dtype)}.`);
  }

  const modelId = nonEmpty(selected.modelId, "lyricsModel.modelId");
  const modelBaseUrl = selected.modelBaseUrl?.toString().trim();
  const systemPrompt = selected.systemPrompt?.trim();
  return {
    id: selected.id?.trim() || `custom:${modelId}`,
    label: selected.label?.trim() || modelId,
    family,
    modelId,
    revision: selected.revision?.trim() || "main",
    dtype,
    ...(modelBaseUrl ? { modelBaseUrl } : {}),
    ...(systemPrompt ? { systemPrompt } : {}),
  };
};
