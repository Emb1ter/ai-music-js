import {
  DEFAULT_LYRICS_MODEL,
  DEFAULT_LYRICS_MODEL_REVISION,
} from "./lyrics";
import {
  BUILT_IN_LYRICS_MODELS,
  QWEN35_2B_MODEL,
  QWEN35_2B_MODEL_REVISION,
  RHYMEAI_GEMMA4_E2B_MODEL,
  RHYMEAI_GEMMA4_E2B_MODEL_BASE_URL,
  RHYMEAI_GEMMA4_E2B_MODEL_REVISION,
} from "./lyrics-models";
import {
  DEFAULT_PLANNER_MODEL,
  DEFAULT_PLANNER_MODEL_REVISION,
} from "./planner";
import {
  DEFAULT_HIGH_QUALITY_PLANNER_MODEL,
  DEFAULT_HIGH_QUALITY_PLANNER_MODEL_REVISION,
} from "./planner-quality";

export const LANGUAGE_CACHE_NAME = "ai-music-js-transformers-v1";

export type LanguageModelAsset = {
  fileName: string;
  bytes: number;
  role: "graph" | "weights" | "tokenizer";
};

export type LanguageModelComponent = {
  id:
    | "music-planner"
    | "music-planner-high-quality"
    | "lyrics-writer"
    | "lyrics-writer-qwen-2b"
    | "lyrics-writer-rhymeai-e2b"
    | "lyrics-writer-rhymeai-e4b-v3";
  label: string;
  modelId: string;
  revision: string;
  modelBaseUrl?: string;
  assets: readonly LanguageModelAsset[];
};

export const PLANNER_MODEL_ASSETS = [
  { fileName: "config.json", bytes: 1_794, role: "tokenizer" },
  {
    fileName: "generation_config.json",
    bytes: 147,
    role: "tokenizer",
  },
  {
    fileName: "tokenizer.json",
    bytes: 24_321_939,
    role: "tokenizer",
  },
  {
    fileName: "tokenizer_config.json",
    bytes: 14_077_283,
    role: "tokenizer",
  },
  {
    fileName: "onnx/model_quantized.onnx",
    bytes: 229_008_251,
    role: "graph",
  },
  {
    fileName: "onnx/model_quantized.onnx_data",
    bytes: 1_256_343_680,
    role: "weights",
  },
  {
    fileName: "onnx/model_quantized.onnx_data_1",
    bytes: 1_249_443_840,
    role: "weights",
  },
  {
    fileName: "onnx/model_quantized.onnx_data_2",
    bytes: 264_437_760,
    role: "weights",
  },
  {
    fileName: "lm_head_q8.bin",
    bytes: 556_042_240,
    role: "weights",
  },
  {
    fileName: "lm_head_q8_scales.f16",
    bytes: 34_752_640,
    role: "weights",
  },
] as const satisfies readonly LanguageModelAsset[];

export const LYRICS_MODEL_ASSETS = [
  { fileName: "config.json", bytes: 1_989, role: "tokenizer" },
  {
    fileName: "generation_config.json",
    bytes: 120,
    role: "tokenizer",
  },
  {
    fileName: "tokenizer.json",
    bytes: 19_226_111,
    role: "tokenizer",
  },
  {
    fileName: "tokenizer_config.json",
    bytes: 9_119,
    role: "tokenizer",
  },
  {
    fileName: "onnx/model_q4f16.onnx",
    bytes: 597_442,
    role: "graph",
  },
  {
    fileName: "onnx/model_q4f16.onnx_data",
    bytes: 469_331_968,
    role: "weights",
  },
] as const satisfies readonly LanguageModelAsset[];

export const QWEN35_2B_MODEL_ASSETS = [
  { fileName: "config.json", bytes: 2_993, role: "tokenizer" },
  {
    fileName: "generation_config.json",
    bytes: 248,
    role: "tokenizer",
  },
  {
    fileName: "chat_template.jinja",
    bytes: 7_755,
    role: "tokenizer",
  },
  {
    fileName: "tokenizer.json",
    bytes: 19_226_111,
    role: "tokenizer",
  },
  {
    fileName: "tokenizer_config.json",
    bytes: 9_161,
    role: "tokenizer",
  },
  {
    fileName: "onnx/decoder_model_merged_q4f16.onnx",
    bytes: 707_377,
    role: "graph",
  },
  {
    fileName: "onnx/decoder_model_merged_q4f16.onnx_data",
    bytes: 1_088_892_928,
    role: "weights",
  },
  {
    fileName: "onnx/embed_tokens_q4f16.onnx",
    bytes: 1_064,
    role: "graph",
  },
  {
    fileName: "onnx/embed_tokens_q4f16.onnx_data",
    bytes: 294_010_880,
    role: "weights",
  },
] as const satisfies readonly LanguageModelAsset[];

export const RHYMEAI_GEMMA4_E2B_MODEL_ASSETS = [
  { fileName: "config.json", bytes: 5_549, role: "tokenizer" },
  {
    fileName: "generation_config.json",
    bytes: 238,
    role: "tokenizer",
  },
  {
    fileName: "chat_template.jinja",
    bytes: 16_317,
    role: "tokenizer",
  },
  {
    fileName: "tokenizer.json",
    bytes: 19_439_251,
    role: "tokenizer",
  },
  {
    fileName: "tokenizer_config.json",
    bytes: 18_807,
    role: "tokenizer",
  },
  {
    fileName: "onnx/decoder_model_merged_q4f16.onnx",
    bytes: 673_231,
    role: "graph",
  },
  {
    fileName: "onnx/decoder_model_merged_q4f16.onnx_data",
    bytes: 1_519_700_992,
    role: "weights",
  },
  {
    fileName: "onnx/embed_tokens_q4f16.onnx",
    bytes: 5_621,
    role: "graph",
  },
  {
    fileName: "onnx/embed_tokens_q4f16.onnx_data",
    bytes: 1_590_689_792,
    role: "weights",
  },
] as const satisfies readonly LanguageModelAsset[];

export const RHYMEAI_GEMMA4_E4B_V3_MODEL_ASSETS = [
  { fileName: "config.json", bytes: 5_741, role: "tokenizer" },
  {
    fileName: "generation_config.json",
    bytes: 238,
    role: "tokenizer",
  },
  {
    fileName: "chat_template.jinja",
    bytes: 16_317,
    role: "tokenizer",
  },
  {
    fileName: "tokenizer.json",
    bytes: 19_439_251,
    role: "tokenizer",
  },
  {
    fileName: "tokenizer_config.json",
    bytes: 18_807,
    role: "tokenizer",
  },
  {
    fileName: "onnx/decoder_model_merged_q4f16.onnx",
    bytes: 850_610,
    role: "graph",
  },
  {
    fileName: "onnx/decoder_model_merged_q4f16.onnx_data",
    bytes: 2_074_847_232,
    role: "weights",
  },
  {
    fileName: "onnx/decoder_model_merged_q4f16.onnx_data_1",
    bytes: 812_318_720,
    role: "weights",
  },
  {
    fileName: "onnx/embed_tokens_q4f16.onnx",
    bytes: 5_619,
    role: "graph",
  },
  {
    fileName: "onnx/embed_tokens_q4f16.onnx_data",
    bytes: 2_017_460_224,
    role: "weights",
  },
] as const satisfies readonly LanguageModelAsset[];

export const HIGH_QUALITY_PLANNER_MODEL_ASSETS = [
  { fileName: "config.json", bytes: 1_762, role: "tokenizer" },
  {
    fileName: "tokenizer.json",
    bytes: 24_321_939,
    role: "tokenizer",
  },
  {
    fileName: "tokenizer_config.json",
    bytes: 14_077_283,
    role: "tokenizer",
  },
  {
    fileName: "int8-fp32/onnx/model.onnx",
    bytes: 502_318,
    role: "graph",
  },
  {
    fileName: "int8-fp32/onnx/model.onnx.data.0",
    bytes: 947_312_640,
    role: "weights",
  },
  {
    fileName: "int8-fp32/onnx/model.onnx.data.1",
    bytes: 941_137_920,
    role: "weights",
  },
  {
    fileName: "int8-fp32/onnx/model.onnx.data.2",
    bytes: 935_485_440,
    role: "weights",
  },
  {
    fileName: "int8-fp32/onnx/model.onnx.data.3",
    bytes: 925_593_600,
    role: "weights",
  },
  {
    fileName: "int8-fp32/onnx/model.onnx.data.4",
    bytes: 189_358_080,
    role: "weights",
  },
  ...Array.from({ length: 7 }, (_, index) => ({
    fileName: `fp32-head/audio_head_${index}.bin`,
    bytes: 83_886_080,
    role: "weights" as const,
  })),
  {
    fileName: "fp32-head/audio_head_7.bin",
    bytes: 68_157_440,
    role: "weights",
  },
] as const satisfies readonly LanguageModelAsset[];

export const LANGUAGE_MODEL_COMPONENTS = [
  {
    id: "music-planner",
    label: "ACE-Step 5 Hz planner 4B · split Q6 body / WebGPU Q8 embedding+head",
    modelId: DEFAULT_PLANNER_MODEL,
    revision: DEFAULT_PLANNER_MODEL_REVISION,
    assets: PLANNER_MODEL_ASSETS,
  },
  {
    id: "music-planner-high-quality",
    label:
      "ACE-Step 5 Hz planner 4B · INT8 weights / FP32 compute high quality",
    modelId: DEFAULT_HIGH_QUALITY_PLANNER_MODEL,
    revision: DEFAULT_HIGH_QUALITY_PLANNER_MODEL_REVISION,
    assets: HIGH_QUALITY_PLANNER_MODEL_ASSETS,
  },
  {
    id: "lyrics-writer",
    label: "Qwen3.5 lyric writer 0.8B · INT4",
    modelId: DEFAULT_LYRICS_MODEL,
    revision: DEFAULT_LYRICS_MODEL_REVISION,
    assets: LYRICS_MODEL_ASSETS,
  },
  {
    id: "lyrics-writer-qwen-2b",
    label: "Qwen3.5 lyric writer 2B · Q4F16",
    modelId: QWEN35_2B_MODEL,
    revision: QWEN35_2B_MODEL_REVISION,
    assets: QWEN35_2B_MODEL_ASSETS,
  },
  {
    id: "lyrics-writer-rhymeai-e2b",
    label: "RhymeAI Gemma 4 E2B lyric writer · Q4F16",
    modelId: RHYMEAI_GEMMA4_E2B_MODEL,
    revision: RHYMEAI_GEMMA4_E2B_MODEL_REVISION,
    modelBaseUrl: RHYMEAI_GEMMA4_E2B_MODEL_BASE_URL,
    assets: RHYMEAI_GEMMA4_E2B_MODEL_ASSETS,
  },
  {
    id: "lyrics-writer-rhymeai-e4b-v3",
    label: "RhymeAI Gemma 4 E4B v3 lyric writer · Q4F16",
    modelId:
      BUILT_IN_LYRICS_MODELS["rhymeai-gemma4-e4b-v3-q4f16"].modelId,
    revision:
      BUILT_IN_LYRICS_MODELS["rhymeai-gemma4-e4b-v3-q4f16"].revision,
    assets: RHYMEAI_GEMMA4_E4B_V3_MODEL_ASSETS,
  },
] as const satisfies readonly LanguageModelComponent[];

export const PLANNER_MODEL_DOWNLOAD_BYTES = PLANNER_MODEL_ASSETS.reduce(
  (sum, asset) => sum + asset.bytes,
  0,
);

export const LYRICS_MODEL_DOWNLOAD_BYTES = LYRICS_MODEL_ASSETS.reduce(
  (sum, asset) => sum + asset.bytes,
  0,
);

export const QWEN35_2B_MODEL_DOWNLOAD_BYTES =
  QWEN35_2B_MODEL_ASSETS.reduce((sum, asset) => sum + asset.bytes, 0);

export const RHYMEAI_GEMMA4_E2B_MODEL_DOWNLOAD_BYTES =
  RHYMEAI_GEMMA4_E2B_MODEL_ASSETS.reduce(
    (sum, asset) => sum + asset.bytes,
    0,
  );

export const RHYMEAI_GEMMA4_E4B_V3_MODEL_DOWNLOAD_BYTES =
  RHYMEAI_GEMMA4_E4B_V3_MODEL_ASSETS.reduce(
    (sum, asset) => sum + asset.bytes,
    0,
  );

export const HIGH_QUALITY_PLANNER_MODEL_DOWNLOAD_BYTES =
  HIGH_QUALITY_PLANNER_MODEL_ASSETS.reduce(
    (sum, asset) => sum + asset.bytes,
    0,
  );

export const LANGUAGE_MODEL_DOWNLOAD_BYTES =
  PLANNER_MODEL_DOWNLOAD_BYTES +
  HIGH_QUALITY_PLANNER_MODEL_DOWNLOAD_BYTES +
  LYRICS_MODEL_DOWNLOAD_BYTES +
  QWEN35_2B_MODEL_DOWNLOAD_BYTES +
  RHYMEAI_GEMMA4_E2B_MODEL_DOWNLOAD_BYTES +
  RHYMEAI_GEMMA4_E4B_V3_MODEL_DOWNLOAD_BYTES;
