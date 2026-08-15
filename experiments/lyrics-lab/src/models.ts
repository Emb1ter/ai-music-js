import {
  DEFAULT_LYRICS_MODEL,
  DEFAULT_LYRICS_MODEL_REVISION,
} from "../../../lib/lyrics";

export type LyricLabModelDtype = "q4" | "q4f16";
export type LyricLabModelFamily = "qwen35" | "gemma4";

export type LyricLabModel = {
  id:
    | "qwen35-4b-q4f16"
    | "qwen35-2b-q4f16"
    | "qwen35-2b-hauhaucs-q4f16"
    | "qwen35-08b-q4"
    | "rhymeai-gemma4-e2b-q4f16"
    | "rhymeai-gemma4-e4b-v3-q4f16";
  label: string;
  shortLabel: string;
  modelId: string;
  revision: string;
  dtype: LyricLabModelDtype;
  family: LyricLabModelFamily;
  downloadBytes: number;
  localOnly: boolean;
};

export const LYRIC_LAB_MODELS = [
  {
    id: "qwen35-4b-q4f16",
    label: "Qwen3.5 4B · Q4F16 · about 2.82 GB",
    shortLabel: "Qwen3.5 4B Q4F16",
    modelId: "onnx-community/Qwen3.5-4B-ONNX-OPT",
    revision: "57b13b4dce7be073be0df3eaf1c842a6bbb2e0a7",
    dtype: "q4f16",
    family: "qwen35",
    downloadBytes: 2_820_843_621,
    localOnly: false,
  },
  {
    id: "qwen35-2b-q4f16",
    label: "Qwen3.5 2B · Q4F16 · about 1.40 GB",
    shortLabel: "Qwen3.5 2B Q4F16",
    modelId: "onnx-community/Qwen3.5-2B-ONNX-OPT",
    revision: "2ea7886f48b926aca97de8b0e041ffca7e3ebaa9",
    dtype: "q4f16",
    family: "qwen35",
    downloadBytes: 1_402_858_517,
    localOnly: false,
  },
  {
    id: "qwen35-2b-hauhaucs-q4f16",
    label: "Qwen3.5 2B HauhauCS uncensored · Q4F16 · local 1.40 GB",
    shortLabel: "Qwen3.5 2B HauhauCS Q4F16",
    modelId:
      "DreamFast/Qwen3.5-2B-Uncensored-HauhauCS-Aggressive-Safetensor-Benchmark",
    revision: "9af5b83f777a0c8d3e36d65a1446aac3a463d0d6",
    dtype: "q4f16",
    family: "qwen35",
    downloadBytes: 1_402_858_517,
    localOnly: true,
  },
  {
    id: "qwen35-08b-q4",
    label: "Qwen3.5 0.8B · Q4 · about 0.49 GB (baseline)",
    shortLabel: "Qwen3.5 0.8B Q4",
    modelId: DEFAULT_LYRICS_MODEL,
    revision: DEFAULT_LYRICS_MODEL_REVISION,
    dtype: "q4",
    family: "qwen35",
    downloadBytes: 489_166_749,
    localOnly: false,
  },
  {
    id: "rhymeai-gemma4-e2b-q4f16",
    label: "RhymeAI Gemma 4 E2B · Q4F16 · local 3.13 GB",
    shortLabel: "RhymeAI Gemma 4 E2B Q4F16",
    modelId: "Shayde182/rhymeai-gemma-4-gguf",
    revision: "27492c6696f2f868bbbe3aeafb633009ef932598",
    dtype: "q4f16",
    family: "gemma4",
    downloadBytes: 3_130_549_798,
    localOnly: true,
  },
  {
    id: "rhymeai-gemma4-e4b-v3-q4f16",
    label: "RhymeAI Gemma 4 E4B v3 · Q4F16 · local 4.92 GB",
    shortLabel: "RhymeAI Gemma 4 E4B v3 Q4F16",
    modelId: "ai-music-js/rhymeai-gemma4-e4b-v3",
    revision: "27492c6696f2f868bbbe3aeafb633009ef932598",
    dtype: "q4f16",
    family: "gemma4",
    downloadBytes: 4_924_962_759,
    localOnly: true,
  },
] as const satisfies readonly LyricLabModel[];

export type LyricLabModelId = (typeof LYRIC_LAB_MODELS)[number]["id"];

export const DEFAULT_LYRIC_LAB_MODEL_ID: LyricLabModelId =
  "qwen35-2b-q4f16";

export const lyricLabModelById = (id: string) => {
  const model = LYRIC_LAB_MODELS.find((candidate) => candidate.id === id);
  if (!model) throw new Error(`Unknown lyric-lab model: ${id}`);
  return model;
};
