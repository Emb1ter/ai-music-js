export const ONNX_REPOSITORY = "shreyask/ACE-Step-v1.5-ONNX";
export const ONNX_REVISION = "bdabfb5684fd70fcc76f98cbb51bb9ebc47ee342";
export const XL_MODEL_REPOSITORY =
  "emb1ter/ACE-Step-v1.5-XL-Turbo-ONNX-WebGPU";
export const XL_MODEL_REVISION =
  "0c0e3d4a14aaa9387990e90bf66ec35c7afed25b";
export const DEFAULT_MODEL_BASE_URL =
  `https://huggingface.co/${XL_MODEL_REPOSITORY}/resolve/${XL_MODEL_REVISION}/`;
export const PIPELINE_BUILD = "2026-07-25-xl-turbo-q4-chunked";
export const ACESTEP_REPOSITORY = "ACE-Step/Ace-Step1.5";
export const ACESTEP_REVISION = "19671f406d603126926c1b7e2adc169acbcade22";
export const CACHE_NAME = `ai-music-js-${ONNX_REVISION.slice(0, 8)}`;
export const MODEL_NAME = "ACE-Step 1.5 XL Turbo";
export const MODEL_PARAMETER_COUNT = 4_168_897_088;
export const DIT_ATTENTION_HEADS = 32;
export const ORT_WASM_FILE =
  "/wasm/ort-wasm-simd-threaded.asyncify.wasm";
export const ORT_WASM_MODULE_FILE =
  "/wasm/ort-wasm-simd-threaded.asyncify.mjs";
export const ORT_WASM_BYTES = 24_254_953;
export const ORT_WASM_SHA256 =
  "7e83cd6cee77e478bc96a7e91b198144fb5e4126287daf1f9b54bb195ebcd55a";
export const ORT_WASM_MODULE_BYTES = 47_507;
export const ORT_WASM_MODULE_SHA256 =
  "7236653b8565da4046e459cd0e274123419a1d9f1f8f18fd36c28058346ca655";

export const SAMPLE_RATE = 48_000;
export const DEFAULT_DURATION_SECONDS = 10;
export const MIN_DURATION_SECONDS = 10;
export const MAX_DURATION_SECONDS = 120;
export const LATENT_FRAME_RATE = 25;
export const LATENT_CHANNELS = 64;
export const VAE_UPSAMPLE_FACTOR = 1_920;
export const INFERENCE_STEPS = 8;
export const TURBO_SHIFT = 3;

export const validateDurationSeconds = (durationSeconds: number) => {
  if (
    !Number.isInteger(durationSeconds) ||
    durationSeconds < MIN_DURATION_SECONDS ||
    durationSeconds > MAX_DURATION_SECONDS
  ) {
    throw new RangeError(
      `Duration must be a whole number from ${MIN_DURATION_SECONDS} to ${MAX_DURATION_SECONDS} seconds.`,
    );
  }
  return durationSeconds;
};

export const durationToLatentFrames = (durationSeconds: number) =>
  validateDurationSeconds(durationSeconds) * LATENT_FRAME_RATE;

export const durationToAudioFrames = (durationSeconds: number) =>
  validateDurationSeconds(durationSeconds) * SAMPLE_RATE;

const hf = (repository: string, revision: string, path: string) =>
  `https://huggingface.co/${repository}/resolve/${revision}/${path}`;

export type DownloadAsset = {
  id: string;
  group: string;
  label: string;
  fileName: string;
  url: string;
  bytes: number;
  role: "graph" | "weights" | "tokenizer" | "conditioning";
};

export type GraphId =
  | "text-encoder"
  | "lyric-embedding"
  | "condition-encoder"
  | "dit"
  | "vae";

export type ModelGraph = {
  id: GraphId;
  label: string;
  graph: DownloadAsset;
  weights: DownloadAsset[];
  webGpuOnlyBlockers: string[];
};

const onnxAsset = (
  group: GraphId,
  label: string,
  fileName: string,
  bytes: number,
  role: "graph" | "weights",
  local: boolean,
  idSuffix = "",
): DownloadAsset => ({
  id: `${group}:${role}${idSuffix ? `:${idSuffix}` : ""}`,
  group,
  label,
  fileName,
  url: local
    ? `/models/${fileName}?build=${PIPELINE_BUILD}`
    : hf(ONNX_REPOSITORY, ONNX_REVISION, `onnx/${fileName}`),
  bytes,
  role,
});

const graph = (
  id: GraphId,
  label: string,
  graphFile: string,
  graphBytes: number,
  weightFiles: number | { fileName: string; bytes: number }[],
  webGpuOnlyBlockers: string[] = [],
  local = false,
): ModelGraph => {
  const files =
    typeof weightFiles === "number"
      ? [{ fileName: `${graphFile}.data`, bytes: weightFiles }]
      : weightFiles;
  return {
    id,
    label,
    graph: onnxAsset(id, label, graphFile, graphBytes, "graph", local),
    weights: files.map((file, index) =>
      onnxAsset(
        id,
        label,
        file.fileName,
        file.bytes,
        "weights",
        local,
        files.length > 1 ? String(index) : "",
      ),
    ),
    webGpuOnlyBlockers,
  };
};

export const MODEL_GRAPHS: ModelGraph[] = [
  graph(
    "text-encoder",
    "Qwen3 text encoder · INT4",
    "text_encoder_q4.onnx",
    4_280_307,
    1_684_127_744,
    ["ai.onnx:IsNaN"],
  ),
  graph(
    "lyric-embedding",
    "Lyric token embedding · FP16",
    "text_embed_tokens_fp16.onnx",
    2_208,
    310_618_112,
  ),
  graph(
    "condition-encoder",
    "ACE XL condition encoder · INT4",
    "condition_encoder_xl_turbo_q4.onnx",
    2_136_693,
    347_366_896,
    [],
    true,
  ),
  graph(
    "dit",
    "ACE-Step XL Turbo 4B DiT · INT4",
    "dit_decoder_xl_turbo_q4.onnx",
    9_017_416,
    [
      {
        fileName: "dit_decoder_xl_turbo_q4.onnx.data.0",
        bytes: 949_514_240,
      },
      {
        fileName: "dit_decoder_xl_turbo_q4.onnx.data.1",
        bytes: 942_622_720,
      },
      {
        fileName: "dit_decoder_xl_turbo_q4.onnx.data.2",
        bytes: 811_806_720,
      },
    ],
    [],
    true,
  ),
  graph(
    "vae",
    "Oobleck VAE decoder · FP16",
    "vae_decoder_fp16.onnx",
    88_166,
    168_770_048,
  ),
];

export const SUPPORT_ASSETS: DownloadAsset[] = [
  {
    id: "tokenizer:json",
    group: "tokenizer",
    label: "Qwen3 tokenizer",
    fileName: "tokenizer.json",
    url: hf(
      ACESTEP_REPOSITORY,
      ACESTEP_REVISION,
      "Qwen3-Embedding-0.6B/tokenizer.json",
    ),
    bytes: 11_423_705,
    role: "tokenizer",
  },
  {
    id: "tokenizer:config",
    group: "tokenizer",
    label: "Qwen3 tokenizer config",
    fileName: "tokenizer_config.json",
    url: hf(
      ACESTEP_REPOSITORY,
      ACESTEP_REVISION,
      "Qwen3-Embedding-0.6B/tokenizer_config.json",
    ),
    bytes: 5_404,
    role: "tokenizer",
  },
  {
    id: "conditioning:silence",
    group: "conditioning",
    label: "ACE-Step silence latent",
    fileName: "silence_latent.pt",
    url: hf(
      ACESTEP_REPOSITORY,
      ACESTEP_REVISION,
      "acestep-v15-turbo/silence_latent.pt",
    ),
    bytes: 3_841_215,
    role: "conditioning",
  },
];

export const ALL_ASSETS = [
  ...MODEL_GRAPHS.flatMap(({ graph: graphAsset, weights }) => [
    graphAsset,
    ...weights,
  ]),
  ...SUPPORT_ASSETS,
];

export const TOTAL_DOWNLOAD_BYTES = ALL_ASSETS.reduce(
  (total, asset) => total + asset.bytes,
  0,
);

export const MODEL_DOWNLOAD_BYTES = MODEL_GRAPHS.reduce(
  (total, item) =>
    total +
    item.graph.bytes +
    item.weights.reduce((sum, asset) => sum + asset.bytes, 0),
  0,
);

export const graphById = (id: GraphId) => {
  const value = MODEL_GRAPHS.find((item) => item.id === id);
  if (!value) {
    throw new Error(`Unknown graph: ${id}`);
  }
  return value;
};

export const buildCaptionPrompt = (
  caption: string,
  durationSeconds: number,
) => `# Instruction
Fill the audio semantic mask based on the given conditions:

# Caption
${caption.trim()}

# Metas
- bpm: N/A
- timesignature: N/A
- keyscale: N/A
- duration: ${validateDurationSeconds(durationSeconds)} seconds
<|endoftext|>
`;

export const MAX_LYRICS_CHARACTERS = 4_096;

export const isInstrumentalLyrics = (lyrics?: string) => {
  const normalized = lyrics?.trim();
  return (
    !normalized ||
    normalized.toLowerCase() === "[instrumental]"
  );
};

export const buildLyricPrompt = (
  lyrics?: string,
  vocalLanguage = "unknown",
) => {
  const normalizedLyrics = lyrics?.trim() || "[Instrumental]";
  if (normalizedLyrics.length > MAX_LYRICS_CHARACTERS) {
    throw new RangeError(
      `Lyrics must contain at most ${MAX_LYRICS_CHARACTERS} characters.`,
    );
  }
  const normalizedLanguage = vocalLanguage.trim() || "unknown";
  if (!/^(?:unknown|[a-z]{2,3}(?:-[A-Za-z0-9]{2,8})?)$/i.test(
    normalizedLanguage,
  )) {
    throw new RangeError(
      "Vocal language must be 'unknown' or a language code such as en, es, or zh-Hans.",
    );
  }
  return `# Languages
${normalizedLanguage}

# Lyric
${normalizedLyrics}<|endoftext|>`;
};

export const INSTRUMENTAL_LYRIC_PROMPT = buildLyricPrompt();
