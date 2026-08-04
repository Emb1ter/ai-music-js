export const ONNX_REPOSITORY = "shreyask/ACE-Step-v1.5-ONNX";
export const ONNX_REVISION = "bdabfb5684fd70fcc76f98cbb51bb9ebc47ee342";
export const XL_MODEL_REPOSITORY =
  "emb1ter/ACE-Step-v1.5-XL-Turbo-ONNX-WebGPU";
export const XL_MODEL_REVISION =
  "cf185389395b3a725d948a59262f3ab4be4b0ad8";
export const DEFAULT_MODEL_BASE_URL =
  `https://huggingface.co/${XL_MODEL_REPOSITORY}/resolve/${XL_MODEL_REVISION}/`;
export const PIPELINE_BUILD =
  "2026-08-03-xl-turbo-fp32-webgpu-vae-v1";
export const ACESTEP_REPOSITORY = "ACE-Step/Ace-Step1.5";
export const ACESTEP_REVISION = "19671f406d603126926c1b7e2adc169acbcade22";
export const CACHE_NAME = `ai-music-js-${ONNX_REVISION.slice(0, 8)}`;
export const MODEL_NAME = "ACE-Step 1.5 XL Turbo";
export const MODEL_PARAMETER_COUNT = 4_168_897_088;
export const DIT_ATTENTION_HEADS = 32;
export const DIT_PATCH_SIZE = 2;
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

export type AudioQuality = "standard" | "high";
export const DEFAULT_AUDIO_QUALITY: AudioQuality = "standard";

export const isAudioQuality = (value: unknown): value is AudioQuality =>
  value === "standard" || value === "high";

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
  | "audio-code-detokenizer"
  | "dit"
  | "vae";

export type ModelGraph = {
  id: GraphId;
  label: string;
  quality?: AudioQuality;
  graph: DownloadAsset;
  weights: DownloadAsset[];
  webGpuOnlyBlockers: string[];
};

const onnxAsset = (
  group: string,
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
  quality?: AudioQuality,
  cacheGroup: string = id,
): ModelGraph => {
  const files =
    typeof weightFiles === "number"
      ? [{ fileName: `${graphFile}.data`, bytes: weightFiles }]
      : weightFiles;
  return {
    id,
    label,
    quality,
    graph: onnxAsset(
      cacheGroup,
      label,
      graphFile,
      graphBytes,
      "graph",
      local,
    ),
    weights: files.map((file, index) =>
      onnxAsset(
        cacheGroup,
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

export const ALL_MODEL_GRAPHS: ModelGraph[] = [
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
    "standard",
  ),
  graph(
    "condition-encoder",
    "ACE XL condition encoder · INT8 high precision",
    "condition_encoder_xl_turbo_q8.onnx",
    2_137_573,
    [
      {
        fileName: "condition_encoder_xl_turbo_q8.onnx.data.0",
        bytes: 656_270_832,
      },
    ],
    [],
    true,
    "high",
    "condition-encoder-int8",
  ),
  graph(
    "audio-code-detokenizer",
    "ACE XL semantic-code detokenizer · FP16",
    "audio_code_detokenizer_xl_fp16.onnx",
    127_091,
    210_820_224,
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
    "standard",
  ),
  graph(
    "dit",
    "ACE-Step XL Turbo 4B DiT · INT8 high precision",
    "dit_decoder_xl_turbo_q8.onnx",
    9_026_143,
    [
      {
        fileName: "dit_decoder_xl_turbo_q8.onnx.data.0",
        bytes: 948_725_760,
      },
      {
        fileName: "dit_decoder_xl_turbo_q8.onnx.data.1",
        bytes: 945_377_280,
      },
      {
        fileName: "dit_decoder_xl_turbo_q8.onnx.data.2",
        bytes: 932_659_200,
      },
      {
        fileName: "dit_decoder_xl_turbo_q8.onnx.data.3",
        bytes: 932_659_200,
      },
      {
        fileName: "dit_decoder_xl_turbo_q8.onnx.data.4",
        bytes: 932_659_200,
      },
      {
        fileName: "dit_decoder_xl_turbo_q8.onnx.data.5",
        bytes: 80_547_840,
      },
    ],
    [],
    true,
    "high",
    "dit-int8",
  ),
  graph(
    "vae",
    "Official Oobleck VAE decoder · FP32 WebGPU",
    "vae_decoder_fp32.onnx",
    1_076_526,
    337_707_008,
    [],
    true,
  ),
];

export const modelGraphsForAudioQuality = (
  quality: AudioQuality = DEFAULT_AUDIO_QUALITY,
) =>
  ALL_MODEL_GRAPHS.filter(
    (item) => item.quality === undefined || item.quality === quality,
  );

/** Backwards-compatible manifest for the default INT4 audio path. */
export const MODEL_GRAPHS = modelGraphsForAudioQuality();
export const HIGH_PRECISION_MODEL_GRAPHS =
  modelGraphsForAudioQuality("high");

export const assetsForAudioQuality = (
  quality: AudioQuality = DEFAULT_AUDIO_QUALITY,
) => [
  ...modelGraphsForAudioQuality(quality).flatMap(
    ({ graph: graphAsset, weights }) => [graphAsset, ...weights],
  ),
  ...SUPPORT_ASSETS,
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
  ...ALL_MODEL_GRAPHS.flatMap(({ graph: graphAsset, weights }) => [
    graphAsset,
    ...weights,
  ]),
  ...SUPPORT_ASSETS,
];

export const TOTAL_DOWNLOAD_BYTES = assetsForAudioQuality().reduce(
  (total, asset) => total + asset.bytes,
  0,
);

export const HIGH_QUALITY_TOTAL_DOWNLOAD_BYTES =
  assetsForAudioQuality("high").reduce(
    (total, asset) => total + asset.bytes,
    0,
  );

export const MODEL_DOWNLOAD_BYTES = modelGraphsForAudioQuality().reduce(
  (total, item) =>
    total +
    item.graph.bytes +
    item.weights.reduce((sum, asset) => sum + asset.bytes, 0),
  0,
);

export const HIGH_PRECISION_MODEL_DOWNLOAD_BYTES =
  modelGraphsForAudioQuality("high").reduce(
    (total, item) =>
      total +
      item.graph.bytes +
      item.weights.reduce((sum, asset) => sum + asset.bytes, 0),
    0,
  );

export const graphById = (
  id: GraphId,
  quality: AudioQuality = DEFAULT_AUDIO_QUALITY,
) => {
  const value = modelGraphsForAudioQuality(quality).find(
    (item) => item.id === id,
  );
  if (!value) {
    throw new Error(`Unknown graph: ${id}`);
  }
  return value;
};

export const buildCaptionPrompt = (
  caption: string,
  durationSeconds: number,
  semanticMetadata?: {
    bpm: number;
    keyScale: string;
    timeSignature: 2 | 3 | 4 | 6;
  },
) => `# Instruction
${
  semanticMetadata
    ? "Generate audio semantic tokens based on the given conditions:"
    : "Fill the audio semantic mask based on the given conditions:"
}

# Caption
${caption.trim()}

# Metas
- bpm: ${semanticMetadata?.bpm ?? "N/A"}
- timesignature: ${
  semanticMetadata ? `${semanticMetadata.timeSignature}/4` : "N/A"
}
- keyscale: ${semanticMetadata?.keyScale ?? "N/A"}
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

export const hasVocalPromptConflict = (
  caption: string,
  lyrics?: string,
) => {
  if (isInstrumentalLyrics(lyrics)) {
    return false;
  }
  const normalizedCaption = caption.trim().toLowerCase();
  const withoutNegativeVoicePhrases = normalizedCaption.replace(
    /\b(?:no|without)\s+(?:vocals?|singing|singers?|voices?)\b/g,
    "",
  );
  const requestsInstrumental =
    /\b(?:instrumental|no vocals?|without vocals?|no singing)\b/.test(
      normalizedCaption,
    );
  const requestsVoice =
    /\b(?:vocals?|vocalist|singers?|singing|sung|voices?|choir|rapping|rapper|spoken word)\b/.test(
      withoutNegativeVoicePhrases,
    );
  return requestsInstrumental && !requestsVoice;
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
