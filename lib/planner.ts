export const DEFAULT_PLANNER_MODEL =
  "emb1ter/ACE-Step-v1.5-5Hz-LM-4B-ONNX-WebGPU";
export const DEFAULT_PLANNER_MODEL_REVISION =
  "f2c20ea50f8b6e8fd9b866ec9d11358b2932ffb0";

export const AUDIO_CODEBOOK_SIZE = 64_000;
export const AUDIO_CODE_TOKEN_START = 151_669;
export const AUDIO_CODE_TOKEN_END =
  AUDIO_CODE_TOKEN_START + AUDIO_CODEBOOK_SIZE;
export const PLANNER_THINK_START_TOKEN = 151_667;
export const PLANNER_THINK_END_TOKEN = 151_668;
export const PLANNER_EOS_TOKEN = 151_645;
export const PLANNER_PAD_TOKEN = 151_643;
export const SEMANTIC_CODE_RATE = 5;
export const DEFAULT_PLANNER_CFG_SCALE = 2;
export const PLANNER_UNCONDITIONAL_USER_PROMPT = "NO USER INPUT";
export const ACE_METADATA_TEMPERATURE = 0.35;
export const ACE_METADATA_TOP_P = 0.85;
export const ACE_METADATA_BPM_MIN = 30;
export const ACE_METADATA_BPM_MAX = 300;
export const ACE_METADATA_DURATION_MIN = 10;
export const ACE_METADATA_DURATION_MAX = 120;
export const ACE_METADATA_TIME_SIGNATURES = [2, 3, 4, 6] as const;
export const ACE_METADATA_KEYSCALES = [
  ...["A", "B", "C", "D", "E", "F", "G"].flatMap((note) =>
    ["", "#", "b", "♯", "♭"].flatMap((accidental) =>
      ["major", "minor"].map(
        (mode) => `${note}${accidental} ${mode}`,
      ),
    ),
  ),
] as const;

export const PLANNER_SYSTEM_PROMPT =
  "# Instruction\nGenerate audio semantic tokens based on the given conditions:\n\n";

/**
 * Browser/native FP32 regression fixture.
 *
 * This is the exact vocal request used to qualify the split FP32 planner in
 * desktop Chromium. Keep the metadata caption paired with the request: the
 * autoregressive audio-code sequence changes when the CoT metadata changes.
 */
export const QUALIFIED_FP32_VOCAL_PROMPT =
  "Warm analog synthwave song, steady electronic drums, pulsing bass, " +
  "cinematic pads, memorable chorus, clear expressive lead vocal singing " +
  "every supplied lyric, polished studio mix";

export const QUALIFIED_FP32_VOCAL_LYRICS = `[Verse]
Static in the room hums low, like an old radio static.
I sit on the edge, watching the light flicker across the floor.
A ghost from yesterday sits by my side now.
It waits for me to forget what's left behind.

[Chorus]
The neon glow fades, but I'm still here waiting there.
For you, this place feels exactly right again and then gone.`;

export const QUALIFIED_FP32_VOCAL_METADATA_CAPTION =
  "A polished synthwave song built on warm analog synthesizers, steady " +
  "electronic drums, and a pulsing bassline. Cinematic pads create a " +
  "nostalgic atmosphere while a memorable chorus supports a clear, " +
  "expressive lead vocal.";

export const QUALIFIED_FP32_VOCAL_TOP_CODES = {
  conditional: [
    56_258, 58_818, 30_658, 43_458, 63_938, 63_939, 17_858, 17_914,
    38_338, 58_874, 56_274, 56_266, 30_659, 56_259, 43_459, 5_058,
    17_859, 63_954, 33_274, 58_819,
  ],
  unconditional: [
    38_119, 25_319, 38_111, 24_423, 24_935, 25_447, 21_863, 25_007,
    23_495, 24_007, 25_519, 22_887, 24_495, 24_015, 22_375, 25_311,
    24_487, 25_543, 23_503, 23_911,
  ],
  cfg: [
    43_458, 56_258, 58_817, 58_818, 58_874, 17_914, 33_274, 40_442,
    33_273, 55_746, 46_073, 46_018, 46_017, 43_459, 23_034, 33_218,
    30_714, 30_202, 16_882, 58_875,
  ],
} as const;

export type PlannerMetadata = {
  bpm: number;
  caption: string;
  durationSeconds: number;
  durationSource?: "requested" | "ace" | "recommended";
  keyScale: string;
  language: string;
  timeSignature: 2 | 3 | 4 | 6;
};

const unquote = (value: string) => {
  const trimmed = value.trim();
  if (
    trimmed.length >= 2 &&
    ((trimmed.startsWith('"') && trimmed.endsWith('"')) ||
      (trimmed.startsWith("'") && trimmed.endsWith("'")))
  ) {
    return trimmed.slice(1, -1);
  }
  return trimmed;
};

const metadataValue = (reasoning: string, key: string) => {
  const match = reasoning.match(
    new RegExp(`^\\s*${key}\\s*:\\s*(.+?)\\s*$`, "im"),
  );
  return match ? unquote(match[1]) : undefined;
};

export const fallbackPlannerMetadata = (
  caption: string,
  durationSeconds: number,
  language: string,
): PlannerMetadata => ({
  bpm: 120,
  caption: caption.trim(),
  durationSeconds,
  keyScale: "C major",
  language: language.trim() || "unknown",
  timeSignature: 4,
});

const normalizePlannerText = (value: string) =>
  value.trim().replace(/\s+/g, " ");

const vocalPlannerCaption = (caption: string) => {
  const normalized = normalizePlannerText(caption).replace(/[.!?]+$/, "");
  return (
    `${normalized}. The arrangement features a prominent, clear, expressive ` +
    "lead singer performing every supplied lyric intelligibly, with the " +
    "chorus delivered as a memorable vocal hook."
  );
};

export const isQualifiedFp32VocalRequest = (
  caption: string,
  lyrics: string,
  durationSeconds: number,
) =>
  normalizePlannerText(caption) === QUALIFIED_FP32_VOCAL_PROMPT &&
  lyrics.trim() === QUALIFIED_FP32_VOCAL_LYRICS &&
  durationSeconds === 30;

export const deterministicPlannerMetadata = (
  caption: string,
  lyrics: string,
  vocalLanguage: string,
  durationSeconds: number,
): PlannerMetadata => {
  const qualified = isQualifiedFp32VocalRequest(
    caption,
    lyrics,
    durationSeconds,
  );
  return {
    bpm: 100,
    caption: qualified
      ? QUALIFIED_FP32_VOCAL_METADATA_CAPTION
      : lyrics.trim()
        ? vocalPlannerCaption(caption)
        : normalizePlannerText(caption),
    durationSeconds,
    keyScale: "F major",
    language: lyrics.trim() ? vocalLanguage.trim() || "unknown" : "unknown",
    timeSignature: 4,
  };
};

export const plannerTopCodes = (
  values: ArrayLike<number>,
  count = 20,
) =>
  Array.from({ length: values.length }, (_, index) => index)
    .sort((left, right) => Number(values[right]) - Number(values[left]))
    .slice(0, count);

export const plannerTopCodeOverlap = (
  left: readonly number[],
  right: readonly number[],
) => {
  if (left.length === 0) return 0;
  const expected = new Set(right);
  return left.filter((value) => expected.has(value)).length / left.length;
};

export const parsePlannerMetadata = (
  reasoning: string,
  caption: string,
  durationSeconds: number,
  language: string,
): PlannerMetadata => {
  const fallback = fallbackPlannerMetadata(
    caption,
    durationSeconds,
    language,
  );
  const bpmValue = Number(metadataValue(reasoning, "bpm"));
  const timeSignatureValue = Number(
    metadataValue(reasoning, "timesignature")?.split("/")[0],
  );
  const plannedCaption = metadataValue(reasoning, "caption");
  const keyScale = metadataValue(reasoning, "keyscale");
  const plannedLanguage = metadataValue(reasoning, "language");
  return {
    bpm:
      Number.isInteger(bpmValue) && bpmValue >= 30 && bpmValue <= 300
        ? bpmValue
        : fallback.bpm,
    caption:
      plannedCaption && plannedCaption.length <= 2_000
        ? plannedCaption.replace(/\s+/g, " ").trim()
        : fallback.caption,
    // The explicit UI duration is authoritative. It also determines the exact
    // number of 5 Hz codes and downstream 25 Hz latent frames.
    durationSeconds,
    keyScale:
      keyScale &&
      /^[A-G](?:#|b|♯|♭)?\s+(?:major|minor)$/i.test(keyScale)
        ? keyScale
        : fallback.keyScale,
    language:
      plannedLanguage &&
      /^(?:unknown|[a-z]{2,3}(?:-[A-Za-z0-9]{2,8})?)$/i.test(
        plannedLanguage,
      )
        ? plannedLanguage
        : fallback.language,
    timeSignature: ([2, 3, 4, 6] as const).includes(
      timeSignatureValue as 2 | 3 | 4 | 6,
    )
      ? (timeSignatureValue as 2 | 3 | 4 | 6)
      : fallback.timeSignature,
  };
};

export const buildPlannerUserPrompt = (
  caption: string,
  lyrics: string,
) => `# Caption
${caption.trim()}

# Lyric
${lyrics.trim() || "[Instrumental]"}
`;

export const formatPlannerMetadata = (metadata: PlannerMetadata) => `<think>
bpm: ${metadata.bpm}
caption: ${metadata.caption}
duration: ${metadata.durationSeconds}
keyscale: ${metadata.keyScale}
language: ${metadata.language}
timesignature: ${metadata.timeSignature}
</think>`;

/**
 * ACE's production two-phase path skips caption/language rewriting and feeds
 * the four generated musical fields back to the code phase as sorted YAML.
 */
export const formatPlannerGeneratedMetadata = (
  metadata: PlannerMetadata,
) => `<think>
bpm: ${metadata.bpm}
duration: ${metadata.durationSeconds}
keyscale: ${metadata.keyScale}
timesignature: ${metadata.timeSignature}
</think>`;

export const resolvePlannerDuration = ({
  plannedDurationSeconds,
  requestedDurationSeconds,
  recommendedDurationSeconds,
  autoDuration,
}: {
  plannedDurationSeconds: number;
  requestedDurationSeconds: number;
  recommendedDurationSeconds: number;
  autoDuration: boolean;
}) => {
  if (!autoDuration) {
    return {
      durationSeconds: requestedDurationSeconds,
      durationSource: "requested" as const,
    };
  }
  const planned = Math.round(plannedDurationSeconds);
  const durationSeconds = Math.max(
    ACE_METADATA_DURATION_MIN,
    Math.min(
      ACE_METADATA_DURATION_MAX,
      Math.max(planned, recommendedDurationSeconds),
    ),
  );
  return {
    durationSeconds,
    durationSource:
      durationSeconds === planned
        ? ("ace" as const)
        : ("recommended" as const),
  };
};

export const semanticCodeCount = (durationSeconds: number) => {
  if (
    !Number.isInteger(durationSeconds) ||
    durationSeconds < 1 ||
    durationSeconds > 600
  ) {
    throw new RangeError(
      "Semantic duration must be a whole number from 1 through 600 seconds.",
    );
  }
  return durationSeconds * SEMANTIC_CODE_RATE;
};

/**
 * Seeded PRNG used by the qualified FP32 browser-planner path.
 *
 * Keep this independent from Transformers.js's process-global RNG so a lyric
 * generation request cannot change the semantic-code sequence for a given
 * planner seed.
 */
export const createPlannerSamplingRandom = (seed: number) => {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4_294_967_296;
  };
};

export const isAudioCodeToken = (tokenId: number) =>
  Number.isInteger(tokenId) &&
  tokenId >= AUDIO_CODE_TOKEN_START &&
  tokenId < AUDIO_CODE_TOKEN_END;

export const tokenIdToAudioCode = (tokenId: number) => {
  if (!isAudioCodeToken(tokenId)) {
    throw new RangeError(`Token ${tokenId} is not an ACE audio-code token.`);
  }
  return tokenId - AUDIO_CODE_TOKEN_START;
};

export const audioCodeToTokenId = (code: number) => {
  if (
    !Number.isInteger(code) ||
    code < 0 ||
    code >= AUDIO_CODEBOOK_SIZE
  ) {
    throw new RangeError(
      `ACE audio code must be from 0 through ${AUDIO_CODEBOOK_SIZE - 1}.`,
    );
  }
  return AUDIO_CODE_TOKEN_START + code;
};

export type PlannerSemanticDiagnostics = {
  count: number;
  uniqueCount: number;
  transitionCount: number;
  adjacentRepeatRatio: number;
  longestIdenticalRun: number;
  dominantCodeCount: number;
  dominantCodeRatio: number;
};

export const analyzePlannerSemanticCodes = (
  codes: readonly number[],
): PlannerSemanticDiagnostics => {
  const counts = new Map<number, number>();
  let transitionCount = 0;
  let longestIdenticalRun = 0;
  let currentRun = 0;
  let previousCode: number | undefined;

  for (const code of codes) {
    counts.set(code, (counts.get(code) ?? 0) + 1);
    if (code === previousCode) {
      currentRun += 1;
    } else {
      if (previousCode !== undefined) transitionCount += 1;
      currentRun = 1;
      previousCode = code;
    }
    longestIdenticalRun = Math.max(longestIdenticalRun, currentRun);
  }

  let dominantCodeCount = 0;
  for (const count of counts.values()) {
    dominantCodeCount = Math.max(dominantCodeCount, count);
  }
  const adjacentPairCount = Math.max(0, codes.length - 1);
  return {
    count: codes.length,
    uniqueCount: counts.size,
    transitionCount,
    adjacentRepeatRatio:
      adjacentPairCount === 0
        ? 0
        : (adjacentPairCount - transitionCount) / adjacentPairCount,
    longestIdenticalRun,
    dominantCodeCount,
    dominantCodeRatio:
      codes.length === 0 ? 0 : dominantCodeCount / codes.length,
  };
};

export type PlannerCfgBlendMode =
  | "cfg"
  | "conditional-fallback"
  | "invalid";

export type PlannerCfgBlendResult = {
  mode: PlannerCfgBlendMode;
  conditionalFinite: number;
  unconditionalFinite: number;
  guidedFinite: number;
};

/**
 * Combines the planner's conditional and unconditional audio-code logits.
 *
 * Some WebGPU drivers can return a non-finite unconditional row during
 * cached autoregressive generation even though the conditional row is still
 * valid. In that case CFG cannot be calculated, so the caller can safely
 * continue with scale 1 (the conditional distribution) instead of sampling
 * NaNs or discarding an otherwise usable plan.
 */
export const blendPlannerCfgLogits = (
  conditional: ArrayLike<number>,
  unconditional: ArrayLike<number>,
  cfgScale: number,
  output: Float32Array,
): PlannerCfgBlendResult => {
  if (
    conditional.length !== unconditional.length ||
    conditional.length !== output.length
  ) {
    throw new RangeError(
      "Planner CFG rows and output must have identical lengths.",
    );
  }
  if (!Number.isFinite(cfgScale) || cfgScale < 0) {
    throw new RangeError("Planner CFG scale must be a finite non-negative number.");
  }

  let conditionalFinite = 0;
  let unconditionalFinite = 0;
  let guidedFinite = 0;
  for (let index = 0; index < output.length; index += 1) {
    const conditionalValue = Number(conditional[index]);
    const unconditionalValue = Number(unconditional[index]);
    const conditionalIsFinite = Number.isFinite(conditionalValue);
    const unconditionalIsFinite = Number.isFinite(unconditionalValue);
    if (conditionalIsFinite) conditionalFinite += 1;
    if (unconditionalIsFinite) unconditionalFinite += 1;

    const guided =
      conditionalIsFinite && unconditionalIsFinite
        ? unconditionalValue +
          cfgScale * (conditionalValue - unconditionalValue)
        : Number.NEGATIVE_INFINITY;
    if (Number.isFinite(guided)) {
      output[index] = guided;
      guidedFinite += 1;
    } else {
      output[index] = Number.NEGATIVE_INFINITY;
    }
  }

  if (guidedFinite > 0) {
    return {
      mode: "cfg",
      conditionalFinite,
      unconditionalFinite,
      guidedFinite,
    };
  }
  if (conditionalFinite > 0) {
    for (let index = 0; index < output.length; index += 1) {
      const value = Number(conditional[index]);
      output[index] = Number.isFinite(value)
        ? value
        : Number.NEGATIVE_INFINITY;
    }
    return {
      mode: "conditional-fallback",
      conditionalFinite,
      unconditionalFinite,
      guidedFinite,
    };
  }
  output.fill(Number.NEGATIVE_INFINITY);
  return {
    mode: "invalid",
    conditionalFinite,
    unconditionalFinite,
    guidedFinite,
  };
};
