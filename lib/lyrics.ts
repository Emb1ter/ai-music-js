export const DEFAULT_LYRICS_MODEL =
  "onnx-community/Qwen3.5-0.8B-Text-ONNX";
export const DEFAULT_LYRICS_MODEL_REVISION =
  "1e45daba048899e7f771657ada617ec49350aa91";

const SECTION_TAG = /^\[[^\]\n]{1,30}\]$/;

export const BACKEND_VOCAL_MIN_WORDS = 40;
export const BACKEND_VOCAL_WORDS_PER_SECOND = 7 / 6;
export const DEFAULT_VOCAL_MIN_DURATION_SECONDS = 30;

export const defaultMaxLyricWords = (durationSeconds: number) =>
  Math.max(
    BACKEND_VOCAL_MIN_WORDS,
    Math.min(
      450,
      Math.round(
        durationSeconds * BACKEND_VOCAL_WORDS_PER_SECOND,
      ),
    ),
  );

export const countLyricWords = (lyrics: string) =>
  lyrics
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line && !SECTION_TAG.test(line))
    .reduce(
      (total, line) => total + line.split(/\s+/u).length,
      0,
    );

export type LyricDurationRecommendation = {
  wordCount: number;
  selectedDurationSeconds: number;
  selectedWordBudget: number;
  recommendedDurationSeconds: number;
  recommendedWordBudget: number;
  fits: boolean;
  exceedsByWords: number;
};

export type RecommendLyricDurationOptions = {
  minimumDurationSeconds?: number;
  maximumDurationSeconds?: number;
  roundToSeconds?: number;
};

const roundDurationUp = (duration: number, step: number) =>
  Math.ceil(duration / step) * step;

export const recommendDurationForLyrics = (
  lyrics: string,
  options: RecommendLyricDurationOptions = {},
) => {
  const minimumDurationSeconds =
    options.minimumDurationSeconds ??
    DEFAULT_VOCAL_MIN_DURATION_SECONDS;
  const maximumDurationSeconds =
    options.maximumDurationSeconds ?? 120;
  const roundToSeconds = options.roundToSeconds ?? 5;
  if (
    !Number.isInteger(minimumDurationSeconds) ||
    !Number.isInteger(maximumDurationSeconds) ||
    minimumDurationSeconds < 1 ||
    maximumDurationSeconds < minimumDurationSeconds ||
    !Number.isInteger(roundToSeconds) ||
    roundToSeconds < 1
  ) {
    throw new RangeError(
      "Lyric duration bounds and rounding must be positive whole seconds.",
    );
  }
  const wordCount = countLyricWords(lyrics);
  let duration = roundDurationUp(
    minimumDurationSeconds,
    roundToSeconds,
  );
  while (
    duration < maximumDurationSeconds &&
    defaultMaxLyricWords(duration) < wordCount
  ) {
    duration += roundToSeconds;
  }
  return Math.min(duration, maximumDurationSeconds);
};

export const assessLyricDuration = (
  lyrics: string,
  selectedDurationSeconds: number,
  options: RecommendLyricDurationOptions = {},
): LyricDurationRecommendation => {
  if (
    !Number.isInteger(selectedDurationSeconds) ||
    selectedDurationSeconds < 1
  ) {
    throw new RangeError(
      "Selected lyric duration must be a positive whole number of seconds.",
    );
  }
  const wordCount = countLyricWords(lyrics);
  const selectedWordBudget = defaultMaxLyricWords(
    selectedDurationSeconds,
  );
  const recommendedDurationSeconds = recommendDurationForLyrics(
    lyrics,
    {
      minimumDurationSeconds:
        options.minimumDurationSeconds ??
        DEFAULT_VOCAL_MIN_DURATION_SECONDS,
      maximumDurationSeconds:
        options.maximumDurationSeconds ??
        Math.max(120, selectedDurationSeconds),
      roundToSeconds: options.roundToSeconds,
    },
  );
  return {
    wordCount,
    selectedDurationSeconds,
    selectedWordBudget,
    recommendedDurationSeconds,
    recommendedWordBudget: defaultMaxLyricWords(
      recommendedDurationSeconds,
    ),
    fits: wordCount <= selectedWordBudget,
    exceedsByWords: Math.max(0, wordCount - selectedWordBudget),
  };
};

export const cleanLyrics = (value: string) => {
  let lyrics = value.trim();
  // Qwen thinking is disabled in the chat template, but also tolerate a
  // complete or token-limit-truncated reasoning block in model output.
  lyrics = lyrics
    .replace(/<think>[\s\S]*?(?:<\/think>|$)/gi, "")
    .trim();
  if (lyrics.startsWith("```") && lyrics.endsWith("```")) {
    lyrics = lyrics
      .replace(/^```(?:text|markdown)?\s*/i, "")
      .replace(/\s*```$/, "");
  }
  lyrics = lyrics.replace(
    /^#{1,6}\s*((?:Verse|Pre-Chorus|Chorus|Refrain|Bridge|Outro)(?:\s+\d+)?)\s*$/gim,
    "[$1]",
  );
  const cleaned: string[] = [];
  for (const rawLine of lyrics.split(/\r?\n/)) {
    const line = rawLine.trimEnd();
    if (SECTION_TAG.test(line.trim())) {
      while (cleaned.length && !cleaned.at(-1)?.trim()) {
        cleaned.pop();
      }
      if (cleaned.length && SECTION_TAG.test(cleaned.at(-1)!.trim())) {
        cleaned[cleaned.length - 1] = line.trim();
        continue;
      }
    }
    cleaned.push(line);
  }
  return cleaned.join("\n").trim();
};

export const compactLyrics = (lyrics: string, maxWords: number) => {
  let wordCount = 0;
  const compacted: string[] = [];
  for (const rawLine of lyrics.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line) {
      if (compacted.length && compacted.at(-1)) {
        compacted.push("");
      }
      continue;
    }
    if (SECTION_TAG.test(line)) {
      while (compacted.length && !compacted.at(-1)) {
        compacted.pop();
      }
      compacted.push(...(compacted.length ? ["", line] : [line]));
      continue;
    }
    const words = line.split(/\s+/).length;
    if (wordCount + words > maxWords) break;
    compacted.push(line);
    wordCount += words;
  }
  while (compacted.length && !compacted.at(-1)) compacted.pop();
  return compacted.join("\n").trim();
};

export const lyricQualityIssues = (lyrics: string, maxWords: number) => {
  const lyricLines = lyrics
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line && !SECTION_TAG.test(line));
  const issues: string[] = [];
  if (lyricLines.length < 4) {
    issues.push("write at least four lyric lines");
  }
  if (
    lyricLines.reduce(
      (total, line) => total + line.split(/\s+/).length,
      0,
    ) > maxWords
  ) {
    issues.push(`keep the song under ${maxWords} words`);
  }
  let previousLine = "";
  let consecutiveRepetitions = 0;
  for (const line of lyricLines) {
    const normalized = line
      .replace(/[^\p{L}\p{N}]+/gu, " ")
      .toLocaleLowerCase()
      .trim();
    if (normalized && normalized === previousLine) {
      consecutiveRepetitions += 1;
    } else {
      previousLine = normalized;
      consecutiveRepetitions = normalized ? 1 : 0;
    }
    if (consecutiveRepetitions >= 4) {
      issues.push("remove the repeated verse loop");
      break;
    }
  }
  return issues;
};

export const buildTimedLyricsPrompt = (
  prompt: string,
  durationSeconds: number,
  maxWords: number,
) => `${prompt.trim()}

Enforced timing: the finished song is ${durationSeconds} seconds long.
The complete lyrics must stay under ${maxWords} words.`;

export const buildLyricsRepairPrompt = (
  sourcePrompt: string,
  draft: string,
  qualityIssues: readonly string[],
  durationSeconds: number,
  maxWords: number,
) => `Rewrite the draft below as a complete, coherent song.
Fix these problems:
- ${qualityIssues.join("\n- ")}
Use [Verse], [Chorus], [Bridge], and [Outro] tags. Never repeat a verse.
The finished song is ${durationSeconds} seconds long. Keep the complete result
under ${maxWords} words so it fits that duration.
The original brief below is authoritative. Keep its subject, mood, and facts.
Do not preserve invented story details from the bad draft.

Original brief:
${sourcePrompt.trim()}

Return only the revised lyrics.

Bad draft to repair:
${draft}`;
