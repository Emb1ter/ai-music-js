export const PROMPT_VARIANTS = [
  { id: "a", label: "A · production baseline" },
  { id: "b", label: "B · songwriter" },
  { id: "c", label: "C · strict budget" },
  { id: "d", label: "D · narrative fidelity" },
  { id: "e", label: "E · two-pass facts" },
  { id: "f", label: "F · minimal fidelity" },
  { id: "g", label: "G · story editor" },
  { id: "h", label: "H · balanced songcraft" },
  { id: "i", label: "I · compact fidelity" },
  { id: "j", label: "J · RhymeAI fact-dense" },
  { id: "k", label: "K · RhymeAI minimal" },
  { id: "l", label: "L · natural over rhyme" },
  { id: "m", label: "M · exact words" },
  { id: "o", label: "O · K loose long-form" },
  { id: "p", label: "P · L loose long-form" },
  { id: "q", label: "Q · literal long-form" },
] as const;

export type PromptVariant = (typeof PROMPT_VARIANTS)[number]["id"];

export type BatchStory = {
  id: string;
  label: string;
  focus: string;
  brief: string;
};

export type BatchJobDescriptor = {
  modelId: string;
  variant: PromptVariant;
  story: BatchStory;
};

export const buildBatchMatrix = ({
  modelIds,
  variants,
  stories,
}: {
  modelIds: readonly string[];
  variants: readonly PromptVariant[];
  stories: readonly BatchStory[];
}) =>
  modelIds.flatMap((modelId) =>
    stories.flatMap((story) =>
      variants.map((variant) => ({ modelId, variant, story })),
    ),
  );

export type BatchResultRow = {
  runIndex: number;
  totalRuns: number;
  status: "success" | "error" | "skipped";
  startedAt: string;
  finishedAt: string;
  modelId: string;
  modelLabel: string;
  modelRepository: string;
  modelRevision: string;
  modelDtype: string;
  modelFamily: string;
  variant: PromptVariant;
  storyId: string;
  storyLabel: string;
  storyFocus: string;
  sourceBrief: string;
  language: string;
  durationSeconds: number;
  maxWords: number;
  seed: number;
  temperature: number;
  topK: number;
  topP: number;
  repetitionPenalty: number;
  maxNewTokens: number;
  systemPrompt: string;
  effectiveUserPrompt: string;
  factSystemPrompt: string;
  factSheet: string;
  factRaw: string;
  rawOutput: string;
  cleanedOutput: string;
  processedLyrics: string;
  issues: string;
  wordCount: number | "";
  generatedWordCount: number | "";
  inputTokens: number | "";
  outputTokens: number | "";
  modelLoadMilliseconds: number | "";
  generationMilliseconds: number | "";
  factInputTokens: number | "";
  factOutputTokens: number | "";
  factExtractionMilliseconds: number | "";
  error: string;
  stack: string;
};

export const BATCH_CSV_COLUMNS = [
  "runIndex",
  "totalRuns",
  "status",
  "startedAt",
  "finishedAt",
  "modelId",
  "modelLabel",
  "modelRepository",
  "modelRevision",
  "modelDtype",
  "modelFamily",
  "variant",
  "storyId",
  "storyLabel",
  "storyFocus",
  "sourceBrief",
  "language",
  "durationSeconds",
  "maxWords",
  "seed",
  "temperature",
  "topK",
  "topP",
  "repetitionPenalty",
  "maxNewTokens",
  "systemPrompt",
  "effectiveUserPrompt",
  "factSystemPrompt",
  "factSheet",
  "factRaw",
  "rawOutput",
  "cleanedOutput",
  "processedLyrics",
  "issues",
  "wordCount",
  "generatedWordCount",
  "inputTokens",
  "outputTokens",
  "modelLoadMilliseconds",
  "generationMilliseconds",
  "factInputTokens",
  "factOutputTokens",
  "factExtractionMilliseconds",
  "error",
  "stack",
] as const satisfies readonly (keyof BatchResultRow)[];

const csvCell = (value: unknown) => {
  let text = value === undefined || value === null ? "" : String(value);
  // Prevent generated text from becoming a formula when opened in a spreadsheet.
  if (/^[=+@\t\r]/.test(text) || /^-\D/.test(text)) text = `'${text}`;
  return `"${text.replaceAll('"', '""')}"`;
};

export const serializeBatchResultsCsv = (
  rows: readonly BatchResultRow[],
) => {
  const header = BATCH_CSV_COLUMNS.map(csvCell).join(",");
  const body = rows.map((row) =>
    BATCH_CSV_COLUMNS.map((column) => csvCell(row[column])).join(","),
  );
  return `\uFEFF${[header, ...body].join("\r\n")}\r\n`;
};
