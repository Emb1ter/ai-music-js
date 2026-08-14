import {
  buildTimedLyricsPrompt,
  defaultMaxLyricWords,
} from "../../../lib/lyrics";
import {
  CANDIDATE_LYRICS_SYSTEM_PROMPT,
  COMPACT_FIDELITY_LYRICS_SYSTEM_PROMPT,
  FACT_EXTRACTION_SYSTEM_PROMPT,
  MINIMAL_FIDELITY_LYRICS_SYSTEM_PROMPT,
  NARRATIVE_FIDELITY_LYRICS_SYSTEM_PROMPT,
  PRODUCTION_BASELINE_LYRICS_SYSTEM_PROMPT,
  RHYMEAI_FACT_DENSE_LYRICS_SYSTEM_PROMPT,
  RHYMEAI_MINIMAL_LYRICS_SYSTEM_PROMPT,
  RHYMEAI_NATURAL_LANGUAGE_SYSTEM_PROMPT,
  RHYMEAI_EXACT_WORDS_SYSTEM_PROMPT,
  RHYMEAI_LITERAL_SENSE_SYSTEM_PROMPT,
  SONGCRAFT_BALANCED_LYRICS_SYSTEM_PROMPT,
  STORY_EDITOR_LYRICS_SYSTEM_PROMPT,
  STRICT_BUDGET_LYRICS_SYSTEM_PROMPT,
  buildCandidateLyricsPrompt,
  buildFactExtractionPrompt,
  buildFactGroundedLyricsPrompt,
  buildRhymeAiLooseLongFormLyricsPrompt,
  buildRhymeAiLiteralLongFormLyricsPrompt,
  buildRhymeAiMinimalLyricsPrompt,
  buildStrictBudgetLyricsPrompt,
} from "./prompts";
import {
  DEFAULT_LYRIC_LAB_MODEL_ID,
  LYRIC_LAB_MODELS,
  lyricLabModelById,
} from "./models";
import {
  DEFAULT_TEST_BRIEF_ID,
  LYRIC_TEST_BRIEFS,
  lyricTestBriefById,
} from "./test-briefs";
import {
  PROMPT_VARIANTS,
  buildBatchMatrix,
  serializeBatchResultsCsv,
  type BatchJobDescriptor,
  type BatchResultRow,
  type PromptVariant,
} from "./batch-evaluation";
import "./styles.css";

type Variant = PromptVariant;

type ResultUpdate = {
  type: "result";
  variant: Variant;
  raw: string;
  cleaned: string;
  processed: string;
  issues: string[];
  inputTokens: number;
  outputTokens: number;
  wordCount: number;
  generatedWordCount: number;
  generationMilliseconds: number;
  modelLoadMilliseconds: number;
  factSheet?: string;
  factRaw?: string;
  factInputTokens?: number;
  factOutputTokens?: number;
  factExtractionMilliseconds?: number;
  effectiveUserPrompt: string;
};

type GenerationRequest = {
  type: "generate";
  variant: Variant;
  mode: "single-pass" | "two-pass";
  systemPrompt: string;
  userPrompt: string;
  sourceBrief: string;
  factSystemPrompt?: string;
  factUserPrompt?: string;
  durationSeconds: number;
  language: string;
  seed: number;
  maxWords: number;
  temperature: number;
  topK: number;
  topP: number;
  repetitionPenalty: number;
  maxNewTokens: number;
  modelId: string;
  modelRevision: string;
  modelDtype: "q4" | "q4f16";
  modelFamily: "qwen35" | "gemma4";
  modelLabel: string;
  modelLocalOnly: boolean;
  wasmUrl: string;
  wasmModuleUrl: string;
};

type BatchJob = BatchJobDescriptor & {
  runIndex: number;
  request: GenerationRequest;
  startedAt?: string;
};

type LabUpdate =
  | ResultUpdate
  | {
      type: "stage";
      variant: Variant;
      stage: string;
      detail: string;
    }
  | {
      type: "download";
      variant: Variant;
      file: string;
      loaded: number;
      total: number;
    }
  | {
      type: "error";
      variant: Variant;
      message: string;
      stack?: string;
    }
  | {
      type: "disposed";
      requestId: string;
    };

const required = <T extends Element>(selector: string) => {
  const element = document.querySelector<T>(selector);
  if (!element) throw new Error(`Missing lyric-lab element: ${selector}`);
  return element;
};

const brief = required<HTMLTextAreaElement>("#brief");
const briefPreset = required<HTMLSelectElement>("#brief-preset");
const briefFocus = required<HTMLElement>("#brief-focus");
const modelSelect = required<HTMLSelectElement>("#model");
const language = required<HTMLSelectElement>("#language");
const duration = required<HTMLSelectElement>("#duration");
const maxWords = required<HTMLInputElement>("#max-words");
const budgetHint = required<HTMLElement>("#budget-hint");
const seed = required<HTMLInputElement>("#seed");
const temperature = required<HTMLInputElement>("#temperature");
const topK = required<HTMLInputElement>("#top-k");
const topP = required<HTMLInputElement>("#top-p");
const repetitionPenalty = required<HTMLInputElement>("#repetition-penalty");
const maxNewTokens = required<HTMLInputElement>("#max-new-tokens");
const systemA = required<HTMLTextAreaElement>("#system-a");
const systemB = required<HTMLTextAreaElement>("#system-b");
const systemC = required<HTMLTextAreaElement>("#system-c");
const systemD = required<HTMLTextAreaElement>("#system-d");
const systemE = required<HTMLTextAreaElement>("#system-e");
const systemF = required<HTMLTextAreaElement>("#system-f");
const systemG = required<HTMLTextAreaElement>("#system-g");
const systemH = required<HTMLTextAreaElement>("#system-h");
const systemI = required<HTMLTextAreaElement>("#system-i");
const systemJ = required<HTMLTextAreaElement>("#system-j");
const systemK = required<HTMLTextAreaElement>("#system-k");
const systemL = required<HTMLTextAreaElement>("#system-l");
const systemM = required<HTMLTextAreaElement>("#system-m");
const systemO = required<HTMLTextAreaElement>("#system-o");
const systemP = required<HTMLTextAreaElement>("#system-p");
const systemQ = required<HTMLTextAreaElement>("#system-q");
const factsSystemE = required<HTMLTextAreaElement>("#facts-system-e");
const userA = required<HTMLTextAreaElement>("#user-a");
const userB = required<HTMLTextAreaElement>("#user-b");
const userC = required<HTMLTextAreaElement>("#user-c");
const userD = required<HTMLTextAreaElement>("#user-d");
const userE = required<HTMLTextAreaElement>("#user-e");
const userF = required<HTMLTextAreaElement>("#user-f");
const userG = required<HTMLTextAreaElement>("#user-g");
const userH = required<HTMLTextAreaElement>("#user-h");
const userI = required<HTMLTextAreaElement>("#user-i");
const userJ = required<HTMLTextAreaElement>("#user-j");
const userK = required<HTMLTextAreaElement>("#user-k");
const userL = required<HTMLTextAreaElement>("#user-l");
const userM = required<HTMLTextAreaElement>("#user-m");
const userO = required<HTMLTextAreaElement>("#user-o");
const userP = required<HTMLTextAreaElement>("#user-p");
const userQ = required<HTMLTextAreaElement>("#user-q");
const runRecommended = required<HTMLButtonElement>("#run-recommended");
const runRhymeAi = required<HTMLButtonElement>("#run-rhymeai");
const runAll = required<HTMLButtonElement>("#run-all");
const cancel = required<HTMLButtonElement>("#cancel");
const progress = required<HTMLProgressElement>("#progress");
const progressLabel = required<HTMLElement>("#progress-label");
const modelStatus = required<HTMLElement>("#model-status");
const batchModels = required<HTMLElement>("#batch-models");
const batchVariants = required<HTMLElement>("#batch-variants");
const batchStories = required<HTMLElement>("#batch-stories");
const batchCount = required<HTMLElement>("#batch-count");
const runBatch = required<HTMLButtonElement>("#run-batch");
const downloadBatchCsv = required<HTMLButtonElement>("#download-batch-csv");
const downloadBatchJson = required<HTMLButtonElement>("#download-batch-json");
const clearBatch = required<HTMLButtonElement>("#clear-batch");
const batchProgress = required<HTMLProgressElement>("#batch-progress");
const batchStatus = required<HTMLElement>("#batch-status");
const batchResultsBody = required<HTMLTableSectionElement>("#batch-results");

for (const model of LYRIC_LAB_MODELS) {
  const option = document.createElement("option");
  option.value = model.id;
  option.textContent = model.label;
  option.selected = model.id === DEFAULT_LYRIC_LAB_MODEL_ID;
  modelSelect.append(option);
}
modelStatus.textContent = `${lyricLabModelById(modelSelect.value).shortLabel} selected`;

const appendBatchChoice = ({
  container,
  group,
  value,
  label,
  detail,
  checked,
}: {
  container: Element;
  group: "model" | "variant" | "story";
  value: string;
  label: string;
  detail?: string;
  checked: boolean;
}) => {
  const wrapper = document.createElement("label");
  const input = document.createElement("input");
  input.type = "checkbox";
  input.className = "batch-choice";
  input.dataset.batchGroup = group;
  input.value = value;
  input.checked = checked;
  const text = document.createElement("span");
  text.textContent = label;
  wrapper.append(input, text);
  if (detail) {
    const note = document.createElement("small");
    note.textContent = detail;
    wrapper.append(note);
  }
  container.append(wrapper);
};

for (const model of LYRIC_LAB_MODELS) {
  appendBatchChoice({
    container: batchModels,
    group: "model",
    value: model.id,
    label: model.shortLabel,
    detail: model.localOnly ? "Requires its local export" : undefined,
    checked: model.id === DEFAULT_LYRIC_LAB_MODEL_ID,
  });
}
for (const variant of PROMPT_VARIANTS) {
  appendBatchChoice({
    container: batchVariants,
    group: "variant",
    value: variant.id,
    label: variant.label,
    checked: variant.id !== "h",
  });
}
for (const story of LYRIC_TEST_BRIEFS) {
  appendBatchChoice({
    container: batchStories,
    group: "story",
    value: story.id,
    label: story.label,
    detail: story.focus,
    checked: true,
  });
}

for (const item of LYRIC_TEST_BRIEFS) {
  const option = document.createElement("option");
  option.value = item.id;
  option.textContent = item.label;
  option.selected = item.id === DEFAULT_TEST_BRIEF_ID;
  briefPreset.append(option);
}
const customBriefOption = document.createElement("option");
customBriefOption.value = "custom";
customBriefOption.textContent = "Custom brief";
briefPreset.append(customBriefOption);

const initialBrief = lyricTestBriefById(DEFAULT_TEST_BRIEF_ID);
brief.value = initialBrief.brief;
briefFocus.textContent = `Tests: ${initialBrief.focus}`;
systemA.value = PRODUCTION_BASELINE_LYRICS_SYSTEM_PROMPT;
systemB.value = CANDIDATE_LYRICS_SYSTEM_PROMPT;
systemC.value = STRICT_BUDGET_LYRICS_SYSTEM_PROMPT;
systemD.value = NARRATIVE_FIDELITY_LYRICS_SYSTEM_PROMPT;
systemE.value = NARRATIVE_FIDELITY_LYRICS_SYSTEM_PROMPT;
systemF.value = MINIMAL_FIDELITY_LYRICS_SYSTEM_PROMPT;
systemG.value = STORY_EDITOR_LYRICS_SYSTEM_PROMPT;
systemH.value = SONGCRAFT_BALANCED_LYRICS_SYSTEM_PROMPT;
systemI.value = COMPACT_FIDELITY_LYRICS_SYSTEM_PROMPT;
systemJ.value = RHYMEAI_FACT_DENSE_LYRICS_SYSTEM_PROMPT;
systemK.value = RHYMEAI_MINIMAL_LYRICS_SYSTEM_PROMPT;
systemL.value = RHYMEAI_NATURAL_LANGUAGE_SYSTEM_PROMPT;
systemM.value = RHYMEAI_EXACT_WORDS_SYSTEM_PROMPT;
systemO.value = RHYMEAI_MINIMAL_LYRICS_SYSTEM_PROMPT;
systemP.value = RHYMEAI_NATURAL_LANGUAGE_SYSTEM_PROMPT;
systemQ.value = RHYMEAI_LITERAL_SENSE_SYSTEM_PROMPT;
factsSystemE.value = FACT_EXTRACTION_SYSTEM_PROMPT;

const numberValue = (input: HTMLInputElement) => Number(input.value);
const selectedDuration = () => Number(duration.value);

const refreshPrompts = () => {
  const seconds = selectedDuration();
  const budget = numberValue(maxWords);
  userA.value = buildTimedLyricsPrompt(
    `${brief.value.trim()}\nWrite the lyrics in ${language.value}.`,
    seconds,
    budget,
  );
  userB.value = buildCandidateLyricsPrompt({
    brief: brief.value,
    durationSeconds: seconds,
    maxWords: budget,
    language: language.value,
  });
  userC.value = buildStrictBudgetLyricsPrompt({
    brief: brief.value,
    durationSeconds: seconds,
    maxWords: budget,
    language: language.value,
  });
  userD.value = buildStrictBudgetLyricsPrompt({
    brief: brief.value,
    durationSeconds: seconds,
    maxWords: budget,
    language: language.value,
  });
  userE.value = buildFactGroundedLyricsPrompt({
    factSheet: "<PASS 1 FACT SHEET WILL BE INSERTED HERE>",
    durationSeconds: seconds,
    maxWords: budget,
    language: language.value,
  });
  const controlledPrompt = buildStrictBudgetLyricsPrompt({
    brief: brief.value,
    durationSeconds: seconds,
    maxWords: budget,
    language: language.value,
  });
  userF.value = controlledPrompt;
  userG.value = controlledPrompt;
  userH.value = controlledPrompt;
  userI.value = controlledPrompt;
  userJ.value = controlledPrompt;
  userK.value = buildRhymeAiMinimalLyricsPrompt({
    brief: brief.value,
    durationSeconds: seconds,
    maxWords: budget,
    language: language.value,
  });
  userL.value = userK.value;
  userM.value = userK.value;
  userO.value = buildRhymeAiLooseLongFormLyricsPrompt({
    brief: brief.value,
    durationSeconds: seconds,
    maxWords: budget,
    language: language.value,
  });
  userP.value = userO.value;
  userQ.value = buildRhymeAiLiteralLongFormLyricsPrompt({
    brief: brief.value,
    durationSeconds: seconds,
    maxWords: budget,
    language: language.value,
  });
};

const refreshBudget = () => {
  const seconds = selectedDuration();
  const budget = defaultMaxLyricWords(seconds);
  maxWords.value = String(budget);
  budgetHint.textContent = `Backend-matched budget for ${seconds} seconds.`;
  refreshPrompts();
};

briefPreset.addEventListener("change", () => {
  if (briefPreset.value === "custom") {
    briefFocus.textContent = "Custom input; no preset test focus.";
    return;
  }
  const selected = lyricTestBriefById(briefPreset.value);
  brief.value = selected.brief;
  briefFocus.textContent = `Tests: ${selected.focus}`;
  refreshPrompts();
});
brief.addEventListener("input", () => {
  const selected = LYRIC_TEST_BRIEFS.find(
    (item) => item.brief === brief.value,
  );
  briefPreset.value = selected?.id ?? "custom";
  briefFocus.textContent = selected
    ? `Tests: ${selected.focus}`
    : "Custom input; no preset test focus.";
  refreshPrompts();
});
language.addEventListener("input", refreshPrompts);
duration.addEventListener("change", refreshBudget);
maxWords.addEventListener("input", refreshPrompts);
refreshBudget();

const checkedBatchValues = (group: "model" | "variant" | "story") =>
  [...document.querySelectorAll<HTMLInputElement>(
    `.batch-choice[data-batch-group="${group}"]:checked`,
  )].map((input) => input.value);

const updateBatchCount = () => {
  const models = checkedBatchValues("model").length;
  const variants = checkedBatchValues("variant").length;
  const stories = checkedBatchValues("story").length;
  const runs = models * variants * stories;
  batchCount.textContent = `${runs} run${runs === 1 ? "" : "s"} selected`;
};
for (const input of document.querySelectorAll<HTMLInputElement>(".batch-choice")) {
  input.addEventListener("change", updateBatchCount);
}
updateBatchCount();

let worker: Worker | undefined;
let busy = false;
let queue: Variant[] = [];
let activeVariant: Variant | undefined;
let runMode: "interactive" | "batch" | undefined;
let batchQueue: BatchJob[] = [];
let activeBatchJob: BatchJob | undefined;
let activeWorkerModelId: string | undefined;
let activeWorkerStage = "";
let batchRows: BatchResultRow[] = [];
let batchCompletedRuns = 0;
let batchTotalRuns = 0;
let batchCancelled = false;
const downloads = new Map<string, { loaded: number; total: number }>();
const WORKER_DISPOSE_TIMEOUT_MS = 15_000;

const wasmUrl = new URL(
  "../../../dist/wasm/ort-wasm-simd-threaded.asyncify.wasm",
  import.meta.url,
).href;
const wasmModuleUrl = new URL(
  "../../../dist/wasm/ort-wasm-simd-threaded.asyncify.mjs",
  import.meta.url,
).href;

const formatDuration = (milliseconds: number) => {
  const seconds = milliseconds / 1000;
  return seconds < 60
    ? `${seconds.toFixed(2)}s`
    : `${Math.floor(seconds / 60)}m ${(seconds % 60).toFixed(1)}s`;
};

const setControls = (isBusy: boolean) => {
  busy = isBusy;
  runRecommended.disabled = isBusy;
  runAll.disabled = isBusy;
  runBatch.disabled = isBusy;
  cancel.disabled = !isBusy;
  modelSelect.disabled = isBusy;
  clearBatch.disabled = isBusy || batchRows.length === 0;
  for (const input of document.querySelectorAll<HTMLInputElement>(".batch-choice")) {
    input.disabled = isBusy;
  }
  for (const button of document.querySelectorAll<HTMLButtonElement>(".run-one")) {
    button.disabled = isBusy;
  }
};

const statusFor = (variant: Variant) =>
  required<HTMLElement>(`#status-${variant}`);
const processedFor = (variant: Variant) =>
  required<HTMLElement>(`#processed-${variant}`);
const rawFor = (variant: Variant) =>
  required<HTMLElement>(`#raw-${variant}`);
const metricsFor = (variant: Variant) =>
  required<HTMLElement>(`#metrics-${variant}`);

const ensureWorker = () => {
  if (worker) return worker;
  worker = new Worker(new URL("./lyrics-lab.worker.ts", import.meta.url), {
    type: "module",
    name: "ai-music-js-lyrics-lab",
  });
  worker.onmessage = (event: MessageEvent<LabUpdate>) => {
    const update = event.data;
    if (update.type === "disposed") return;
    if (update.type === "download") {
      downloads.set(update.file, {
        loaded: update.loaded,
        total: update.total,
      });
      const totals = [...downloads.values()].reduce(
        (sum, item) => ({
          loaded: sum.loaded + item.loaded,
          total: sum.total + item.total,
        }),
        { loaded: 0, total: 0 },
      );
      const fraction = totals.total ? totals.loaded / totals.total : 0;
      progress.value = Math.min(0.8, fraction * 0.8);
      progressLabel.textContent = `Downloading ${update.file} · ${Math.round(fraction * 100)}%`;
      const activeModel = activeBatchJob?.modelId ?? modelSelect.value;
      modelStatus.textContent = `Loading ${lyricLabModelById(activeModel).shortLabel}`;
      if (runMode === "batch") {
        batchProgress.value = Math.min(
          1,
          (batchCompletedRuns + fraction * 0.8) / Math.max(1, batchTotalRuns),
        );
      }
      return;
    }
    if (update.type === "stage") {
      activeWorkerStage = update.stage;
      statusFor(update.variant).textContent = update.detail;
      progress.value = update.stage === "generating" ? 0.85 : progress.value;
      progressLabel.textContent = update.detail;
      const activeModel = activeBatchJob?.modelId ?? modelSelect.value;
      modelStatus.textContent =
        update.stage === "generating"
          ? `${lyricLabModelById(activeModel).shortLabel} ready`
          : `Loading ${lyricLabModelById(activeModel).shortLabel}`;
      if (runMode === "batch") batchStatus.textContent = update.detail;
      return;
    }
    if (update.type === "error") {
      if (runMode === "batch" && activeBatchJob) {
        const failedDuringLoad = activeWorkerStage === "model-load";
        finishBatchError(update.message, update.stack ?? "", failedDuringLoad);
        return;
      }
      statusFor(update.variant).textContent = `Failed: ${update.message}`;
      progressLabel.textContent = update.message;
      progress.classList.add("failed");
      queue = [];
      activeVariant = undefined;
      setControls(false);
      return;
    }
    renderResult(update);
    if (runMode === "batch" && activeBatchJob) {
      finishBatchSuccess(update);
      return;
    }
    modelStatus.textContent = `${lyricLabModelById(modelSelect.value).shortLabel} loaded · WebGPU`;
    activeVariant = undefined;
    void runNext();
  };
  worker.onerror = (event) => {
    if (runMode === "batch" && activeBatchJob) {
      worker?.terminate();
      worker = undefined;
      activeWorkerModelId = undefined;
      finishBatchError(
        `Worker crashed: ${event.message}`,
        "",
        true,
      );
      return;
    }
    const variant = activeVariant ?? "a";
    statusFor(variant).textContent = `Worker crashed: ${event.message}`;
    progressLabel.textContent = "The lyric Worker crashed.";
    queue = [];
    activeVariant = undefined;
    setControls(false);
    worker?.terminate();
    worker = undefined;
  };
  return worker;
};

const releaseWorker = () => {
  const currentWorker = worker;
  worker = undefined;
  if (!currentWorker) return Promise.resolve();

  const requestId = crypto.randomUUID();
  return new Promise<void>((resolve) => {
    let settled = false;
    const finish = () => {
      if (settled) return;
      settled = true;
      window.clearTimeout(timeout);
      currentWorker.removeEventListener("message", onMessage);
      currentWorker.removeEventListener("error", finish);
      currentWorker.terminate();
      resolve();
    };
    const onMessage = (event: MessageEvent<LabUpdate>) => {
      if (
        event.data.type === "disposed" &&
        event.data.requestId === requestId
      ) {
        finish();
      }
    };
    const timeout = window.setTimeout(finish, WORKER_DISPOSE_TIMEOUT_MS);
    currentWorker.addEventListener("message", onMessage);
    currentWorker.addEventListener("error", finish, { once: true });
    currentWorker.postMessage({ type: "dispose", requestId });
  });
};

const renderResult = (update: ResultUpdate) => {
  const budget = numberValue(maxWords);
  processedFor(update.variant).textContent = update.processed || "(empty)";
  rawFor(update.variant).textContent = update.raw || "(empty)";
  if (update.variant === "e") {
    required<HTMLElement>("#facts-e").textContent =
      update.factSheet || "(empty fact sheet)";
    userE.value = update.effectiveUserPrompt;
  }
  statusFor(update.variant).textContent = `Complete in ${formatDuration(update.generationMilliseconds)}`;
  const metrics = metricsFor(update.variant);
  metrics.replaceChildren();
  const values = [
    ["Generated words", String(update.generatedWordCount)],
    ["Kept words", `${update.wordCount}/${budget}`],
    ["Input tokens", String(update.inputTokens)],
    ["Output tokens", String(update.outputTokens)],
    ["Generation", formatDuration(update.generationMilliseconds)],
    ...(update.factExtractionMilliseconds === undefined
      ? []
      : [["Fact extraction", formatDuration(update.factExtractionMilliseconds)]]),
    ["Model load", formatDuration(update.modelLoadMilliseconds)],
    ["Basic validator", update.issues.length ? update.issues.join("; ") : "passed"],
  ];
  for (const [label, value] of values) {
    const item = document.createElement("div");
    const name = document.createElement("span");
    const result = document.createElement("strong");
    name.textContent = label;
    result.textContent = value;
    item.append(name, result);
    metrics.append(item);
  }
};

type EvaluationSnapshot = {
  language: string;
  durationSeconds: number;
  maxWords: number;
  seed: number;
  temperature: number;
  topK: number;
  topP: number;
  repetitionPenalty: number;
  maxNewTokens: number;
  systemPrompts: Record<Variant, string>;
  factSystemPrompt: string;
};

const captureEvaluationSnapshot = (): EvaluationSnapshot => ({
  language: language.value,
  durationSeconds: selectedDuration(),
  maxWords: numberValue(maxWords),
  seed: numberValue(seed),
  temperature: numberValue(temperature),
  topK: numberValue(topK),
  topP: numberValue(topP),
  repetitionPenalty: numberValue(repetitionPenalty),
  maxNewTokens: numberValue(maxNewTokens),
  systemPrompts: {
    a: systemA.value,
    b: systemB.value,
    c: systemC.value,
    d: systemD.value,
    e: systemE.value,
    f: systemF.value,
    g: systemG.value,
    h: systemH.value,
    i: systemI.value,
    j: systemJ.value,
    k: systemK.value,
    l: systemL.value,
    m: systemM.value,
    o: systemO.value,
    p: systemP.value,
    q: systemQ.value,
  },
  factSystemPrompt: factsSystemE.value,
});

const generatedUserPrompt = (
  variant: Variant,
  sourceBrief: string,
  snapshot: EvaluationSnapshot,
) => {
  if (variant === "a") {
    return buildTimedLyricsPrompt(
      `${sourceBrief.trim()}\nWrite the lyrics in ${snapshot.language}.`,
      snapshot.durationSeconds,
      snapshot.maxWords,
    );
  }
  if (variant === "b") {
    return buildCandidateLyricsPrompt({
      brief: sourceBrief,
      durationSeconds: snapshot.durationSeconds,
      maxWords: snapshot.maxWords,
      language: snapshot.language,
    });
  }
  if (variant === "e") {
    return buildFactGroundedLyricsPrompt({
      factSheet: "<PASS 1 FACT SHEET WILL BE INSERTED HERE>",
      durationSeconds: snapshot.durationSeconds,
      maxWords: snapshot.maxWords,
      language: snapshot.language,
    });
  }
  if (variant === "k" || variant === "l" || variant === "m") {
    return buildRhymeAiMinimalLyricsPrompt({
      brief: sourceBrief,
      durationSeconds: snapshot.durationSeconds,
      maxWords: snapshot.maxWords,
      language: snapshot.language,
    });
  }
  if (variant === "o" || variant === "p") {
    return buildRhymeAiLooseLongFormLyricsPrompt({
      brief: sourceBrief,
      durationSeconds: snapshot.durationSeconds,
      maxWords: snapshot.maxWords,
      language: snapshot.language,
    });
  }
  if (variant === "q") {
    return buildRhymeAiLiteralLongFormLyricsPrompt({
      brief: sourceBrief,
      durationSeconds: snapshot.durationSeconds,
      maxWords: snapshot.maxWords,
      language: snapshot.language,
    });
  }
  return buildStrictBudgetLyricsPrompt({
    brief: sourceBrief,
    durationSeconds: snapshot.durationSeconds,
    maxWords: snapshot.maxWords,
    language: snapshot.language,
  });
};

const createRequest = ({
  variant,
  modelId,
  sourceBrief,
  snapshot,
  userPrompt,
}: {
  variant: Variant;
  modelId: string;
  sourceBrief: string;
  snapshot: EvaluationSnapshot;
  userPrompt?: string;
}): GenerationRequest => {
  const selectedModel = lyricLabModelById(modelId);
  return {
    type: "generate",
    variant,
    mode: variant === "e" ? "two-pass" : "single-pass",
    systemPrompt: snapshot.systemPrompts[variant],
    userPrompt: userPrompt ?? generatedUserPrompt(variant, sourceBrief, snapshot),
    sourceBrief,
    factSystemPrompt:
      variant === "e" ? snapshot.factSystemPrompt : undefined,
    factUserPrompt:
      variant === "e"
        ? buildFactExtractionPrompt({
            brief: sourceBrief,
            language: snapshot.language,
          })
        : undefined,
    durationSeconds: snapshot.durationSeconds,
    language: snapshot.language,
    seed: snapshot.seed,
    maxWords: snapshot.maxWords,
    temperature: snapshot.temperature,
    topK: snapshot.topK,
    topP: snapshot.topP,
    repetitionPenalty: snapshot.repetitionPenalty,
    maxNewTokens: snapshot.maxNewTokens,
    modelId: selectedModel.modelId,
    modelRevision: selectedModel.revision,
    modelDtype: selectedModel.dtype,
    modelFamily: selectedModel.family,
    modelLabel: selectedModel.shortLabel,
    modelLocalOnly: selectedModel.localOnly,
    wasmUrl,
    wasmModuleUrl,
  };
};

const requestFor = (variant: Variant) => {
  const userPrompts: Record<Variant, string> = {
    a: userA.value,
    b: userB.value,
    c: userC.value,
    d: userD.value,
    e: userE.value,
    f: userF.value,
    g: userG.value,
    h: userH.value,
    i: userI.value,
    j: userJ.value,
    k: userK.value,
    l: userL.value,
    m: userM.value,
    o: userO.value,
    p: userP.value,
    q: userQ.value,
  };
  return createRequest({
    variant,
    modelId: modelSelect.value,
    sourceBrief: brief.value,
    snapshot: captureEvaluationSnapshot(),
    userPrompt: userPrompts[variant],
  });
};

const batchBaseRow = (
  job: BatchJob,
  status: BatchResultRow["status"],
): BatchResultRow => {
  const model = lyricLabModelById(job.modelId);
  const request = job.request;
  return {
    runIndex: job.runIndex,
    totalRuns: batchTotalRuns,
    status,
    startedAt: job.startedAt ?? "",
    finishedAt: new Date().toISOString(),
    modelId: model.id,
    modelLabel: model.shortLabel,
    modelRepository: model.modelId,
    modelRevision: model.revision,
    modelDtype: model.dtype,
    modelFamily: model.family,
    variant: job.variant,
    storyId: job.story.id,
    storyLabel: job.story.label,
    storyFocus: job.story.focus,
    sourceBrief: job.story.brief,
    language: request.language,
    durationSeconds: request.durationSeconds,
    maxWords: request.maxWords,
    seed: request.seed,
    temperature: request.temperature,
    topK: request.topK,
    topP: request.topP,
    repetitionPenalty: request.repetitionPenalty,
    maxNewTokens: request.maxNewTokens,
    systemPrompt: request.systemPrompt,
    effectiveUserPrompt: request.userPrompt,
    factSystemPrompt: request.factSystemPrompt ?? "",
    factSheet: "",
    factRaw: "",
    rawOutput: "",
    cleanedOutput: "",
    processedLyrics: "",
    issues: "",
    wordCount: "",
    generatedWordCount: "",
    inputTokens: "",
    outputTokens: "",
    modelLoadMilliseconds: "",
    generationMilliseconds: "",
    factInputTokens: "",
    factOutputTokens: "",
    factExtractionMilliseconds: "",
    error: "",
    stack: "",
  };
};

const updateBatchExportControls = () => {
  const hasRows = batchRows.length > 0;
  downloadBatchCsv.disabled = !hasRows;
  downloadBatchJson.disabled = !hasRows;
  clearBatch.disabled = !hasRows || busy;
};

const addBatchTableRow = (row: BatchResultRow) => {
  batchResultsBody.querySelector(".empty-row")?.remove();
  const tr = document.createElement("tr");
  if (row.status !== "success") tr.className = "failed";
  const values = [
    String(row.runIndex),
    row.modelLabel,
    row.storyLabel,
    row.variant.toUpperCase(),
    row.status,
    row.wordCount === "" ? "—" : `${row.wordCount}/${row.maxWords}`,
    row.generationMilliseconds === ""
      ? "—"
      : formatDuration(row.generationMilliseconds),
    row.status === "success" ? row.processedLyrics : row.error,
  ];
  for (const value of values) {
    const td = document.createElement("td");
    td.textContent = value;
    td.title = value;
    tr.append(td);
  }
  batchResultsBody.append(tr);
};

const recordBatchRow = (row: BatchResultRow) => {
  batchRows.push(row);
  batchCompletedRuns += 1;
  addBatchTableRow(row);
  batchProgress.value = batchCompletedRuns / Math.max(1, batchTotalRuns);
  updateBatchExportControls();
};

const finishBatchSuccess = (update: ResultUpdate) => {
  const job = activeBatchJob;
  if (!job) return;
  const row = batchBaseRow(job, "success");
  Object.assign(row, {
    effectiveUserPrompt: update.effectiveUserPrompt,
    factSheet: update.factSheet ?? "",
    factRaw: update.factRaw ?? "",
    rawOutput: update.raw,
    cleanedOutput: update.cleaned,
    processedLyrics: update.processed,
    issues: update.issues.join(" | "),
    wordCount: update.wordCount,
    generatedWordCount: update.generatedWordCount,
    inputTokens: update.inputTokens,
    outputTokens: update.outputTokens,
    modelLoadMilliseconds: update.modelLoadMilliseconds,
    generationMilliseconds: update.generationMilliseconds,
    factInputTokens: update.factInputTokens ?? "",
    factOutputTokens: update.factOutputTokens ?? "",
    factExtractionMilliseconds: update.factExtractionMilliseconds ?? "",
  });
  recordBatchRow(row);
  modelStatus.textContent = `${row.modelLabel} loaded · WebGPU`;
  activeBatchJob = undefined;
  activeVariant = undefined;
  activeWorkerStage = "";
  void runNextBatch();
};

const finishBatchError = (
  message: string,
  stack: string,
  skipRemainingModel: boolean,
) => {
  const job = activeBatchJob;
  if (!job) return;
  const failed = batchBaseRow(job, "error");
  failed.error = message;
  failed.stack = stack;
  recordBatchRow(failed);
  statusFor(job.variant).textContent = `Failed: ${message}`;

  if (skipRemainingModel) {
    while (batchQueue[0]?.modelId === job.modelId) {
      const skippedJob = batchQueue.shift();
      if (!skippedJob) break;
      const skipped = batchBaseRow(skippedJob, "skipped");
      skipped.error = `Skipped after ${job.request.modelLabel} failed: ${message}`;
      recordBatchRow(skipped);
    }
    activeWorkerModelId = undefined;
  }
  activeBatchJob = undefined;
  activeVariant = undefined;
  activeWorkerStage = "";
  void runNextBatch();
};

const finishBatch = (message: string) => {
  batchProgress.value = batchTotalRuns
    ? batchCompletedRuns / batchTotalRuns
    : 0;
  batchStatus.textContent = message;
  progress.value = batchCancelled ? progress.value : 1;
  progressLabel.textContent = message;
  activeBatchJob = undefined;
  activeVariant = undefined;
  runMode = undefined;
  setControls(false);
  updateBatchExportControls();
};

const runNextBatch = async () => {
  if (batchCancelled) {
    finishBatch(
      `Cancelled after ${batchCompletedRuns}/${batchTotalRuns} runs; partial results are exportable.`,
    );
    return;
  }
  const job = batchQueue.shift();
  if (!job) {
    finishBatch(`Batch complete · ${batchCompletedRuns}/${batchTotalRuns} rows ready.`);
    return;
  }

  if (activeWorkerModelId !== job.modelId) {
    await releaseWorker();
    if (batchCancelled) {
      finishBatch(
        `Cancelled after ${batchCompletedRuns}/${batchTotalRuns} runs; partial results are exportable.`,
      );
      return;
    }
    activeWorkerModelId = job.modelId;
    downloads.clear();
  }

  activeBatchJob = job;
  activeVariant = job.variant;
  activeWorkerStage = "";
  job.startedAt = new Date().toISOString();
  modelSelect.value = job.modelId;
  statusFor(job.variant).textContent =
    `Batch ${job.runIndex}/${batchTotalRuns}: ${job.story.label}`;
  batchStatus.textContent =
    `${job.runIndex}/${batchTotalRuns} · ${job.request.modelLabel} · ${job.story.label} · prompt ${job.variant.toUpperCase()}`;
  progress.value = batchCompletedRuns / Math.max(1, batchTotalRuns);
  progress.classList.remove("failed");
  ensureWorker().postMessage(job.request);
};

const startBatch = () => {
  if (busy) return;
  const modelIds = checkedBatchValues("model");
  const variants = checkedBatchValues("variant") as Variant[];
  const storyIds = checkedBatchValues("story");
  if (!modelIds.length || !variants.length || !storyIds.length) {
    batchStatus.textContent = "Select at least one model, prompt, and story.";
    return;
  }
  try {
    validate();
  } catch (error) {
    batchStatus.textContent = error instanceof Error ? error.message : String(error);
    return;
  }

  const snapshot = captureEvaluationSnapshot();
  const descriptors = buildBatchMatrix({
    modelIds,
    variants,
    stories: storyIds.map(lyricTestBriefById),
  });
  batchTotalRuns = descriptors.length;
  batchCompletedRuns = 0;
  batchRows = [];
  batchCancelled = false;
  batchResultsBody.innerHTML =
    '<tr class="empty-row"><td colspan="8">Batch starting…</td></tr>';
  batchQueue = descriptors.map((descriptor, index) => ({
    ...descriptor,
    runIndex: index + 1,
    request: createRequest({
      variant: descriptor.variant,
      modelId: descriptor.modelId,
      sourceBrief: descriptor.story.brief,
      snapshot,
    }),
  }));
  queue = [];
  runMode = "batch";
  activeWorkerModelId = undefined;
  batchProgress.value = 0;
  batchStatus.textContent = `Starting ${batchTotalRuns} runs…`;
  updateBatchExportControls();
  setControls(true);
  void runNextBatch();
};

const downloadTextFile = (
  contents: string,
  mimeType: string,
  extension: string,
) => {
  const timestamp = new Date().toISOString().replaceAll(":", "-");
  const blob = new Blob([contents], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `ai-music-js-lyrics-evaluation-${timestamp}.${extension}`;
  anchor.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 0);
};

const validate = () => {
  if (!brief.value.trim()) throw new Error("Enter a song brief.");
  if (!Number.isInteger(numberValue(seed)) || numberValue(seed) < 0) {
    throw new Error("Seed must be a non-negative whole number.");
  }
  if (!Number.isInteger(numberValue(maxWords)) || numberValue(maxWords) < 20) {
    throw new Error("Word budget must be a whole number of at least 20.");
  }
};

const runNext = async () => {
  const variant = queue.shift();
  if (!variant) {
    progress.value = 1;
    progressLabel.textContent = "Comparison ready";
    runMode = undefined;
    setControls(false);
    updateBatchExportControls();
    return;
  }
  activeVariant = variant;
  activeWorkerModelId = modelSelect.value;
  statusFor(variant).textContent = "Queued for the selected lyric model…";
  progress.value = 0;
  progress.classList.remove("failed");
  ensureWorker().postMessage(requestFor(variant));
};

const start = (variants: Variant[]) => {
  if (busy) return;
  try {
    validate();
  } catch (error) {
    progressLabel.textContent = error instanceof Error ? error.message : String(error);
    return;
  }
  setControls(true);
  runMode = "interactive";
  batchCancelled = false;
  downloads.clear();
  queue = [...variants];
  void runNext();
};

modelSelect.addEventListener("change", () => {
  const selectedModel = lyricLabModelById(modelSelect.value);
  setControls(true);
  cancel.disabled = true;
  queue = [];
  activeVariant = undefined;
  activeWorkerModelId = undefined;
  downloads.clear();
  progress.value = 0;
  progressLabel.textContent = "Releasing the previous WebGPU model…";
  modelStatus.textContent = `Switching to ${selectedModel.shortLabel}`;
  void releaseWorker().then(() => {
    progressLabel.textContent = "Ready";
    modelStatus.textContent = `${selectedModel.shortLabel} selected`;
    setControls(false);
  });
});

runRecommended.addEventListener("click", () => start(["c", "g", "i"]));
runRhymeAi.addEventListener("click", () => start(["k", "o", "p", "q"]));
runAll.addEventListener("click", () =>
  start(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "o", "p", "q"]),
);
for (const button of document.querySelectorAll<HTMLButtonElement>(".run-one")) {
  button.addEventListener("click", () => start([button.dataset.run as Variant]));
}
runBatch.addEventListener("click", startBatch);

downloadBatchCsv.addEventListener("click", () => {
  downloadTextFile(
    serializeBatchResultsCsv(batchRows),
    "text/csv;charset=utf-8",
    "csv",
  );
});
downloadBatchJson.addEventListener("click", () => {
  downloadTextFile(
    JSON.stringify(
      {
        schemaVersion: 1,
        exportedAt: new Date().toISOString(),
        resultCount: batchRows.length,
        results: batchRows,
      },
      null,
      2,
    ),
    "application/json;charset=utf-8",
    "json",
  );
});
clearBatch.addEventListener("click", () => {
  batchRows = [];
  batchCompletedRuns = 0;
  batchTotalRuns = 0;
  batchProgress.value = 0;
  batchStatus.textContent = "Ready";
  batchResultsBody.innerHTML =
    '<tr class="empty-row"><td colspan="8">No batch results yet.</td></tr>';
  updateBatchExportControls();
});

cancel.addEventListener("click", () => {
  const wasBatch = runMode === "batch";
  batchCancelled = wasBatch;
  worker?.terminate();
  worker = undefined;
  queue = [];
  batchQueue = [];
  activeVariant = undefined;
  activeBatchJob = undefined;
  activeWorkerModelId = undefined;
  downloads.clear();
  progress.value = 0;
  progressLabel.textContent = wasBatch
    ? `Cancelled after ${batchCompletedRuns}/${batchTotalRuns} runs; partial results are exportable.`
    : "Cancelled; the model will reload on the next run.";
  if (wasBatch) batchStatus.textContent = progressLabel.textContent;
  modelStatus.textContent = "Model released";
  runMode = undefined;
  setControls(false);
  updateBatchExportControls();
});

for (const button of document.querySelectorAll<HTMLButtonElement>(".copy")) {
  button.addEventListener("click", async () => {
    const target = button.dataset.copy;
    if (!target) return;
    const value = required<HTMLElement>(`#${target}`).textContent ?? "";
    await navigator.clipboard.writeText(value);
    const previous = button.textContent;
    button.textContent = "Copied";
    window.setTimeout(() => {
      button.textContent = previous;
    }, 1200);
  });
}
