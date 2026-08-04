import {
  AceStepWebGpu,
  AceStepWebGpuError,
  DEFAULT_INSTRUMENTAL_PROMPT,
  DEFAULT_VOCAL_PROMPT,
  assessLyricDuration,
  defaultMaxLyricWords,
  type CacheInventory,
  type PlannerProfileReport,
  type WorkerUpdate,
} from "ai-music-js";
import "./styles.css";

const prompt = document.querySelector<HTMLTextAreaElement>("#prompt");
const mode = document.querySelector<HTMLSelectElement>("#mode");
const audioQuality =
  document.querySelector<HTMLSelectElement>("#audio-quality");
const plannerQuality =
  document.querySelector<HTMLSelectElement>("#planner-quality");
const lyricsPanel = document.querySelector<HTMLElement>("#lyrics-panel");
const lyrics = document.querySelector<HTMLTextAreaElement>("#lyrics");
const lyricsGuidance =
  document.querySelector<HTMLElement>("#lyrics-guidance");
const lyricsFit =
  document.querySelector<HTMLElement>("#lyrics-fit");
const vocalLanguage =
  document.querySelector<HTMLSelectElement>("#vocal-language");
const sampler = document.querySelector<HTMLSelectElement>("#sampler");
const samplerGuidance =
  document.querySelector<HTMLElement>("#sampler-guidance");
const seed = document.querySelector<HTMLInputElement>("#seed");
const duration = document.querySelector<HTMLSelectElement>("#duration");
const batchSize = document.querySelector<HTMLSelectElement>("#batch-size");
const dcwEnabled =
  document.querySelector<HTMLInputElement>("#dcw-enabled");
const dcwMode = document.querySelector<HTMLSelectElement>("#dcw-mode");
const dcwScaler =
  document.querySelector<HTMLInputElement>("#dcw-scaler");
const dcwHighScaler =
  document.querySelector<HTMLInputElement>("#dcw-high-scaler");
const stage = document.querySelector<HTMLElement>("#stage");
const detail = document.querySelector<HTMLElement>("#detail");
const progress = document.querySelector<HTMLProgressElement>("#progress");
const progressLabel = document.querySelector<HTMLElement>("#progress-label");
const generationElapsed =
  document.querySelector<HTMLElement>("#generation-elapsed");
const timingList = document.querySelector<HTMLElement>("#timing-list");
const plannerProfile =
  document.querySelector<HTMLElement>("#planner-profile");
const plannerProfileStatus =
  document.querySelector<HTMLElement>("#planner-profile-status");
const plannerProfileInput =
  document.querySelector<HTMLElement>("#planner-profile-input");
const plannerProfileEmbedding =
  document.querySelector<HTMLElement>("#planner-profile-embedding");
const plannerProfileRows =
  document.querySelector<HTMLTableSectionElement>("#planner-profile-rows");
const audio = document.querySelector<HTMLAudioElement>("#audio");
const generate = document.querySelector<HTMLButtonElement>("#generate");
const cancel = document.querySelector<HTMLButtonElement>("#cancel");
const download = document.querySelector<HTMLAnchorElement>("#download");
const batchResults = document.querySelector<HTMLElement>("#batch-results");
const log = document.querySelector<HTMLPreElement>("#log");
const refreshCache =
  document.querySelector<HTMLButtonElement>("#refresh-cache");
const clearCache = document.querySelector<HTMLButtonElement>("#clear-cache");
const cacheSummary = document.querySelector<HTMLElement>("#cache-summary");
const storageAvailable =
  document.querySelector<HTMLElement>("#storage-available");
const storageUsed = document.querySelector<HTMLElement>("#storage-used");
const storageQuota = document.querySelector<HTMLElement>("#storage-quota");
const cacheList = document.querySelector<HTMLElement>("#cache-list");

if (
  !prompt ||
  !mode ||
  !audioQuality ||
  !plannerQuality ||
  !lyricsPanel ||
  !lyrics ||
  !lyricsGuidance ||
  !lyricsFit ||
  !vocalLanguage ||
  !sampler ||
  !samplerGuidance ||
  !seed ||
  !duration ||
  !batchSize ||
  !dcwEnabled ||
  !dcwMode ||
  !dcwScaler ||
  !dcwHighScaler ||
  !stage ||
  !detail ||
  !progress ||
  !progressLabel ||
  !generationElapsed ||
  !timingList ||
  !plannerProfile ||
  !plannerProfileStatus ||
  !plannerProfileInput ||
  !plannerProfileEmbedding ||
  !plannerProfileRows ||
  !audio ||
  !generate ||
  !cancel ||
  !download ||
  !batchResults ||
  !log ||
  !refreshCache ||
  !clearCache ||
  !cacheSummary ||
  !storageAvailable ||
  !storageUsed ||
  !storageQuota ||
  !cacheList
) {
  throw new Error("Package smoke-test DOM is incomplete.");
}

prompt.value = DEFAULT_INSTRUMENTAL_PROMPT;

let audioUrls: string[] = [];
let appBusy = false;
let generating = false;
let generationStartedAt: number | null = null;
let generationClockTimer: number | null = null;
const downloads = new Map<string, { loaded: number; total: number }>();
const generationTimings = new Map<string, number>();

const formatElapsed = (milliseconds: number) => {
  const totalSeconds = Math.max(0, milliseconds) / 1000;
  if (totalSeconds < 60) return `${totalSeconds.toFixed(1)}s`;
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  return hours
    ? `${hours}h ${String(minutes).padStart(2, "0")}m ${seconds.toFixed(1).padStart(4, "0")}s`
    : `${minutes}m ${seconds.toFixed(1).padStart(4, "0")}s`;
};

const renderGenerationClock = () => {
  if (generationStartedAt === null) return;
  generationElapsed.textContent = formatElapsed(
    performance.now() - generationStartedAt,
  );
};

const startGenerationClock = () => {
  if (generationClockTimer !== null) {
    window.clearInterval(generationClockTimer);
  }
  generationStartedAt = performance.now();
  generationElapsed.parentElement?.classList.add("is-running");
  renderGenerationClock();
  generationClockTimer = window.setInterval(
    renderGenerationClock,
    100,
  );
};

const stopGenerationClock = () => {
  if (generationStartedAt === null) return undefined;
  if (generationClockTimer !== null) {
    window.clearInterval(generationClockTimer);
    generationClockTimer = null;
  }
  const milliseconds = performance.now() - generationStartedAt;
  generationElapsed.textContent = formatElapsed(milliseconds);
  generationElapsed.parentElement?.classList.remove("is-running");
  generationStartedAt = null;
  return milliseconds;
};

const requestedDurationSeconds = () =>
  duration.value === "auto" ? 30 : Number(duration.value);

const updateLyricFit = () => {
  const vocalsEnabled = mode.value !== "instrumental";
  if (!vocalsEnabled) {
    lyricsFit.textContent = "";
    lyricsFit.className = "lyrics-fit";
    return;
  }
  const requested = requestedDurationSeconds();
  if (mode.value === "ai-vocals") {
    lyricsFit.textContent =
      `Qwen is limited to ${defaultMaxLyricWords(requested)} words for the ` +
      `${requested}-second preference. ` +
      (duration.value === "auto"
        ? plannerQuality.value === "high-quality"
          ? "ACE Phase 1 will choose the final duration."
          : "The direct Turbo path will use the duration recommendation."
        : "Select Auto if ACE should be allowed to extend it.");
    lyricsFit.className = "lyrics-fit is-ok";
    return;
  }
  const assessment = assessLyricDuration(
    lyrics.value,
    requested,
    {
      minimumDurationSeconds: Math.max(30, requested),
      maximumDurationSeconds: 120,
    },
  );
  if (!assessment.wordCount) {
    lyricsFit.textContent =
      `This duration supports up to ${assessment.selectedWordBudget} sung words.`;
    lyricsFit.className = "lyrics-fit";
    return;
  }
  if (duration.value === "auto") {
    lyricsFit.textContent =
      `${assessment.wordCount} sung words · automatic duration will use at least ` +
      `${assessment.recommendedDurationSeconds} seconds` +
      (plannerQuality.value === "high-quality"
        ? "; ACE Phase 1 may choose longer."
        : ". Enable High quality planning for ACE-generated metadata.");
    lyricsFit.className = "lyrics-fit is-ok";
    return;
  }
  if (!assessment.fits) {
    lyricsFit.textContent =
      `${assessment.wordCount} sung words exceed the ${requested}-second ` +
      `budget of ${assessment.selectedWordBudget} by ${assessment.exceedsByWords}. ` +
      `Use at least ${assessment.recommendedDurationSeconds} seconds or select Auto.`;
    lyricsFit.className = "lyrics-fit is-warning";
    return;
  }
  lyricsFit.textContent =
    `${assessment.wordCount}/${assessment.selectedWordBudget} sung words · ` +
    `fits the selected duration.`;
  lyricsFit.className = "lyrics-fit is-ok";
};

const formatBytes = (bytes: number) => {
  if (bytes < 1_000_000) return `${(bytes / 1_000).toFixed(1)} kB`;
  if (bytes < 1_000_000_000) return `${(bytes / 1_000_000).toFixed(1)} MB`;
  return `${(bytes / 1_000_000_000).toFixed(2)} GB`;
};

const appendLog = (message: string) => {
  const timestamp = new Date().toLocaleTimeString();
  log.textContent = `${log.textContent}${timestamp}  ${message}\n`
    .split("\n")
    .slice(-15)
    .join("\n");
  log.scrollTop = log.scrollHeight;
};

const renderTimingSummary = () => {
  const entries = [...generationTimings.entries()]
    .filter(([name]) => !/^(?:euler|heun|euler-sde):\d+$/.test(name))
    .filter(([name]) => !/^vae-decode:\d+$/.test(name))
    .sort((left, right) => right[1] - left[1])
    .slice(0, 18);
  if (!entries.length) {
    const empty = document.createElement("span");
    empty.className = "timing-empty";
    empty.textContent = "Run a generation to collect timings.";
    timingList.replaceChildren(empty);
    return;
  }
  timingList.replaceChildren(
    ...entries.map(([name, milliseconds]) => {
      const row = document.createElement("div");
      row.className = "timing-row";
      const label = document.createElement("span");
      label.textContent = name.replaceAll("pipeline:", "pipeline · ");
      const value = document.createElement("strong");
      value.textContent = formatElapsed(milliseconds);
      row.append(label, value);
      return row;
    }),
  );
};

const formatProfileTime = (milliseconds: number) =>
  milliseconds < 1_000
    ? `${milliseconds.toFixed(milliseconds < 10 ? 2 : 1)}ms`
    : formatElapsed(milliseconds);

const shortHash = (hash: string) => `${hash.slice(0, 12)}…`;

const renderPlannerProfile = (report?: PlannerProfileReport) => {
  if (!report) {
    plannerProfile.hidden = true;
    plannerProfileStatus.textContent = "Waiting";
    plannerProfileInput.textContent = "";
    plannerProfileEmbedding.textContent = "";
    plannerProfileRows.replaceChildren();
    return;
  }
  plannerProfile.hidden = false;
  plannerProfileStatus.textContent = report.final
    ? `Final · ${report.completedSemanticSteps}/${report.targetSemanticSteps} codes`
    : `Running · ${report.completedSemanticSteps}/${report.targetSemanticSteps} codes`;
  const input = report.input;
  plannerProfileInput.textContent = input
    ? `${input.paddedTokens} padded tokens · ${input.realTokens.join("/")} real ` +
      `(conditional/unconditional) · input IDs ${shortHash(input.conditionalInputIdsSha256)} / ` +
      `${shortHash(input.unconditionalInputIdsSha256)} · prompt ${shortHash(input.promptSha256)} · ` +
      `lyrics ${shortHash(input.lyricsSha256)} · metadata ${shortHash(input.metadataReasoningSha256)}`
    : "Input fingerprint is not available.";
  const { total, metadata, semantic } = report.embedding;
  plannerProfileEmbedding.textContent =
    `Sparse embedding source: ${report.embeddingSource} · total ${total.rangeRequests} range requests / ` +
    `${total.persistentHits} persistent hits / ${total.memoryHits} memory hits / ` +
    `${total.injectedRows} head-row reuses / ${formatBytes(total.fetchedBytes)} fetched · ` +
    `metadata ${metadata.rangeRequests} requests · semantic ${semantic.rangeRequests} requests.`;
  plannerProfileRows.replaceChildren(
    ...report.metrics.map((metric) => {
      const row = document.createElement("tr");
      if (metric.includesChildren) row.className = "is-parent";
      const label = document.createElement("td");
      label.textContent = `${metric.phase} · ${metric.label}`;
      const totalCell = document.createElement("td");
      totalCell.textContent = formatProfileTime(metric.totalMilliseconds);
      const calls = document.createElement("td");
      calls.textContent = String(metric.calls);
      const average = document.createElement("td");
      average.textContent = formatProfileTime(metric.averageMilliseconds);
      const minimum = document.createElement("td");
      minimum.textContent = formatProfileTime(metric.minimumMilliseconds);
      const maximum = document.createElement("td");
      maximum.textContent = formatProfileTime(metric.maximumMilliseconds);
      row.append(label, totalCell, calls, average, minimum, maximum);
      return row;
    }),
  );
};

const clearAudioResults = () => {
  for (const url of audioUrls) {
    URL.revokeObjectURL(url);
  }
  audioUrls = [];
  audio.pause();
  audio.removeAttribute("src");
  audio.hidden = true;
  download.hidden = true;
  batchResults.replaceChildren();
};

const updateModeControls = () => {
  const vocalsEnabled = mode.value !== "instrumental";
  const aiLyrics = mode.value === "ai-vocals";
  lyricsPanel.hidden = !vocalsEnabled;
  lyrics.disabled = !vocalsEnabled || aiLyrics || appBusy;
  vocalLanguage.disabled = !vocalsEnabled || appBusy;
  lyricsGuidance.textContent = aiLyrics
    ? "Qwen3.5 writes this field locally before ACE-Step starts. The model is about 0.49 GB."
    : "These lyrics are passed directly to ACE-Step.";
  lyrics.placeholder = aiLyrics
    ? "Qwen3.5-generated lyrics will appear here."
    : "[Verse]\nWrite your first verse here\n\n[Chorus]\nWrite a memorable chorus";
  const sdeOption = sampler.querySelector<HTMLOptionElement>(
    'option[value="euler-sde"]',
  );
  if (sdeOption) {
    sdeOption.disabled = vocalsEnabled;
  }
  updateLyricFit();
};

const updateSamplerGuidance = () => {
  samplerGuidance.textContent =
    sampler.value === "heun"
      ? "Vocal-compatible predictor/corrector; about 15 DiT evaluations."
      : sampler.value === "euler-sde"
        ? "Experimental and currently restricted to instrumental output."
        : "Recommended and verified for instrumental and vocal output.";
};

const applyModeDefaults = () => {
  const vocalsEnabled = mode.value !== "instrumental";
  const currentPrompt = prompt.value.trim();
  if (
    vocalsEnabled &&
    currentPrompt === DEFAULT_INSTRUMENTAL_PROMPT
  ) {
    prompt.value = DEFAULT_VOCAL_PROMPT;
  } else if (
    !vocalsEnabled &&
    currentPrompt === DEFAULT_VOCAL_PROMPT
  ) {
    prompt.value = DEFAULT_INSTRUMENTAL_PROMPT;
  }
  if (vocalsEnabled && sampler.value === "euler-sde") {
    sampler.value = "euler";
    detail.textContent =
      "Vocal mode uses Euler by default; Euler SDE failed the vocal quality gate.";
  }
  updateModeControls();
  updateSamplerGuidance();
};

const updateDcwControls = () => {
  const enabled = dcwEnabled.checked && !appBusy;
  dcwMode.disabled = !enabled;
  dcwScaler.disabled = !enabled;
  dcwHighScaler.disabled =
    !enabled || dcwMode.value !== "double";
};

const updateControlState = () => {
  generate.disabled = appBusy;
  cancel.disabled = !generating;
  refreshCache.disabled = appBusy;
  clearCache.disabled = appBusy;
  prompt.disabled = appBusy;
  mode.disabled = appBusy;
  audioQuality.disabled = appBusy;
  plannerQuality.disabled = appBusy;
  sampler.disabled = appBusy;
  seed.disabled = appBusy;
  duration.disabled = appBusy;
  batchSize.disabled = appBusy;
  dcwEnabled.disabled = appBusy;
  updateModeControls();
  updateDcwControls();
  for (const button of cacheList.querySelectorAll<HTMLButtonElement>(
    "[data-model-id]",
  )) {
    button.disabled = appBusy || button.dataset.cached !== "true";
  }
};

const renderCacheInventory = (inventory: CacheInventory) => {
  const availableBytes =
    inventory.availableBytes ??
    (inventory.quotaBytes !== undefined &&
    inventory.usageBytes !== undefined
      ? Math.max(0, inventory.quotaBytes - inventory.usageBytes)
      : undefined);
  storageAvailable.textContent =
    availableBytes === undefined ? "Not reported" : formatBytes(availableBytes);
  storageUsed.textContent =
    inventory.usageBytes === undefined
      ? "Not reported"
      : formatBytes(inventory.usageBytes);
  storageQuota.textContent =
    inventory.quotaBytes === undefined
      ? "Not reported"
      : formatBytes(inventory.quotaBytes);

  const quota =
    inventory.quotaBytes === undefined
      ? "quota unavailable"
      : `${formatBytes(inventory.usageBytes ?? 0)} origin usage / ${formatBytes(inventory.quotaBytes)} quota`;
  const availability =
    availableBytes === undefined
      ? "available space unavailable"
      : `${formatBytes(availableBytes)} available`;
  const persistence =
    inventory.persisted === undefined
      ? "persistence unknown"
      : inventory.persisted
        ? "persistent storage granted"
        : "persistent storage not granted";
  cacheSummary.textContent =
    `${formatBytes(inventory.storedBytes)} of ${formatBytes(inventory.expectedBytes)} model data stored · ${availability} · ${quota} · ${persistence} · ${inventory.origin}`;

  const rows = inventory.models.map((model) => {
    const row = document.createElement("article");
    row.className = "cache-row";

    const copy = document.createElement("div");
    copy.className = "cache-copy";
    const heading = document.createElement("h3");
    heading.textContent = model.label;
    const status = document.createElement("p");
    status.className = `cache-status ${model.complete ? "complete" : model.partial ? "partial" : ""}`;
    status.textContent = model.complete
      ? `Downloaded · ${formatBytes(model.storedBytes)}`
      : model.partial
        ? `Partial · ${formatBytes(model.storedBytes)} / ${formatBytes(model.expectedBytes)}`
        : `Not downloaded · ${formatBytes(model.expectedBytes)}`;
    const files = document.createElement("details");
    const filesSummary = document.createElement("summary");
    const cachedFiles = model.assets.filter((asset) => asset.cached).length;
    filesSummary.textContent =
      `${cachedFiles} of ${model.assets.length} files ready`;
    const fileList = document.createElement("ul");
    for (const asset of model.assets) {
      const item = document.createElement("li");
      const assetState = asset.cached
        ? `${formatBytes(asset.storedBytes)} · ${asset.storage}`
        : asset.storedBytes > 0
          ? `partial ${formatBytes(asset.storedBytes)} · ${asset.storage}`
          : "not stored";
      item.textContent = `${asset.fileName} — ${assetState}`;
      fileList.append(item);
    }
    files.append(filesSummary, fileList);
    copy.append(heading, status, files);
    row.append(copy);

    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "secondary cache-remove";
    remove.dataset.modelId = model.id;
    remove.dataset.cached = model.storedBytes > 0 ? "true" : "false";
    remove.textContent = "Remove";
    remove.disabled = appBusy || model.storedBytes === 0;
    row.append(remove);
    return row;
  });
  cacheList.replaceChildren(...rows);
  updateControlState();
};

const report = (update: WorkerUpdate) => {
  if (update.type === "progress") {
    progress.max = 1;
    progress.value = update.progress;
    progressLabel.textContent = `${Math.round(update.progress * 100)}%`;
    stage.textContent = update.stage.replaceAll("-", " ");
    if (update.detail) detail.textContent = update.detail;
    return;
  }

  if (update.type === "download") {
    downloads.set(update.assetId, {
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
    progress.max = Math.max(1, totals.total);
    progress.value = totals.loaded;
    progressLabel.textContent = `${formatBytes(totals.loaded)} / ${formatBytes(totals.total)}`;
    detail.textContent = update.cached
      ? `${update.label} loaded from browser cache`
      : `Downloading ${update.label}`;
    return;
  }

  if (update.type === "stage") {
    stage.textContent = update.stage.replaceAll("-", " ");
    detail.textContent = update.detail;
    appendLog(`${update.stage}: ${update.detail}`);
    return;
  }

  if (update.type === "batch-progress") {
    const number = update.index + 1;
    stage.textContent = `batch ${number} of ${update.total}`;
    detail.textContent =
      update.status === "started"
        ? `Generating seed ${update.seed}.`
        : `Seed ${update.seed} completed.`;
    appendLog(
      `batch ${number}/${update.total}: seed ${update.seed} ${update.status}`,
    );
    return;
  }

  if (update.type === "compatibility") {
    appendLog(update.message);
    return;
  }

  if (update.type === "timing") {
    generationTimings.set(
      update.stage,
      (generationTimings.get(update.stage) ?? 0) + update.milliseconds,
    );
    renderTimingSummary();
    appendLog(`${update.stage} finished in ${(update.milliseconds / 1000).toFixed(2)}s`);
    return;
  }

  if (update.type === "planner-profile") {
    renderPlannerProfile(update.report);
    return;
  }

  if (update.type === "plan-complete") {
    appendLog(
      `ACE ${update.plannerQuality} plan: ${update.metadata.bpm} BPM · ${update.metadata.keyScale} · ${update.metadata.timeSignature}/4 · ${update.semanticCodeIds.length} semantic codes`,
    );
    return;
  }

  if (update.type === "diagnostic") {
    appendLog(`${update.key}: ${update.value}`);
    return;
  }

  if (update.type === "cached-model-removed") {
    appendLog(
      `Removed ${update.modelId} cache (${formatBytes(update.removedBytes)}).`,
    );
  }
};

const localModelBaseUrl =
  import.meta.env.VITE_ACE_MODEL_BASE_URL?.trim();
const runtime = new AceStepWebGpu({
  onUpdate: report,
  ...(localModelBaseUrl
    ? { modelBaseUrl: localModelBaseUrl }
    : {}),
});

mode.addEventListener("change", applyModeDefaults);
lyrics.addEventListener("input", updateLyricFit);
duration.addEventListener("change", updateLyricFit);
plannerQuality.addEventListener("change", updateLyricFit);
sampler.addEventListener("change", updateSamplerGuidance);
dcwEnabled.addEventListener("change", updateDcwControls);
dcwMode.addEventListener("change", updateDcwControls);
updateModeControls();
updateSamplerGuidance();
updateDcwControls();

const inspectCache = async () => {
  appBusy = true;
  updateControlState();
  cacheSummary.textContent = "Inspecting model files stored by this site…";
  storageAvailable.textContent = "Checking…";
  storageUsed.textContent = "Checking…";
  storageQuota.textContent = "Checking…";
  try {
    const inventory = await runtime.listCachedModels();
    renderCacheInventory(inventory);
  } catch (error) {
    storageAvailable.textContent = "Unavailable";
    storageUsed.textContent = "Unavailable";
    storageQuota.textContent = "Unavailable";
    cacheSummary.textContent =
      `Could not inspect browser storage: ${error instanceof Error ? error.message : String(error)}`;
  } finally {
    appBusy = false;
    updateControlState();
  }
};

generate.addEventListener("click", async () => {
  const promptValue = prompt.value.trim();
  if (!promptValue) {
    prompt.focus();
    return;
  }
  const lyricsValue = mode.value === "vocals" ? lyrics.value.trim() : "";
  if (mode.value === "vocals" && !lyricsValue) {
    lyrics.focus();
    detail.textContent = "Add lyrics before starting vocal generation.";
    return;
  }

  appBusy = true;
  generating = true;
  startGenerationClock();
  updateControlState();
  clearAudioResults();
  downloads.clear();
  generationTimings.clear();
  renderTimingSummary();
  renderPlannerProfile();
  progress.removeAttribute("value");
  progressLabel.textContent = "Preparing runtime";
  stage.textContent = "starting";
  detail.textContent = "Checking WebGPU and browser storage.";

  try {
    const baseSeed = Number(seed.value);
    const count = Number(batchSize.value);
    const seeds = Array.from(
      { length: count },
      (_, index) => (baseSeed + index) >>> 0,
    );
    const generationOptions = {
      prompt: promptValue,
      audioQuality: audioQuality.value as "standard" | "high",
      plannerQuality: plannerQuality.value as
        | "turbo"
        | "high-quality",
      lyrics: lyricsValue,
      writeLyrics: mode.value === "ai-vocals",
      vocalLanguage: vocalLanguage.value,
      durationSeconds: requestedDurationSeconds(),
      autoDuration: duration.value === "auto",
      sampler: sampler.value as "euler" | "heun" | "euler-sde",
      dcw: {
        enabled: dcwEnabled.checked,
        mode: dcwMode.value as "low" | "high" | "double" | "pix",
        scaler: Number(dcwScaler.value),
        highScaler: Number(dcwHighScaler.value),
      },
    };
    const results =
      count === 1
        ? [
            await runtime.generate({
              ...generationOptions,
              seed: baseSeed,
            }),
          ]
        : await runtime.generateBatch({
            ...generationOptions,
            seeds,
          });

    if (results.length === 1) {
      const [result] = results;
      if (result.lyrics) {
        lyrics.value = result.lyrics;
      }
      const audioUrl = URL.createObjectURL(result.wav);
      audioUrls.push(audioUrl);
      audio.src = audioUrl;
      audio.hidden = false;
      download.href = audioUrl;
      download.download =
        `ai-music-${result.seed}-${result.durationSeconds}s-${result.sampler}.wav`;
      download.hidden = false;
    } else {
      if (results[0]?.lyrics) {
        lyrics.value = results[0].lyrics;
      }
      const cards = results.map((result, index) => {
        const audioUrl = URL.createObjectURL(result.wav);
        audioUrls.push(audioUrl);
        const card = document.createElement("article");
        card.className = "batch-result";
        const heading = document.createElement("h3");
        heading.textContent = `Result ${index + 1} · seed ${result.seed}`;
        const metadata = document.createElement("p");
        metadata.textContent =
          `${result.durationSeconds}s · ${result.sampler} · ${
            result.instrumental ? "instrumental" : "vocals"
          }`;
        const player = document.createElement("audio");
        player.controls = true;
        player.src = audioUrl;
        const save = document.createElement("a");
        save.className = "download";
        save.href = audioUrl;
        save.download =
          `ai-music-${result.seed}-${result.durationSeconds}s-${result.sampler}.wav`;
        save.textContent = "Download WAV";
        card.append(heading, metadata, player, save);
        return card;
      });
      batchResults.replaceChildren(...cards);
    }
    const totalPeak = Math.max(
      ...results.map((result) => result.estimatedPeakBytes),
    );
    for (const result of results) {
      for (const [name, milliseconds] of Object.entries(result.timings)) {
        if (name.startsWith("pipeline:")) {
          generationTimings.set(
            name,
            (generationTimings.get(name) ?? 0) + milliseconds,
          );
        }
      }
    }
    renderTimingSummary();
    progress.max = 1;
    progress.value = 1;
    progressLabel.textContent = "Complete";
    stage.textContent = "music ready";
    detail.textContent =
      `${results.length} × ${results[0]!.durationSeconds}s stereo WAV · ${results[0]!.sampler} · estimated peak ${formatBytes(totalPeak)}`;
    appendLog(
      `${results.length} generation${results.length === 1 ? "" : "s"} completed.`,
    );
  } catch (error) {
    stage.textContent = "generation stopped";
    detail.textContent =
      error instanceof AceStepWebGpuError
        ? `${error.stage}: ${error.message}`
        : error instanceof Error
          ? error.message
          : String(error);
    appendLog(detail.textContent);
  } finally {
    const elapsedMilliseconds = stopGenerationClock();
    if (elapsedMilliseconds !== undefined) {
      generationTimings.set("demo:click-to-ready", elapsedMilliseconds);
      renderTimingSummary();
      appendLog(
        `Total generation time: ${formatElapsed(elapsedMilliseconds)}.`,
      );
    }
    appBusy = false;
    generating = false;
    updateControlState();
    void inspectCache();
  }
});

cancel.addEventListener("click", () => {
  runtime.cancel();
  generating = false;
  updateControlState();
});

refreshCache.addEventListener("click", () => {
  void inspectCache();
});

clearCache.addEventListener("click", async () => {
  if (
    !window.confirm(
      "Remove every ai-music-js model file stored by this site? A fully cold AI-lyrics generation can download about 9.57 GB again.",
    )
  ) {
    return;
  }
  appBusy = true;
  updateControlState();
  cacheSummary.textContent = "Removing all stored model data…";
  try {
    await runtime.clearCache();
    const inventory = await runtime.listCachedModels();
    renderCacheInventory(inventory);
    appendLog("Removed all ai-music-js model data from this origin.");
  } catch (error) {
    cacheSummary.textContent =
      `Could not clear browser storage: ${error instanceof Error ? error.message : String(error)}`;
  } finally {
    appBusy = false;
    updateControlState();
  }
});

cacheList.addEventListener("click", async (event) => {
  const target =
    event.target instanceof Element
      ? event.target.closest<HTMLButtonElement>("[data-model-id]")
      : null;
  const modelId = target?.dataset.modelId;
  if (!modelId || appBusy) {
    return;
  }
  const label =
    target.closest(".cache-row")?.querySelector("h3")?.textContent ??
    modelId;
  if (
    !window.confirm(
      `Remove ${label} from this site's browser storage? It will be downloaded again when needed.`,
    )
  ) {
    return;
  }
  appBusy = true;
  updateControlState();
  cacheSummary.textContent = `Removing ${label}…`;
  try {
    const inventory = await runtime.removeCachedModel(modelId);
    renderCacheInventory(inventory);
  } catch (error) {
    cacheSummary.textContent =
      `Could not remove ${label}: ${error instanceof Error ? error.message : String(error)}`;
  } finally {
    appBusy = false;
    updateControlState();
  }
});

void inspectCache();
