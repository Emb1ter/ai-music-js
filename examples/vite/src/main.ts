import {
  AceStepWebGpu,
  AceStepWebGpuError,
  DEFAULT_INSTRUMENTAL_PROMPT,
  DEFAULT_VOCAL_PROMPT,
  type CacheInventory,
  type WorkerUpdate,
} from "ai-music-js";
import "./styles.css";

const prompt = document.querySelector<HTMLTextAreaElement>("#prompt");
const mode = document.querySelector<HTMLSelectElement>("#mode");
const lyricsPanel = document.querySelector<HTMLElement>("#lyrics-panel");
const lyrics = document.querySelector<HTMLTextAreaElement>("#lyrics");
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
const cacheList = document.querySelector<HTMLElement>("#cache-list");

if (
  !prompt ||
  !mode ||
  !lyricsPanel ||
  !lyrics ||
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
  !audio ||
  !generate ||
  !cancel ||
  !download ||
  !batchResults ||
  !log ||
  !refreshCache ||
  !clearCache ||
  !cacheSummary ||
  !cacheList
) {
  throw new Error("Package smoke-test DOM is incomplete.");
}

prompt.value = DEFAULT_INSTRUMENTAL_PROMPT;

let audioUrls: string[] = [];
let appBusy = false;
let generating = false;
const downloads = new Map<string, { loaded: number; total: number }>();

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
  const vocalsEnabled = mode.value === "vocals";
  lyricsPanel.hidden = !vocalsEnabled;
  lyrics.disabled = !vocalsEnabled || appBusy;
  vocalLanguage.disabled = !vocalsEnabled || appBusy;
  const sdeOption = sampler.querySelector<HTMLOptionElement>(
    'option[value="euler-sde"]',
  );
  if (sdeOption) {
    sdeOption.disabled = vocalsEnabled;
  }
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
  const vocalsEnabled = mode.value === "vocals";
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
  const quota =
    inventory.quotaBytes === undefined
      ? "quota unavailable"
      : `${formatBytes(inventory.usageBytes ?? 0)} origin usage / ${formatBytes(inventory.quotaBytes)} quota`;
  const persistence =
    inventory.persisted === undefined
      ? "persistence unknown"
      : inventory.persisted
        ? "persistent storage granted"
        : "persistent storage not granted";
  cacheSummary.textContent =
    `${formatBytes(inventory.storedBytes)} of ${formatBytes(inventory.expectedBytes)} model data stored · ${quota} · ${persistence} · ${inventory.origin}`;

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
    appendLog(`${update.stage} finished in ${(update.milliseconds / 1000).toFixed(2)}s`);
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

const runtime = new AceStepWebGpu({
  onUpdate: report,
});

mode.addEventListener("change", applyModeDefaults);
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
  try {
    const inventory = await runtime.listCachedModels();
    renderCacheInventory(inventory);
  } catch (error) {
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
  updateControlState();
  clearAudioResults();
  downloads.clear();
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
      lyrics: lyricsValue,
      vocalLanguage: vocalLanguage.value,
      durationSeconds: Number(duration.value),
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
      const audioUrl = URL.createObjectURL(result.wav);
      audioUrls.push(audioUrl);
      audio.src = audioUrl;
      audio.hidden = false;
      download.href = audioUrl;
      download.download =
        `ai-music-${result.seed}-${result.durationSeconds}s-${result.sampler}.wav`;
      download.hidden = false;
    } else {
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
      "Remove every ai-music-js model file stored by this site? The next generation will download about 5.25 GB again.",
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
