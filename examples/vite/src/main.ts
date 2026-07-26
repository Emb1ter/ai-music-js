import {
  AceStepWebGpu,
  AceStepWebGpuError,
  DEFAULT_INSTRUMENTAL_PROMPT,
  type CacheInventory,
  type WorkerUpdate,
} from "ai-music-js";
import "./styles.css";

const prompt = document.querySelector<HTMLTextAreaElement>("#prompt");
const seed = document.querySelector<HTMLInputElement>("#seed");
const duration = document.querySelector<HTMLSelectElement>("#duration");
const stage = document.querySelector<HTMLElement>("#stage");
const detail = document.querySelector<HTMLElement>("#detail");
const progress = document.querySelector<HTMLProgressElement>("#progress");
const progressLabel = document.querySelector<HTMLElement>("#progress-label");
const audio = document.querySelector<HTMLAudioElement>("#audio");
const generate = document.querySelector<HTMLButtonElement>("#generate");
const cancel = document.querySelector<HTMLButtonElement>("#cancel");
const download = document.querySelector<HTMLAnchorElement>("#download");
const log = document.querySelector<HTMLPreElement>("#log");
const refreshCache =
  document.querySelector<HTMLButtonElement>("#refresh-cache");
const clearCache = document.querySelector<HTMLButtonElement>("#clear-cache");
const cacheSummary = document.querySelector<HTMLElement>("#cache-summary");
const cacheList = document.querySelector<HTMLElement>("#cache-list");

if (
  !prompt ||
  !seed ||
  !duration ||
  !stage ||
  !detail ||
  !progress ||
  !progressLabel ||
  !audio ||
  !generate ||
  !cancel ||
  !download ||
  !log ||
  !refreshCache ||
  !clearCache ||
  !cacheSummary ||
  !cacheList
) {
  throw new Error("Package smoke-test DOM is incomplete.");
}

prompt.value = DEFAULT_INSTRUMENTAL_PROMPT;

let audioUrl: string | null = null;
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

const updateControlState = () => {
  generate.disabled = appBusy;
  cancel.disabled = !generating;
  refreshCache.disabled = appBusy;
  clearCache.disabled = appBusy;
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

  appBusy = true;
  generating = true;
  updateControlState();
  download.hidden = true;
  audio.removeAttribute("src");
  downloads.clear();
  progress.removeAttribute("value");
  progressLabel.textContent = "Preparing runtime";
  stage.textContent = "starting";
  detail.textContent = "Checking WebGPU and browser storage.";

  try {
    const result = await runtime.generate({
      prompt: promptValue,
      seed: Number(seed.value),
      durationSeconds: Number(duration.value),
    });

    if (audioUrl) URL.revokeObjectURL(audioUrl);
    audioUrl = URL.createObjectURL(result.wav);
    audio.src = audioUrl;
    download.href = audioUrl;
    download.download = `ai-music-${seed.value}-${duration.value}s.wav`;
    download.hidden = false;
    progress.max = 1;
    progress.value = 1;
    progressLabel.textContent = "Complete";
    stage.textContent = "music ready";
    detail.textContent =
      `${result.durationSeconds}s stereo WAV · estimated peak ${formatBytes(result.estimatedPeakBytes)}`;
    appendLog("Generation completed. Listen to the entire result.");
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
