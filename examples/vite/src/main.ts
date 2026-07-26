import {
  AceStepWebGpu,
  DEFAULT_INSTRUMENTAL_PROMPT,
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
  !log
) {
  throw new Error("Package smoke-test DOM is incomplete.");
}

prompt.value = DEFAULT_INSTRUMENTAL_PROMPT;

let audioUrl: string | null = null;
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
  }
};

const runtime = new AceStepWebGpu({
  onUpdate: report,
});

generate.addEventListener("click", async () => {
  const promptValue = prompt.value.trim();
  if (!promptValue) {
    prompt.focus();
    return;
  }

  generate.disabled = true;
  cancel.disabled = false;
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
    detail.textContent = error instanceof Error ? error.message : String(error);
    appendLog(detail.textContent);
  } finally {
    generate.disabled = false;
    cancel.disabled = true;
  }
});

cancel.addEventListener("click", () => {
  runtime.cancel();
  cancel.disabled = true;
});
