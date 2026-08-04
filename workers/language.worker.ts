/// <reference lib="webworker" />

import {
  Qwen3ForCausalLM,
  Qwen3_5ForCausalLM,
  AutoTokenizer,
  LogitsProcessor,
  LogitsProcessorList,
  Tensor,
  env,
  random,
} from "@huggingface/transformers";
import * as ort from "onnxruntime-web/webgpu";
import {
  HIGH_QUALITY_PLANNER_MODEL_ASSETS,
  LANGUAGE_CACHE_NAME,
  PLANNER_MODEL_ASSETS,
  PLANNER_MODEL_DOWNLOAD_BYTES,
} from "../lib/language-model-manifest";
import {
  CACHE_NAME,
  TOTAL_DOWNLOAD_BYTES,
  assetsForAudioQuality,
  type DownloadAsset,
} from "../lib/model-manifest";
import {
  decidePlannerAssetCache,
  missingAssetBytes,
} from "../lib/planner-cache-policy";
import { PlannerQ8WebGpuHead } from "../lib/planner-webgpu-head";
import {
  FP32_AUDIO_CODE_COUNT,
  FP32_AUDIO_CODE_TOKEN_START,
  FP32_PLANNER_HIDDEN_SIZE,
  FP32_PLANNER_ROW_BYTES,
  Fp32PlannerAudioCodeHead,
  Fp32PlannerEmbeddingTable,
  type Fp32PlannerEmbeddingRowStore,
  type Fp32PlannerTimingEvent,
} from "../lib/planner-fp32-webgpu";
import {
  PlannerProfiler,
  type PlannerInputFingerprint,
  type PlannerProfileTimingId,
} from "../lib/planner-profile";
import {
  LYRICS_SYSTEM_PROMPT,
  buildLyricsRepairPrompt,
  buildTimedLyricsPrompt,
  cleanLyrics,
  compactLyrics,
  lyricQualityIssues,
} from "../lib/lyrics";
import {
  AUDIO_CODE_TOKEN_END,
  AUDIO_CODE_TOKEN_START,
  ACE_METADATA_BPM_MAX,
  ACE_METADATA_BPM_MIN,
  ACE_METADATA_DURATION_MAX,
  ACE_METADATA_KEYSCALES,
  ACE_METADATA_TEMPERATURE,
  ACE_METADATA_TIME_SIGNATURES,
  ACE_METADATA_TOP_P,
  DEFAULT_PLANNER_CFG_SCALE,
  DEFAULT_PLANNER_MODEL,
  DEFAULT_PLANNER_MODEL_REVISION,
  PLANNER_EOS_TOKEN,
  PLANNER_PAD_TOKEN,
  PLANNER_SYSTEM_PROMPT,
  PLANNER_UNCONDITIONAL_USER_PROMPT,
  analyzePlannerSemanticCodes,
  blendPlannerCfgLogits,
  buildPlannerUserPrompt,
  createPlannerSamplingRandom,
  deterministicPlannerMetadata,
  formatPlannerGeneratedMetadata,
  formatPlannerMetadata,
  plannerTopCodes,
  resolvePlannerDuration,
  semanticCodeCount,
  tokenIdToAudioCode,
} from "../lib/planner";
import {
  HIGH_QUALITY_PLANNER_BODY_BYTES,
  HIGH_QUALITY_PLANNER_BODY_FILES,
  HIGH_QUALITY_PLANNER_EMBEDDING_FILE,
  HIGH_QUALITY_PLANNER_EMBEDDING_ROW_CACHE_PARAMETER,
  HIGH_QUALITY_PLANNER_GRAPH_FILE,
  HIGH_QUALITY_PLANNER_HEAD_FILES,
} from "../lib/planner-quality";
import plannerDiagnosticGraphUrl from "../planner-diagnostics/model_q4f16.onnx?inline";
import type {
  DownloadUpdate,
  ErrorUpdate,
  LyricsCompleteUpdate,
  PlanCompleteUpdate,
  PlannerProfileUpdate,
  PlanMusicRequest,
  StageUpdate,
  TimingUpdate,
  WorkerUpdate,
  LanguageWorkerRequest,
  WriteLyricsRequest,
  WorkerAssetConfig,
} from "../lib/worker-protocol";

declare const self: DedicatedWorkerGlobalScope;

// Keep ai-music-js model data isolated from unrelated Transformers.js models
// that another feature may use on the same origin.
env.cacheKey = LANGUAGE_CACHE_NAME;

const post = (
  update: WorkerUpdate,
  transfer: Transferable[] = [],
) => self.postMessage(update, transfer);

const stage = (name: string, detail: string): StageUpdate => ({
  type: "stage",
  stage: name,
  detail,
  startedAt: performance.now(),
});

const asNumber = (value: unknown) =>
  typeof value === "number" && Number.isFinite(value) ? value : undefined;

const progressCallback =
  (
    group:
      | "lyrics-writer"
      | "music-planner"
      | "music-planner-high-quality",
  ) =>
  (event: unknown) => {
  if (!event || typeof event !== "object") return;
  const progress = event as Record<string, unknown>;
  const file =
    typeof progress.file === "string"
      ? progress.file
      : typeof progress.name === "string"
        ? progress.name
        : group === "lyrics-writer"
          ? "Qwen3.5 model"
          : group === "music-planner-high-quality"
            ? "ACE INT8-weight / FP32-compute 5 Hz planner"
            : "ACE 5 Hz planner";
  const loaded = asNumber(progress.loaded);
  const total = asNumber(progress.total);
  if (progress.status === "progress" && loaded !== undefined && total) {
    const update: DownloadUpdate = {
      type: "download",
      assetId: `${group}:${file}`,
      group,
      label: file,
      loaded,
      total,
      cached: false,
    };
    post(update);
  }
  };

const generateOnce = async (
  model: Awaited<ReturnType<typeof Qwen3_5ForCausalLM.from_pretrained>>,
  tokenizer: Awaited<ReturnType<typeof AutoTokenizer.from_pretrained>>,
  prompt: string,
  seed: number,
  temperature: number,
) => {
  random.seed(seed);
  const inputs = tokenizer.apply_chat_template(
    [
      { role: "system", content: LYRICS_SYSTEM_PROMPT },
      { role: "user", content: prompt },
    ],
    {
      add_generation_prompt: true,
      tokenize: true,
      return_tensor: true,
      return_dict: true,
      tokenizer_kwargs: { enable_thinking: false },
    },
  );
  const inputLength = inputs.input_ids.dims.at(-1) ?? 0;
  const output = (await model.generate({
    ...inputs,
    max_new_tokens: 500,
    do_sample: true,
    temperature,
    top_k: 20,
    repetition_penalty: 1.15,
  })) as unknown as {
    data: BigInt64Array | BigUint64Array | Int32Array;
    dispose: () => void;
  };
  const generatedIds: number[] = [];
  let tokenIndex = 0;
  for (const token of output.data) {
    if (tokenIndex >= inputLength) generatedIds.push(Number(token));
    tokenIndex += 1;
  }
  const text = tokenizer.decode(generatedIds, {
    skip_special_tokens: true,
  });
  output.dispose();
  for (const value of Object.values(inputs)) {
    if (
      value &&
      typeof value === "object" &&
      "dispose" in value &&
      typeof value.dispose === "function"
    ) {
      value.dispose();
    }
  }
  return cleanLyrics(text);
};

const writeLyrics = async (request: WriteLyricsRequest) => {
  if (!("gpu" in self.navigator) || !self.navigator.gpu) {
    throw new Error(
      "WebGPU is unavailable in this Worker. Use desktop Chromium with WebGPU enabled.",
    );
  }
  const onnxEnvironment = env.backends.onnx;
  if (onnxEnvironment.wasm) {
    onnxEnvironment.wasm.numThreads = 1;
    onnxEnvironment.wasm.proxy = false;
    onnxEnvironment.wasm.wasmPaths = {
      mjs: new URL(
        "./wasm/ort-wasm-simd-threaded.asyncify.mjs",
        self.location.href,
      ).href,
      wasm: new URL(
        "./wasm/ort-wasm-simd-threaded.asyncify.wasm",
        self.location.href,
      ).href,
    };
  }
  if (onnxEnvironment.webgpu) {
    onnxEnvironment.webgpu.powerPreference = "high-performance";
  }

  const timings: Record<string, number> = {};
  post(
    stage(
      "lyrics-model",
      `Loading ${request.modelId} with Transformers.js and WebGPU.`,
    ),
  );
  const loadStart = performance.now();
  const [tokenizer, model] = await Promise.all([
    AutoTokenizer.from_pretrained(request.modelId, {
      revision: request.revision,
      progress_callback: progressCallback("lyrics-writer"),
    }),
    Qwen3_5ForCausalLM.from_pretrained(request.modelId, {
      revision: request.revision,
      device: "webgpu",
      dtype: "q4",
      progress_callback: progressCallback("lyrics-writer"),
    }),
  ]);
  timings["lyrics-model-load"] = performance.now() - loadStart;
  post({
    type: "timing",
    stage: "lyrics-model-load",
    milliseconds: timings["lyrics-model-load"],
  } satisfies TimingUpdate);

  try {
    post(
      stage(
        "lyrics-writing",
        `Writing up to ${request.maxWords} words for ${request.durationSeconds} seconds.`,
      ),
    );
    const generationStart = performance.now();
    let lyrics = compactLyrics(
      await generateOnce(
        model,
        tokenizer,
        buildTimedLyricsPrompt(
          request.prompt,
          request.durationSeconds,
          request.maxWords,
        ),
        request.seed,
        0.55,
      ),
      request.maxWords,
    );
    let attempts = 1;
    let issues = lyricQualityIssues(lyrics, request.maxWords);
    if (issues.length) {
      post(
        stage(
          "lyrics-repair",
          `Qwen is repairing its draft: ${issues.join("; ")}.`,
        ),
      );
      lyrics = compactLyrics(
        await generateOnce(
          model,
          tokenizer,
          buildLyricsRepairPrompt(
            request.prompt,
            lyrics,
            issues,
            request.durationSeconds,
            request.maxWords,
          ),
          (request.seed + 1) % 2_147_483_648,
          0.35,
        ),
        request.maxWords,
      );
      attempts = 2;
      issues = lyricQualityIssues(lyrics, request.maxWords);
    }
    if (!lyrics) {
      throw new Error("Qwen returned empty lyrics.");
    }
    if (issues.length) {
      throw new Error(
        `Qwen could not produce usable lyrics: ${issues.join("; ")}.`,
      );
    }
    timings["lyrics-generation"] = performance.now() - generationStart;
    post({
      type: "timing",
      stage: "lyrics-generation",
      milliseconds: timings["lyrics-generation"],
    } satisfies TimingUpdate);
    const complete: LyricsCompleteUpdate = {
      type: "lyrics-complete",
      lyrics,
      model: request.modelId,
      revision: request.revision,
      seed: request.seed,
      durationSeconds: request.durationSeconds,
      maxWords: request.maxWords,
      attempts,
      timings,
    };
    post(complete);
  } finally {
    await model.dispose();
  }
};

const disposeInputs = (inputs: Record<string, unknown>) => {
  for (const value of Object.values(inputs)) {
    if (
      value &&
      typeof value === "object" &&
      "dispose" in value &&
      typeof value.dispose === "function"
    ) {
      value.dispose();
    }
  }
};

const generatedIds = (
  output: {
    data: BigInt64Array | BigUint64Array | Int32Array;
    dims: readonly number[];
  },
  promptLength: number,
) => {
  const sequenceLength = output.dims.at(-1) ?? output.data.length;
  return Array.from(output.data.slice(0, sequenceLength), Number).slice(
    promptLength,
  );
};

const PLANNER_DIAGNOSTIC_GRAPH =
  "nonfinite-node-probes-final-block-fp32-v5";

const installPlannerDiagnosticGraph = async (
  modelId: string,
  revision: string,
  dtype: "fp16" | "q4f16" | "q8",
) => {
  if (dtype !== "q4f16") return;
  if (
    modelId !== DEFAULT_PLANNER_MODEL ||
    revision !== DEFAULT_PLANNER_MODEL_REVISION
  ) {
    return;
  }
  if (!("caches" in self)) {
    throw new Error(
      "The instrumented ACE planner requires the browser Cache API.",
    );
  }

  const graphUrl =
    `https://huggingface.co/${modelId}/resolve/` +
    `${encodeURIComponent(revision)}/onnx/model_q4f16.onnx`;
  const cache = await caches.open(LANGUAGE_CACHE_NAME);
  const current = await cache.match(graphUrl);
  if (
    current?.headers.get("x-ai-music-js-planner-diagnostic") ===
    PLANNER_DIAGNOSTIC_GRAPH
  ) {
    post({
      type: "diagnostic",
      key: "planner graph diagnostic",
      value: `${PLANNER_DIAGNOSTIC_GRAPH} · cache hit`,
    });
    return;
  }

  const bundled = await fetch(plannerDiagnosticGraphUrl);
  if (!bundled.ok) {
    throw new Error(
      `Bundled planner diagnostic graph returned HTTP ${bundled.status}.`,
    );
  }
  const graph = await bundled.arrayBuffer();
  await cache.put(
    graphUrl,
    new Response(graph, {
      headers: {
        "content-length": String(graph.byteLength),
        "content-type": "application/octet-stream",
        "x-ai-music-js-planner-diagnostic": PLANNER_DIAGNOSTIC_GRAPH,
      },
    }),
  );
  post({
    type: "diagnostic",
    key: "planner graph diagnostic",
    value:
      `${PLANNER_DIAGNOSTIC_GRAPH} · installed ${graph.byteLength} byte ` +
      "graph with compact per-node finite-value probes",
  });
};

type PlannerForwardInputs = Record<string, unknown>;
type PlannerForwardOutputs = Record<string, Tensor>;
type MutablePlannerModel = {
  forward: (
    inputs: PlannerForwardInputs,
  ) => Promise<PlannerForwardOutputs>;
};

const installPlannerWebGpuHead = (
  model: unknown,
  head: PlannerQ8WebGpuHead,
) => {
  const mutableModel = model as MutablePlannerModel;
  const bodyForward = mutableModel.forward.bind(mutableModel);
  mutableModel.forward = async (inputs) => {
    const inputIds = inputs.input_ids;
    if (!(inputIds instanceof Tensor)) {
      throw new Error("Planner body forward is missing input_ids.");
    }
    const inputsEmbeds = await head.embed(inputIds);
    let outputs: PlannerForwardOutputs;
    try {
      outputs = await bodyForward({
        ...inputs,
        inputs_embeds: inputsEmbeds,
      });
    } finally {
      inputsEmbeds.dispose();
    }
    const hidden = outputs.last_hidden_state;
    if (!(hidden instanceof Tensor)) {
      disposePlannerOutputs(outputs);
      throw new Error(
        "Planner body did not return its last_hidden_state output.",
      );
    }
    try {
      outputs.logits = await head.forward(hidden);
      return outputs;
    } catch (error) {
      for (const tensor of Object.values(outputs)) {
        if (tensor !== hidden) tensor.dispose();
      }
      throw error;
    } finally {
      hidden.dispose();
      delete outputs.last_hidden_state;
    }
  };
};

const disposePlannerOutputs = (outputs: PlannerForwardOutputs) => {
  for (const tensor of Object.values(outputs)) tensor.dispose();
};

const lastInputToken = (inputs: PlannerForwardInputs) => {
  const inputIds = inputs.input_ids;
  if (!(inputIds instanceof Tensor) || inputIds.data.length === 0) {
    return "unknown";
  }
  return String(inputIds.data[inputIds.data.length - 1]);
};

const contextLength = (inputs: PlannerForwardInputs) => {
  const attentionMask = inputs.attention_mask;
  if (!(attentionMask instanceof Tensor)) return "unknown";
  return String(attentionMask.dims.at(-1) ?? "unknown");
};

const installPlannerForwardDiagnostics = (model: unknown) => {
  const mutableModel = model as MutablePlannerModel;
  const originalForward = mutableModel.forward.bind(mutableModel);
  let enabled = false;
  let cachedForwardCount = 0;

  mutableModel.forward = async (inputs) => {
    const outputs = await originalForward(inputs);
    const inputIds = inputs.input_ids;
    const isCachedTokenForward =
      inputIds instanceof Tensor && inputIds.dims.at(-1) === 1;
    if (enabled && isCachedTokenForward) cachedForwardCount += 1;
    let firstFailure:
      | { name: string; nonFiniteCount: number }
      | undefined;

    for (const [name, tensor] of Object.entries(outputs)) {
      if (!name.startsWith("planner_diag.")) continue;
      const nonFiniteCount = Number(tensor.data[0] ?? 0);
      tensor.dispose();
      delete outputs[name];
      if (
        enabled &&
        isCachedTokenForward &&
        !firstFailure &&
        nonFiniteCount > 0
      ) {
        firstFailure = { name, nonFiniteCount };
      }
    }

    if (firstFailure) {
      disposePlannerOutputs(outputs);
      throw new Error(
        "Planner WebGPU first non-finite tensor: " +
          `${firstFailure.name} (${firstFailure.nonFiniteCount} values) ` +
          `during cached semantic forward ${cachedForwardCount}; input token ` +
          `${lastInputToken(inputs)}, context length ${contextLength(inputs)}.`,
      );
    }
    return outputs;
  };

  return {
    enable() {
      enabled = true;
      cachedForwardCount = 0;
    },
  };
};

const sampleTopPToken = (
  values: Float32Array,
  tokenOffset: number,
  temperature: number,
  topP: number,
  nextRandom: () => number = () => random.random(),
) => {
  let maximum = Number.NEGATIVE_INFINITY;
  for (const value of values) {
    if (Number.isFinite(value) && value > maximum) maximum = value;
  }
  if (!Number.isFinite(maximum)) {
    throw new Error("Planner produced no finite logits to sample.");
  }

  // Match ACE-Step's PyTorch path exactly: nucleus membership is calculated
  // from the untempered logits first, then temperature is applied only to the
  // final multinomial distribution over the retained tokens.
  const nucleusWeights = new Float64Array(values.length);
  const tokenIds = Array.from(
    { length: values.length },
    (_, index) => index,
  );
  let nucleusTotal = 0;
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index]!;
    const weight = Number.isFinite(value)
      ? Math.exp(value - maximum)
      : 0;
    nucleusWeights[index] = weight;
    nucleusTotal += weight;
  }
  if (!(nucleusTotal > 0) || !Number.isFinite(nucleusTotal)) {
    throw new Error("Planner sampling distribution is invalid.");
  }

  tokenIds.sort(
    (left, right) =>
      nucleusWeights[right]! - nucleusWeights[left]!,
  );
  let keptWeight = 0;
  let keptCount = 0;
  for (const tokenId of tokenIds) {
    keptWeight += nucleusWeights[tokenId]!;
    keptCount += 1;
    if (keptWeight / nucleusTotal >= topP) break;
  }

  const sampleWeights = new Float64Array(keptCount);
  let sampleTotal = 0;
  for (let index = 0; index < keptCount; index += 1) {
    const tokenId = tokenIds[index]!;
    const weight = Math.exp(
      (values[tokenId]! - maximum) / temperature,
    );
    sampleWeights[index] = weight;
    sampleTotal += weight;
  }
  if (!(sampleTotal > 0) || !Number.isFinite(sampleTotal)) {
    throw new Error("Planner tempered sampling distribution is invalid.");
  }
  let draw = nextRandom() * sampleTotal;
  for (let index = 0; index < keptCount; index += 1) {
    const tokenId = tokenIds[index]!;
    draw -= sampleWeights[index]!;
    if (draw <= 0 || index === keptCount - 1) {
      return tokenOffset + tokenId;
    }
  }
  throw new Error("Planner nucleus sampling failed to select a token.");
};

class AudioCodeSamplerProcessor extends LogitsProcessor {
  private readonly guidedAudioLogits = new Float32Array(
    AUDIO_CODE_TOKEN_END - AUDIO_CODE_TOKEN_START,
  );

  constructor(
    private readonly promptLength: number,
    private readonly targetCodes: number,
    private readonly cfgScale = DEFAULT_PLANNER_CFG_SCALE,
    private readonly temperature = 0.85,
    private readonly topP = 0.9,
  ) {
    super();
  }

  override _call(inputIds: bigint[][], logits: Tensor) {
    if (inputIds.length !== 2 || logits.dims[0] !== 2) {
      throw new Error(
        "ACE planner CFG expects conditioned and unconditional sequences.",
      );
    }
    const data = logits.data as Float32Array;
    const rowSize = logits.dims.at(-1) ?? 0;
    if (rowSize < AUDIO_CODE_TOKEN_END) {
      throw new Error(
        `ACE planner vocabulary is ${rowSize}; expected at least ${AUDIO_CODE_TOKEN_END}.`,
      );
    }
    const generatedCount = inputIds[0].length - this.promptLength;
    let selectedToken: number;
    if (generatedCount >= this.targetCodes) {
      selectedToken = PLANNER_EOS_TOKEN;
    } else {
      const conditionalAudioLogits = data.subarray(
        AUDIO_CODE_TOKEN_START,
        AUDIO_CODE_TOKEN_END,
      );
      const unconditionalAudioLogits = data.subarray(
        rowSize + AUDIO_CODE_TOKEN_START,
        rowSize + AUDIO_CODE_TOKEN_END,
      );
      const blend = blendPlannerCfgLogits(
        conditionalAudioLogits,
        unconditionalAudioLogits,
        this.cfgScale,
        this.guidedAudioLogits,
      );
      if (blend.mode !== "cfg") {
        const previousToken = inputIds[0]?.at(-1);
        throw new Error(
          `Planner WebGPU CFG logits became non-finite at semantic step ${generatedCount + 1}/${this.targetCodes} ` +
            `(conditioned row ${blend.conditionalFinite}/64000 finite, ` +
            `unconditional row ${blend.unconditionalFinite}/64000 finite, ` +
            `previous token ${previousToken === undefined ? "unknown" : String(previousToken)}, ` +
            `tensor ${logits.type} ${JSON.stringify(logits.dims)}, data ${data.constructor.name}).`,
        );
      }
      selectedToken = sampleTopPToken(
        this.guidedAudioLogits,
        AUDIO_CODE_TOKEN_START,
        this.temperature,
        this.topP,
      );
    }
    data.fill(Number.NEGATIVE_INFINITY);
    data[selectedToken] = 0;
    data[rowSize + selectedToken] = 0;
    return logits;
  }
}

const ensurePlannerChatTemplate = async (
  tokenizer: Awaited<ReturnType<typeof AutoTokenizer.from_pretrained>>,
  modelId: string,
  revision: string,
) => {
  if (tokenizer.chat_template) return;

  const templateUrl =
    `https://huggingface.co/${modelId}/resolve/` +
    `${encodeURIComponent(revision)}/chat_template.jinja`;
  const cache =
    "caches" in self ? await caches.open(LANGUAGE_CACHE_NAME) : null;
  let response = await cache?.match(templateUrl);
  if (!response) {
    response = await fetch(templateUrl, {
      mode: "cors",
      credentials: "omit",
    });
    if (!response.ok) {
      throw new Error(
        `Planner tokenizer has no embedded chat template and ${templateUrl} returned HTTP ${response.status}.`,
      );
    }
    await cache?.put(templateUrl, response.clone()).catch(() => undefined);
  }
  const template = await response.text();
  if (!template.trim()) {
    throw new Error("Planner chat_template.jinja is empty.");
  }
  tokenizer.chat_template = template;
};

const pruneSupersededPlannerCache = async (
  modelId: string,
  revision: string,
) => {
  if (!("caches" in self)) return;
  const cache = await caches.open(LANGUAGE_CACHE_NAME);
  const repositoryPrefix =
    `https://huggingface.co/${modelId}/resolve/`;
  const currentPrefix =
    `${repositoryPrefix}${encodeURIComponent(revision)}/`;
  let removedEntries = 0;
  let removedBytes = 0;
  for (const request of await cache.keys()) {
    if (
      !request.url.startsWith(repositoryPrefix) ||
      request.url.startsWith(currentPrefix)
    ) {
      continue;
    }
    const response = await cache.match(request);
    const contentLength = Number(
      response?.headers.get("content-length"),
    );
    if (Number.isFinite(contentLength) && contentLength > 0) {
      removedBytes += contentLength;
    }
    if (await cache.delete(request)) removedEntries += 1;
  }
  if (removedEntries > 0) {
    post({
      type: "diagnostic",
      key: "superseded planner cache removed",
      value:
        `${removedEntries} files · ` +
        `${(removedBytes / 1e9).toFixed(2)} GB`,
    });
  }
};

const configurePlannerBrowserCache = async () => {
  const estimate = await self.navigator.storage?.estimate?.();
  const quota = Number(estimate?.quota);
  const usage = Number(estimate?.usage);
  if (
    !Number.isFinite(quota) ||
    !Number.isFinite(usage) ||
    quota <= usage
  ) {
    return;
  }
  const available = quota - usage;
  const largestPlannerShard = PLANNER_MODEL_ASSETS.reduce(
    (largest, asset) =>
      asset.role === "weights"
        ? Math.max(largest, asset.bytes)
        : largest,
    0,
  );
  // Cache the 3.63 GB split planner only when the origin can also retain the audio
  // pipeline and one full-shard Cache.put transaction. Otherwise stream the
  // planner for this Worker lifetime, release it before audio inference, and
  // avoid turning a successful model download into an origin-quota failure.
  const safeCacheBytes =
    PLANNER_MODEL_DOWNLOAD_BYTES +
    TOTAL_DOWNLOAD_BYTES +
    largestPlannerShard;
  if (available < safeCacheBytes) {
    env.useBrowserCache = false;
    post({
      type: "diagnostic",
      key: "planner cache policy",
      value:
        `streaming only · ${(available / 1e9).toFixed(2)} GB available, ` +
        `${(safeCacheBytes / 1e9).toFixed(2)} GB needed for planner + ` +
        "audio reserve + cache-write headroom",
    });
    return;
  }
  env.useBrowserCache = true;
  post({
    type: "diagnostic",
    key: "planner cache policy",
    value:
      `persistent Cache API · ${(available / 1e9).toFixed(2)} GB available`,
  });
};

type Fp32CacheTensors = Record<string, ort.Tensor>;

const fp32PlannerAssetUrl = (
  modelId: string,
  revision: string,
  fileName: string,
) =>
  `https://huggingface.co/${modelId}/resolve/` +
  `${encodeURIComponent(revision)}/${fileName}`;

const allocateLargeByteBuffer = (byteLength: number) => {
  try {
    return new Uint8Array(new ArrayBuffer(byteLength));
  } catch (error) {
    if (!(error instanceof RangeError)) throw error;
    const pages = Math.ceil(byteLength / 65_536);
    return new Uint8Array(
      new WebAssembly.Memory({
        initial: pages,
        maximum: pages,
      }).buffer,
      0,
      byteLength,
    );
  }
};

type PlannerStorageManager = StorageManager & {
  getDirectory?: () => Promise<FileSystemDirectoryHandle>;
};

const plannerAssetDirectoryUrl = (value: string) =>
  value.endsWith("/") ? value : `${value}/`;

const resolvedAudioAssetUrl = (
  asset: DownloadAsset,
  config?: WorkerAssetConfig,
) => {
  const baseUrl =
    config?.allAssetsBaseUrl ??
    (asset.url.startsWith("/models/")
      ? config?.modelBaseUrl
      : undefined);
  if (!baseUrl) {
    return new URL(asset.url, self.location.origin).href;
  }
  const configured = new URL(
    asset.fileName,
    new URL(plannerAssetDirectoryUrl(baseUrl), self.location.href),
  );
  const pinned = new URL(asset.url, self.location.origin);
  configured.search = pinned.search;
  return configured.href;
};

const existingAudioAssetState = async (
  asset: DownloadAsset,
  directory: FileSystemDirectoryHandle | null,
  cache: Cache | null,
  config?: WorkerAssetConfig,
) => {
  if (directory) {
    try {
      const handle = await directory.getFileHandle(asset.fileName);
      const file = await handle.getFile();
      if (file.size === asset.bytes) {
        return { bytes: asset.bytes, cached: true };
      }
    } catch (error) {
      if (!(error instanceof DOMException && error.name === "NotFoundError")) {
        throw error;
      }
    }
  }
  const response = await cache?.match(
    resolvedAudioAssetUrl(asset, config),
  );
  if (!response) {
    return { bytes: asset.bytes, cached: false };
  }
  const contentLengthHeader = response.headers.get("content-length");
  const contentLength =
    contentLengthHeader === null
      ? undefined
      : Number(contentLengthHeader);
  return {
    bytes: asset.bytes,
    cached:
      contentLength === undefined ||
      (Number.isFinite(contentLength) &&
        contentLength === asset.bytes),
  };
};

const inspectMissingAudioBytes = async (
  request: PlanMusicRequest,
) => {
  const storage = self.navigator.storage as
    | PlannerStorageManager
    | undefined;
  let directory: FileSystemDirectoryHandle | null = null;
  if (storage?.getDirectory) {
    try {
      const root = await storage.getDirectory();
      directory = await root.getDirectoryHandle(CACHE_NAME);
    } catch (error) {
      if (!(error instanceof DOMException && error.name === "NotFoundError")) {
        throw error;
      }
    }
  }
  const cache =
    "caches" in self ? await caches.open(CACHE_NAME) : null;
  const assets = assetsForAudioQuality(request.audioQuality);
  const states = [];
  for (const asset of assets) {
    states.push(
      await existingAudioAssetState(
        asset,
        directory,
        cache,
        request.assets,
      ),
    );
  }
  return {
    expectedBytes: assets.reduce(
      (total, asset) => total + asset.bytes,
      0,
    ),
    missingBytes: missingAssetBytes(states),
  };
};

const createHighQualityCachePolicy = async (
  request: PlanMusicRequest,
) => {
  const audio = await inspectMissingAudioBytes(request);
  const initialEstimate =
    await self.navigator.storage?.estimate?.();
  const initialQuota = Number(initialEstimate?.quota);
  const initialUsage = Number(initialEstimate?.usage);
  const initialAvailable =
    Number.isFinite(initialQuota) &&
    Number.isFinite(initialUsage) &&
    initialQuota > initialUsage
      ? initialQuota - initialUsage
      : undefined;
  let persistedThisRunBytes = 0;

  post({
    type: "diagnostic",
    key: "high-quality planner cache policy",
    value:
      `per-file Cache API · ${(audio.missingBytes / 1e9).toFixed(2)} GB ` +
      `of ${(audio.expectedBytes / 1e9).toFixed(2)} GB selected audio assets still missing` +
      (initialAvailable === undefined
        ? " · storage availability unavailable"
        : ` · ${(initialAvailable / 1e9).toFixed(2)} GB available`),
  });

  return {
    async decide(fileName: string, assetBytes: number) {
      const estimate = await self.navigator.storage?.estimate?.();
      const quota = Number(estimate?.quota);
      const usage = Number(estimate?.usage);
      const liveAvailable =
        Number.isFinite(quota) &&
        Number.isFinite(usage) &&
        quota > usage
          ? quota - usage
          : undefined;
      const trackedAvailable =
        initialAvailable === undefined
          ? undefined
          : Math.max(
              0,
              initialAvailable - persistedThisRunBytes,
            );
      const availableBytes =
        liveAvailable === undefined
          ? trackedAvailable
          : trackedAvailable === undefined
            ? liveAvailable
            : Math.min(liveAvailable, trackedAvailable);
      if (availableBytes === undefined) {
        return {
          cache: false,
          detail: `${fileName}: browser storage availability is unavailable`,
        };
      }
      const decision = decidePlannerAssetCache({
        availableBytes,
        missingAudioBytes: audio.missingBytes,
        assetBytes,
      });
      return {
        cache: decision.cache,
        detail:
          `${fileName}: ${(availableBytes / 1e9).toFixed(2)} GB available, ` +
          `${(audio.missingBytes / 1e9).toFixed(2)} GB reserved for missing audio, ` +
          `${(decision.writeHeadroomBytes / 1e9).toFixed(2)} GB write headroom`,
      };
    },
    recordPersisted(assetBytes: number) {
      persistedThisRunBytes += assetBytes;
    },
  };
};

const createFp32PlannerAssetLoader = async (
  request: PlanMusicRequest,
) => {
  const policy = await createHighQualityCachePolicy(request);
  const cache =
    "caches" in self ? await caches.open(LANGUAGE_CACHE_NAME) : null;
  let reportedCacheFailure = false;

  return async (fileName: string, expectedBytes: number) => {
    const url = fp32PlannerAssetUrl(
      request.highQualityModelId,
      request.highQualityRevision,
      fileName,
    );
    let response = await cache?.match(url);
    const cached = Boolean(response);
    if (!response) {
      response = await fetch(url, {
        cache: "no-store",
        mode: "cors",
        credentials: "omit",
      });
    }
    if (!response.ok) {
      throw new Error(
        `High-quality planner asset ${fileName} returned HTTP ${response.status}.`,
      );
    }
    const declaredBytes = Number(response.headers.get("content-length"));
    if (
      Number.isFinite(declaredBytes) &&
      declaredBytes > 0 &&
      declaredBytes !== expectedBytes
    ) {
      throw new Error(
        `High-quality planner asset ${fileName} is ${declaredBytes} bytes; expected ${expectedBytes}.`,
      );
    }

    let cacheWrite: Promise<void> | undefined;
    if (!cached && cache) {
      const decision = await policy.decide(fileName, expectedBytes);
      if (decision.cache) {
        cacheWrite = cache
          .put(url, response.clone())
          .then(() => {
            policy.recordPersisted(expectedBytes);
          })
          .catch((error) => {
            if (!reportedCacheFailure) {
              reportedCacheFailure = true;
              post({
                type: "diagnostic",
                key: "high-quality planner cache write",
                value:
                  `continuing with per-file cache · ` +
                  `${error instanceof Error ? error.message : String(error)}`,
              });
            }
          });
      } else {
        post({
          type: "diagnostic",
          key: "planner shard streamed",
          value: decision.detail,
        });
      }
    }

    const bytes = allocateLargeByteBuffer(expectedBytes);
    if (response.body) {
      const reader = response.body.getReader();
      let loaded = 0;
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        if (!value?.byteLength) continue;
        if (loaded + value.byteLength > expectedBytes) {
          throw new Error(
            `High-quality planner asset ${fileName} exceeded its manifest size.`,
          );
        }
        bytes.set(value, loaded);
        loaded += value.byteLength;
        post({
          type: "download",
          assetId: `music-planner-high-quality:${fileName}`,
          group: "music-planner-high-quality",
          label: fileName,
          loaded,
          total: expectedBytes,
          cached,
        });
      }
      if (loaded !== expectedBytes) {
        throw new Error(
          `High-quality planner asset ${fileName} downloaded ${loaded} bytes; expected ${expectedBytes}.`,
        );
      }
    } else {
      const payload = new Uint8Array(await response.arrayBuffer());
      if (payload.byteLength !== expectedBytes) {
        throw new Error(
          `High-quality planner asset ${fileName} downloaded ${payload.byteLength} bytes; expected ${expectedBytes}.`,
        );
      }
      bytes.set(payload);
    }
    await cacheWrite;
    return bytes;
  };
};

const fp32EmbeddingRowCacheUrl = (
  weightUrl: string,
  tokenId: number,
) => {
  const url = new URL(weightUrl);
  url.searchParams.set(
    HIGH_QUALITY_PLANNER_EMBEDDING_ROW_CACHE_PARAMETER,
    String(tokenId),
  );
  return url.href;
};

/**
 * Persist only the small FP32 rows actually used by prompts and metadata.
 * This avoids duplicating the complete 2.2 GiB tied-embedding tensor while
 * making repeat generations independent of Hugging Face range latency.
 */
const createFp32PlannerEmbeddingRowStore = async (
  weightUrl: string,
): Promise<Fp32PlannerEmbeddingRowStore | undefined> => {
  if (!("caches" in self)) return undefined;
  const cache = await caches.open(LANGUAGE_CACHE_NAME);
  let reportedFailure = false;
  const reportFailure = (operation: string, error: unknown) => {
    if (reportedFailure) return;
    reportedFailure = true;
    post({
      type: "diagnostic",
      key: "planner embedding row cache",
      value:
        `${operation} failed; continuing without that cached row · ` +
        `${error instanceof Error ? error.message : String(error)}`,
    });
  };
  return {
    async load(tokenId) {
      const key = fp32EmbeddingRowCacheUrl(weightUrl, tokenId);
      try {
        const response = await cache.match(key);
        if (!response) return undefined;
        const payload = await response.arrayBuffer();
        if (payload.byteLength !== FP32_PLANNER_ROW_BYTES) {
          await cache.delete(key);
          return undefined;
        }
        return new Float32Array(payload);
      } catch (error) {
        reportFailure("read", error);
        return undefined;
      }
    },
    async save(tokenId, row) {
      const key = fp32EmbeddingRowCacheUrl(weightUrl, tokenId);
      try {
        const payload = new Uint8Array(
          row.buffer,
          row.byteOffset,
          row.byteLength,
        ).slice();
        await cache.put(
          key,
          new Response(payload.buffer, {
            headers: {
              "content-length": String(payload.byteLength),
              "content-type": "application/octet-stream",
              "x-ai-music-js-embedding-token": String(tokenId),
            },
          }),
        );
      } catch (error) {
        reportFailure("write", error);
      }
    },
  };
};

const createFp32EmptyCache = (batchSize: number) => {
  const cache: Fp32CacheTensors = {};
  for (let layer = 0; layer < 36; layer += 1) {
    cache[`past_key_values.${layer}.key`] = new ort.Tensor(
      "float32",
      new Float32Array(0),
      [batchSize, 8, 0, 128],
    );
    cache[`past_key_values.${layer}.value`] = new ort.Tensor(
      "float32",
      new Float32Array(0),
      [batchSize, 8, 0, 128],
    );
  }
  return cache;
};

const disposeFp32Tensors = (values: Iterable<ort.Tensor>) => {
  const disposed = new Set<ort.Tensor>();
  for (const value of values) {
    if (disposed.has(value)) continue;
    disposed.add(value);
    value.dispose();
  }
};

const takeFp32PresentCache = (
  outputs: Record<string, ort.Tensor>,
) => {
  const cache: Fp32CacheTensors = {};
  for (let layer = 0; layer < 36; layer += 1) {
    for (const kind of ["key", "value"] as const) {
      const outputName = `present.${layer}.${kind}`;
      const value = outputs[outputName];
      if (!value) {
        throw new Error(`FP32 planner body did not return ${outputName}.`);
      }
      cache[`past_key_values.${layer}.${kind}`] = value;
    }
  }
  return cache;
};

const fp32Int64Tensor = (
  values: readonly number[],
  dims: readonly number[],
) =>
  new ort.Tensor(
    "int64",
    BigInt64Array.from(values, (value) => BigInt(value)),
    [...dims],
  );

const fp32PositionIds = (
  attentionMask: readonly number[],
  batchSize: number,
  sequenceLength: number,
) => {
  const positions = new Array<number>(attentionMask.length);
  const tokenCounts = new Array<number>(batchSize).fill(0);
  for (let batch = 0; batch < batchSize; batch += 1) {
    let position = 0;
    for (let column = 0; column < sequenceLength; column += 1) {
      const index = batch * sequenceLength + column;
      if (attentionMask[index]) {
        positions[index] = position;
        position += 1;
      } else {
        positions[index] = 0;
      }
    }
    tokenCounts[batch] = position;
  }
  return { positions, tokenCounts };
};

type Fp32BodyTimingEvent =
  | "body-total"
  | "body-feed-prep"
  | "body-session-run"
  | "body-output-cache";

type Fp32BodyTimingRecorder = (
  event: Fp32BodyTimingEvent,
  milliseconds: number,
) => void;

type Fp32MetadataTimingRecorder = (
  event: "sparse-score" | "sampling" | "embedding-prefetch",
  milliseconds: number,
) => void;

const sha256Hex = async (payload: BufferSource) => {
  const digest = await self.crypto.subtle.digest("SHA-256", payload);
  return Array.from(new Uint8Array(digest), (byte) =>
    byte.toString(16).padStart(2, "0"),
  ).join("");
};

const sha256Numbers = (values: readonly number[]) =>
  sha256Hex(Uint32Array.from(values).buffer);

const sha256Text = (value: string) =>
  sha256Hex(new TextEncoder().encode(value));

const runFp32PlannerBody = async (
  session: ort.InferenceSession,
  embeddings: Fp32PlannerEmbeddingTable,
  inputIds: readonly number[],
  attentionMask: readonly number[],
  positionIds: readonly number[],
  inputShape: readonly [number, number],
  positionShape: readonly [number, number],
  cache: Fp32CacheTensors,
  recordTiming: Fp32BodyTimingRecorder = () => undefined,
) => {
  const totalStart = performance.now();
  const inputsEmbeds = await embeddings.embed(inputIds, inputShape);
  const feedStart = performance.now();
  const ids = fp32Int64Tensor(inputIds, inputShape);
  const mask = fp32Int64Tensor(attentionMask, [
    inputShape[0],
    attentionMask.length / inputShape[0],
  ]);
  const positions = fp32Int64Tensor(positionIds, positionShape);
  const feeds: Record<string, ort.Tensor> = {
    input_ids: ids,
    inputs_embeds: inputsEmbeds,
    attention_mask: mask,
    position_ids: positions,
    ...cache,
  };
  recordTiming("body-feed-prep", performance.now() - feedStart);
  let outputs: Record<string, ort.Tensor>;
  try {
    const runStart = performance.now();
    outputs = await session.run(feeds);
    recordTiming("body-session-run", performance.now() - runStart);
  } finally {
    ids.dispose();
    inputsEmbeds.dispose();
    mask.dispose();
    positions.dispose();
    disposeFp32Tensors(Object.values(cache));
  }
  const hidden = outputs.last_hidden_state;
  if (!hidden) {
    disposeFp32Tensors(Object.values(outputs));
    throw new Error("FP32 planner body returned no last_hidden_state.");
  }
  const outputStart = performance.now();
  const nextCache = takeFp32PresentCache(outputs);
  recordTiming("body-output-cache", performance.now() - outputStart);
  recordTiming("body-total", performance.now() - totalStart);
  return {
    hidden,
    cache: nextCache,
  };
};

type PlannerTokenizer = Awaited<
  ReturnType<typeof AutoTokenizer.from_pretrained>
>;

type MetadataTokenTrie = {
  children: Map<number, MetadataTokenTrie>;
  value?: string;
};

const plannerMetadataDurationCandidates = (
  request: PlanMusicRequest,
) =>
  request.autoDuration
    ? Array.from(
        {
          length:
            ACE_METADATA_DURATION_MAX -
            request.recommendedDurationSeconds +
            1,
        },
        (_, index) => request.recommendedDurationSeconds + index,
      )
    : [request.durationSeconds];

const plannerTextTokenIds = (
  tokenizer: PlannerTokenizer,
  text: string,
) => {
  const encoded = tokenizer(text, {
    add_special_tokens: false,
    return_tensor: true,
  }) as Record<string, Tensor>;
  const inputIds = encoded.input_ids;
  if (!inputIds) {
    disposeInputs(encoded);
    throw new Error("ACE metadata tokenizer returned no input IDs.");
  }
  const values = Array.from(inputIds.data, Number);
  disposeInputs(encoded);
  return values;
};

const plannerMetadataPrefetchTokenIds = (
  tokenizer: PlannerTokenizer,
  promptIds: readonly number[],
  request: PlanMusicRequest,
) => {
  const ids = new Set<number>(promptIds);
  const addText = (text: string) => {
    for (const tokenId of plannerTextTokenIds(tokenizer, text)) {
      ids.add(tokenId);
    }
  };
  for (const forcedText of [
    "<think>\nbpm:",
    "duration:",
    "keyscale:",
    "timesignature:",
    "</think>",
  ]) {
    addText(forcedText);
  }
  const candidateGroups: readonly (readonly (string | number)[])[] = [
    Array.from(
      { length: ACE_METADATA_BPM_MAX - ACE_METADATA_BPM_MIN + 1 },
      (_, index) => ACE_METADATA_BPM_MIN + index,
    ),
    plannerMetadataDurationCandidates(request),
    ACE_METADATA_KEYSCALES,
    ACE_METADATA_TIME_SIGNATURES,
  ];
  for (const candidates of candidateGroups) {
    for (const candidate of candidates) addText(` ${candidate}\n`);
  }
  return [...ids];
};

const metadataTokenTrie = (
  tokenizer: PlannerTokenizer,
  values: readonly (string | number)[],
) => {
  const root: MetadataTokenTrie = { children: new Map() };
  for (const candidate of values) {
    const value = String(candidate);
    const tokenIds = plannerTextTokenIds(
      tokenizer,
      ` ${value}\n`,
    );
    if (!tokenIds.length) {
      throw new Error(
        `ACE metadata candidate ${JSON.stringify(value)} tokenized empty.`,
      );
    }
    let node = root;
    for (const tokenId of tokenIds) {
      let child = node.children.get(tokenId);
      if (!child) {
        child = { children: new Map() };
        node.children.set(tokenId, child);
      }
      node = child;
    }
    node.value = value;
  }
  return root;
};

const generateFp32PlannerMetadata = async (
  session: ort.InferenceSession,
  embeddings: Fp32PlannerEmbeddingTable,
  tokenizer: PlannerTokenizer,
  request: PlanMusicRequest,
  recordTiming: Fp32BodyTimingRecorder,
  recordMetadataTiming: Fp32MetadataTimingRecorder,
) => {
  const messages = [
    { role: "system", content: PLANNER_SYSTEM_PROMPT },
    {
      role: "user",
      content: buildPlannerUserPrompt(
        request.prompt,
        request.lyrics,
      ),
    },
  ];
  const formattedPrompt = tokenizer.apply_chat_template(messages, {
    add_generation_prompt: true,
    tokenize: false,
  }) as string;
  const promptIds = plannerTextTokenIds(
    tokenizer,
    formattedPrompt,
  );
  if (!promptIds.length) {
    throw new Error("ACE metadata prompt tokenized empty.");
  }
  const embeddingPrefetchStart = performance.now();
  await embeddings.prefetch(
    plannerMetadataPrefetchTokenIds(tokenizer, promptIds, request),
  );
  recordMetadataTiming(
    "embedding-prefetch",
    performance.now() - embeddingPrefetchStart,
  );
  const promptMask = new Array<number>(promptIds.length).fill(1);
  const promptPositions = Array.from(
    { length: promptIds.length },
    (_, index) => index,
  );
  let cache = createFp32EmptyCache(1);
  let hidden: ort.Tensor | undefined;
  let contextLength = promptIds.length;
  const nextRandom = createPlannerSamplingRandom(
    request.seed ^ 0x4d455441,
  );

  const consumeToken = async (tokenId: number) => {
    hidden?.dispose();
    hidden = undefined;
    const nextContextLength = contextLength + 1;
    const body = await runFp32PlannerBody(
      session,
      embeddings,
      [tokenId],
      new Array<number>(nextContextLength).fill(1),
      [contextLength],
      [1, 1],
      [1, 1],
      cache,
      recordTiming,
    );
    cache = body.cache;
    hidden = body.hidden;
    contextLength = nextContextLength;
  };

  const forceText = async (text: string) => {
    for (const tokenId of plannerTextTokenIds(tokenizer, text)) {
      await consumeToken(tokenId);
    }
  };

  const generateValue = async (
    field: string,
    candidates: readonly (string | number)[],
  ) => {
    let node = metadataTokenTrie(tokenizer, candidates);
    while (node.value === undefined) {
      const tokenIds = [...node.children.keys()];
      if (!tokenIds.length || !hidden) {
        throw new Error(
          `ACE metadata ${field} has no valid token continuation.`,
        );
      }
      const scoreStart = performance.now();
      const scores = await embeddings.scoreTokenIds(
        hidden,
        tokenIds,
      );
      const sparseScoreMilliseconds = performance.now() - scoreStart;
      const samplingStart = performance.now();
      const selectedIndex = sampleTopPToken(
        scores,
        0,
        ACE_METADATA_TEMPERATURE,
        ACE_METADATA_TOP_P,
        nextRandom,
      );
      recordMetadataTiming("sparse-score", sparseScoreMilliseconds);
      recordMetadataTiming(
        "sampling",
        performance.now() - samplingStart,
      );
      const tokenId = tokenIds[selectedIndex];
      if (tokenId === undefined) {
        throw new Error(
          `ACE metadata ${field} selected an unknown token.`,
        );
      }
      node = node.children.get(tokenId)!;
      await consumeToken(tokenId);
    }
    return node.value;
  };

  try {
    const prefill = await runFp32PlannerBody(
      session,
      embeddings,
      promptIds,
      promptMask,
      promptPositions,
      [1, promptIds.length],
      [1, promptIds.length],
      cache,
      recordTiming,
    );
    cache = prefill.cache;
    hidden = prefill.hidden;

    await forceText("<think>\nbpm:");
    const bpm = Number(
      await generateValue(
        "bpm",
        Array.from(
          {
            length:
              ACE_METADATA_BPM_MAX -
              ACE_METADATA_BPM_MIN +
              1,
          },
          (_, index) => ACE_METADATA_BPM_MIN + index,
        ),
      ),
    );

    await forceText("duration:");
    const durationCandidates = plannerMetadataDurationCandidates(request);
    const plannedDurationSeconds = Number(
      await generateValue("duration", durationCandidates),
    );

    await forceText("keyscale:");
    const keyScale = await generateValue(
      "keyscale",
      ACE_METADATA_KEYSCALES,
    );

    await forceText("timesignature:");
    const timeSignature = Number(
      await generateValue(
        "timesignature",
        ACE_METADATA_TIME_SIGNATURES,
      ),
    ) as 2 | 3 | 4 | 6;
    await forceText("</think>");

    const resolvedDuration = resolvePlannerDuration({
      plannedDurationSeconds,
      requestedDurationSeconds: request.durationSeconds,
      recommendedDurationSeconds:
        request.recommendedDurationSeconds,
      autoDuration: request.autoDuration,
    });
    const metadata = {
      bpm,
      caption: request.prompt.trim(),
      durationSeconds: resolvedDuration.durationSeconds,
      durationSource: resolvedDuration.durationSource,
      keyScale,
      language: request.lyrics.trim()
        ? request.vocalLanguage.trim() || "unknown"
        : "unknown",
      timeSignature,
    };
    return {
      metadata,
      reasoning: formatPlannerGeneratedMetadata(metadata),
    };
  } finally {
    hidden?.dispose();
    disposeFp32Tensors(Object.values(cache));
  }
};

const planMusicHighQuality = async (
  request: PlanMusicRequest,
) => {
  const workerStart = performance.now();
  const profiler = new PlannerProfiler();
  let activePlannerPhase: "metadata" | "semantic" = "metadata";
  const recordBodyTiming: Fp32BodyTimingRecorder = (
    event,
    milliseconds,
  ) => {
    profiler.record(
      `${activePlannerPhase}-${event}` as PlannerProfileTimingId,
      milliseconds,
    );
  };
  const recordComponentTiming = (
    event: Fp32PlannerTimingEvent,
    milliseconds: number,
  ) => {
    const setupEvent =
      event === "head-adapter-device" ||
      event === "head-pipeline-create" ||
      event === "head-weight-asset-load" ||
      event === "head-weight-gpu-upload";
    profiler.record(
      (setupEvent
        ? event
        : `${activePlannerPhase}-${event}`) as PlannerProfileTimingId,
      milliseconds,
    );
  };
  if (!(("gpu" in self.navigator) && self.navigator.gpu)) {
    throw new Error(
      "WebGPU is unavailable in this Worker. Use desktop Chromium with WebGPU enabled.",
    );
  }
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.proxy = false;
  ort.env.wasm.wasmPaths = {
    mjs: new URL(
      "./wasm/ort-wasm-simd-threaded.asyncify.mjs",
      self.location.href,
    ).href,
    wasm: new URL(
      "./wasm/ort-wasm-simd-threaded.asyncify.wasm",
      self.location.href,
    ).href,
  };
  ort.env.webgpu.powerPreference = "high-performance";
  const timings: Record<string, number> = {};
  const cachePruneStart = performance.now();
  await pruneSupersededPlannerCache(
    request.highQualityModelId,
    request.highQualityRevision,
  );
  profiler.record("cache-prune", performance.now() - cachePruneStart);
  const loaderStart = performance.now();
  const load = await createFp32PlannerAssetLoader(request);
  profiler.record(
    "asset-loader-create",
    performance.now() - loaderStart,
  );
  const tokenizerProgress = progressCallback(
    "music-planner-high-quality",
  );
  post(
    stage(
      "planner-model",
      `Loading the browser-qualified INT8-weight / FP32-compute ACE 5 Hz planner from ${request.highQualityModelId}.`,
    ),
  );
  const loadStart = performance.now();
  const tokenizerStart = performance.now();
  const tokenizer = await AutoTokenizer.from_pretrained(
    request.highQualityModelId,
    {
      revision: request.highQualityRevision,
      progress_callback: tokenizerProgress,
    },
  );
  await ensurePlannerChatTemplate(
    tokenizer,
    request.highQualityModelId,
    request.highQualityRevision,
  );
  profiler.record(
    "tokenizer-load",
    performance.now() - tokenizerStart,
  );

  post(
    stage(
      "planner-body",
      `Loading the ${(HIGH_QUALITY_PLANNER_BODY_BYTES / 1024 ** 3).toFixed(2)} GiB ` +
        "INT8-weight / FP32-compute transformer body in sub-1-GiB shards.",
    ),
  );
  const graphAsset = HIGH_QUALITY_PLANNER_MODEL_ASSETS.find(
    (asset) => asset.fileName === HIGH_QUALITY_PLANNER_GRAPH_FILE,
  )!;
  const bodyAssetsStart = performance.now();
  const graph = await load(graphAsset.fileName, graphAsset.bytes);
  const externalData: Array<{ path: string; data: Uint8Array }> = [];
  for (const fileName of HIGH_QUALITY_PLANNER_BODY_FILES) {
    const asset = HIGH_QUALITY_PLANNER_MODEL_ASSETS.find(
      (candidate) => candidate.fileName === fileName,
    );
    if (!asset) {
      throw new Error(`FP32 planner manifest is missing ${fileName}.`);
    }
    externalData.push({
      path: fileName.slice(fileName.lastIndexOf("/") + 1),
      data: await load(fileName, asset.bytes),
    });
  }
  profiler.record(
    "body-assets-load",
    performance.now() - bodyAssetsStart,
  );
  const preferredOutputLocation: Record<
    string,
    "cpu" | "gpu-buffer"
  > = {
    last_hidden_state: "cpu",
  };
  for (let layer = 0; layer < 36; layer += 1) {
    preferredOutputLocation[`present.${layer}.key`] = "gpu-buffer";
    preferredOutputLocation[`present.${layer}.value`] = "gpu-buffer";
  }
  const sessionStart = performance.now();
  const session = await ort.InferenceSession.create(graph, {
    executionProviders: ["webgpu"],
    externalData,
    graphOptimizationLevel: "disabled",
    enableCpuMemArena: false,
    enableMemPattern: false,
    preferredOutputLocation,
  });
  profiler.record(
    "body-session-create",
    performance.now() - sessionStart,
  );
  externalData.length = 0;
  timings["planner-model-load"] = performance.now() - loadStart;
  post({
    type: "timing",
    stage: "planner-model-load",
    milliseconds: timings["planner-model-load"],
  });

  const embeddingUrl = fp32PlannerAssetUrl(
    request.highQualityModelId,
    request.highQualityRevision,
    HIGH_QUALITY_PLANNER_EMBEDDING_FILE,
  );
  const embeddingSource = (() => {
    const parsed = new URL(embeddingUrl);
    return `${parsed.origin}${parsed.pathname}`;
  })();
  const embeddingRowStore =
    await createFp32PlannerEmbeddingRowStore(embeddingUrl);
  const embeddings = new Fp32PlannerEmbeddingTable(
    embeddingUrl,
    recordComponentTiming,
    embeddingRowStore,
  );
  post({
    type: "diagnostic",
    key: "planner embedding rows",
    value: embeddingRowStore
      ? "persistent 10 KiB row cache enabled"
      : "memory-only cache; Cache API is unavailable",
  });
  post(
    stage(
      "planner-metadata",
      request.autoDuration
        ? "Running ACE Phase 1 for BPM, duration, key and time signature."
        : "Running ACE Phase 1 with the requested duration constrained.",
    ),
  );
  const metadataStart = performance.now();
  let metadataPhase: Awaited<
    ReturnType<typeof generateFp32PlannerMetadata>
  >;
  try {
    metadataPhase = await generateFp32PlannerMetadata(
      session,
      embeddings,
      tokenizer,
      request,
      recordBodyTiming,
      (event, milliseconds) =>
        profiler.record(
          `metadata-${event}` as PlannerProfileTimingId,
          milliseconds,
        ),
    );
  } catch (error) {
    embeddings.clear();
    await session.release();
    throw error;
  }
  const { metadata, reasoning } = metadataPhase;
  timings["planner-metadata"] =
    performance.now() - metadataStart;
  profiler.record("metadata-total", timings["planner-metadata"]);
  const embeddingAfterMetadata = embeddings.stats();
  post({
    type: "timing",
    stage: "planner-metadata",
    milliseconds: timings["planner-metadata"],
  });
  post({
    type: "diagnostic",
    key: "ACE metadata",
    value:
      `${metadata.bpm} BPM · ${metadata.keyScale} · ` +
      `${metadata.timeSignature}/4 · ${metadata.durationSeconds}s ` +
      `(${metadata.durationSource})`,
  });

  const semanticTokenizationStart = performance.now();
  const messages = [
    { role: "system", content: PLANNER_SYSTEM_PROMPT },
    {
      role: "user",
      content: buildPlannerUserPrompt(
        request.prompt,
        request.lyrics,
      ),
    },
  ];
  const conditionalBase = tokenizer.apply_chat_template(messages, {
    add_generation_prompt: true,
    tokenize: false,
  }) as string;
  const unconditionalBase = tokenizer.apply_chat_template(
    [
      { role: "system", content: PLANNER_SYSTEM_PROMPT },
      {
        role: "user",
        content: PLANNER_UNCONDITIONAL_USER_PROMPT,
      },
    ],
    {
      add_generation_prompt: true,
      tokenize: false,
    },
  ) as string;
  const prompts = [
    `${conditionalBase}${reasoning}\n\n`,
    `${unconditionalBase}<think>\n\n</think>\n\n`,
  ];
  const originalPaddingSide = tokenizer.padding_side;
  tokenizer.padding_side = "left";
  const tokenized = tokenizer(prompts, {
    add_special_tokens: false,
    padding: true,
    return_tensor: true,
  }) as Record<string, Tensor>;
  tokenizer.padding_side = originalPaddingSide;
  const inputIdsTensor = tokenized.input_ids;
  const attentionMaskTensor = tokenized.attention_mask;
  if (!inputIdsTensor || !attentionMaskTensor) {
    disposeInputs(tokenized);
    embeddings.clear();
    await session.release();
    throw new Error("FP32 planner tokenizer returned no input tensors.");
  }
  const batchSize = Number(inputIdsTensor.dims[0]);
  const promptLength = Number(inputIdsTensor.dims[1]);
  if (batchSize !== 2 || promptLength < 1) {
    disposeInputs(tokenized);
    embeddings.clear();
    await session.release();
    throw new Error(
      `FP32 planner tokenizer returned ${JSON.stringify(inputIdsTensor.dims)}.`,
    );
  }
  const promptIds = Array.from(inputIdsTensor.data, Number);
  const promptMask = Array.from(
    attentionMaskTensor.data,
    Number,
  );
  disposeInputs(tokenized);
  const { positions: promptPositions, tokenCounts } =
    fp32PositionIds(promptMask, batchSize, promptLength);
  const [
    conditionalInputIdsSha256,
    unconditionalInputIdsSha256,
    conditionalAttentionMaskSha256,
    unconditionalAttentionMaskSha256,
    promptSha256,
    lyricsSha256,
    metadataReasoningSha256,
  ] = await Promise.all([
    sha256Numbers(promptIds.slice(0, promptLength)),
    sha256Numbers(promptIds.slice(promptLength)),
    sha256Numbers(promptMask.slice(0, promptLength)),
    sha256Numbers(promptMask.slice(promptLength)),
    sha256Text(request.prompt),
    sha256Text(request.lyrics),
    sha256Text(reasoning),
  ]);
  const inputFingerprint: PlannerInputFingerprint = {
    batchSize,
    paddedTokens: promptLength,
    realTokens: [...tokenCounts],
    conditionalInputIdsSha256,
    unconditionalInputIdsSha256,
    conditionalAttentionMaskSha256,
    unconditionalAttentionMaskSha256,
    promptSha256,
    lyricsSha256,
    metadataReasoningSha256,
  };
  profiler.record(
    "semantic-tokenization",
    performance.now() - semanticTokenizationStart,
  );
  activePlannerPhase = "semantic";
  const semanticEmbeddingPrefetchStart = performance.now();
  try {
    await embeddings.prefetch(promptIds);
  } catch (error) {
    embeddings.clear();
    await session.release();
    throw error;
  }
  profiler.record(
    "semantic-embedding-prefetch",
    performance.now() - semanticEmbeddingPrefetchStart,
  );
  post({
    type: "diagnostic",
    key: "planner semantic input",
    value:
      `${promptLength} padded tokens · ${tokenCounts.join("/")} real · ` +
      `conditional IDs ${conditionalInputIdsSha256.slice(0, 12)}…`,
  });

  post(
    stage(
      "planner-head",
      "Loading the sharded 64,000-row FP32 audio-code head.",
    ),
  );
  const headFiles = HIGH_QUALITY_PLANNER_HEAD_FILES.map(
    (fileName, index) => {
      const asset = HIGH_QUALITY_PLANNER_MODEL_ASSETS.find(
        (candidate) => candidate.fileName === fileName,
      );
      if (!asset) {
        throw new Error(`FP32 planner manifest is missing ${fileName}.`);
      }
      return {
        fileName,
        bytes: asset.bytes,
        rowCount: index === 7 ? 6_656 : 8_192,
      };
    },
  );
  let head: Fp32PlannerAudioCodeHead;
  const headLoadStart = performance.now();
  try {
    head = await Fp32PlannerAudioCodeHead.create(
      headFiles,
      load,
      (_fileName, loaded, total) => {
        if (loaded === total) {
          post({
            type: "diagnostic",
            key: "FP32 audio-code head",
            value: `${(total / 1e9).toFixed(2)} GB loaded in ${headFiles.length} WebGPU shards`,
          });
        }
      },
      recordComponentTiming,
    );
    profiler.record("head-load", performance.now() - headLoadStart);
  } catch (error) {
    embeddings.clear();
    await session.release();
    throw error;
  }

  let cache = createFp32EmptyCache(batchSize);
  const targetCodes = semanticCodeCount(
    metadata.durationSeconds,
  );
  const semanticCodeIds: number[] = [];
  const guided = new Float32Array(FP32_AUDIO_CODE_COUNT);
  const nextRandom = createPlannerSamplingRandom(request.seed);
  const codesStart = performance.now();
  activePlannerPhase = "semantic";
  const postPlannerProfile = (final = false) => {
    const update: PlannerProfileUpdate = {
      type: "planner-profile",
      report: profiler.report({
        completedSemanticSteps: semanticCodeIds.length,
        targetSemanticSteps: targetCodes,
        embeddingSource,
        embeddingTotal: embeddings.stats(),
        embeddingAfterMetadata,
        input: inputFingerprint,
        final,
      }),
    };
    post(update);
    return update.report;
  };

  try {
    post(
      stage(
        "planner-semantic-codes",
        `Generating exactly ${targetCodes} high-quality INT8-weight / FP32-compute semantic codes.`,
      ),
    );
    let body = await runFp32PlannerBody(
      session,
      embeddings,
      promptIds,
      promptMask,
      promptPositions,
      [batchSize, promptLength],
      [batchSize, promptLength],
      cache,
      recordBodyTiming,
    );
    cache = body.cache;
    let audioLogits = await head.forward(body.hidden);
    body.hidden.dispose();

    while (semanticCodeIds.length < targetCodes) {
      const conditional = audioLogits.subarray(
        0,
        FP32_AUDIO_CODE_COUNT,
      );
      const unconditional = audioLogits.subarray(
        FP32_AUDIO_CODE_COUNT,
      );
      const cfgStart = performance.now();
      const blend = blendPlannerCfgLogits(
        conditional,
        unconditional,
        DEFAULT_PLANNER_CFG_SCALE,
        guided,
      );
      profiler.record("semantic-cfg", performance.now() - cfgStart);
      if (blend.mode !== "cfg") {
        throw new Error(
          `FP32 planner CFG failed at semantic step ${semanticCodeIds.length + 1}/${targetCodes}: ` +
            `${blend.conditionalFinite}/${FP32_AUDIO_CODE_COUNT} conditional, ` +
            `${blend.unconditionalFinite}/${FP32_AUDIO_CODE_COUNT} unconditional, ` +
            `${blend.guidedFinite}/${FP32_AUDIO_CODE_COUNT} guided logits finite.`,
        );
      }
      if (semanticCodeIds.length === 0) {
        const rankingStart = performance.now();
        const conditionalTop = plannerTopCodes(conditional);
        const cfgTop = plannerTopCodes(guided);
        post({
          type: "diagnostic",
          key: "FP32 first semantic ranking",
          value:
            `conditional ${conditionalTop.slice(0, 5).join(",")} · ` +
            `CFG ${cfgTop.slice(0, 5).join(",")}`,
        });
        profiler.record(
          "semantic-first-ranking",
          performance.now() - rankingStart,
        );
      }
      const samplingStart = performance.now();
      const code = sampleTopPToken(guided, 0, 0.85, 0.9, nextRandom);
      profiler.record(
        "semantic-sampling",
        performance.now() - samplingStart,
      );
      semanticCodeIds.push(code);
      if (
        semanticCodeIds.length === 1 ||
        semanticCodeIds.length % 10 === 0
      ) {
        postPlannerProfile();
      }
      if (semanticCodeIds.length >= targetCodes) break;

      const tokenId = FP32_AUDIO_CODE_TOKEN_START + code;
      embeddings.setRow(
        tokenId,
        await head.readEmbeddingRow(code),
      );
      const nextInputStart = performance.now();
      const generatedCount = semanticCodeIds.length;
      const totalLength = promptLength + generatedCount;
      const nextMask = new Array<number>(batchSize * totalLength);
      for (let batch = 0; batch < batchSize; batch += 1) {
        const previous = promptMask.slice(
          batch * promptLength,
          (batch + 1) * promptLength,
        );
        const offset = batch * totalLength;
        for (let index = 0; index < previous.length; index += 1) {
          nextMask[offset + index] = previous[index]!;
        }
        for (let index = promptLength; index < totalLength; index += 1) {
          nextMask[offset + index] = 1;
        }
      }
      const nextPositions = tokenCounts.map(
        (count) => count + generatedCount - 1,
      );
      profiler.record(
        "semantic-next-input-prep",
        performance.now() - nextInputStart,
      );
      body = await runFp32PlannerBody(
        session,
        embeddings,
        [tokenId, tokenId],
        nextMask,
        nextPositions,
        [batchSize, 1],
        [batchSize, 1],
        cache,
        recordBodyTiming,
      );
      cache = body.cache;
      audioLogits = await head.forward(body.hidden);
      body.hidden.dispose();
    }

    timings["planner-semantic-codes"] =
      performance.now() - codesStart;
    profiler.record(
      "semantic-total",
      timings["planner-semantic-codes"],
    );

    const diagnostics = analyzePlannerSemanticCodes(
      semanticCodeIds,
    );
    post({
      type: "diagnostic",
      key: "planner semantic diversity",
      value:
        `${diagnostics.uniqueCount}/${targetCodes} unique · ` +
        `${diagnostics.transitionCount} transitions · ` +
        `${(diagnostics.adjacentRepeatRatio * 100).toFixed(1)}% adjacent repeats · ` +
        `longest run ${diagnostics.longestIdenticalRun}`,
    });
    post({
      type: "diagnostic",
      key: "planner semantic code sample",
      value:
        `${semanticCodeIds.slice(0, 16).join(",")} … ` +
        semanticCodeIds.slice(-16).join(","),
    });
    post({
      type: "timing",
      stage: "planner-semantic-codes",
      milliseconds: timings["planner-semantic-codes"],
    });
  } finally {
    const releaseStart = performance.now();
    disposeFp32Tensors(Object.values(cache));
    embeddings.clear();
    head.dispose();
    await session.release();
    profiler.record(
      "resource-release",
      performance.now() - releaseStart,
    );
  }
  profiler.record("worker-total", performance.now() - workerStart);
  const plannerProfile = postPlannerProfile(true);
  post({
    type: "plan-complete",
    plannerQuality: "high-quality",
    semanticCodeIds,
    metadata,
    reasoning,
    model: request.highQualityModelId,
    revision: request.highQualityRevision,
    seed: request.seed,
    timings,
    plannerProfile,
  } satisfies PlanCompleteUpdate);
};

const planMusicTurbo = async (request: PlanMusicRequest) => {
  if (!(("gpu" in self.navigator) && self.navigator.gpu)) {
    throw new Error(
      "WebGPU is unavailable in this Worker. Use desktop Chromium with WebGPU enabled.",
    );
  }
  const onnxEnvironment = env.backends.onnx;
  if (onnxEnvironment.wasm) {
    onnxEnvironment.wasm.numThreads = 1;
    onnxEnvironment.wasm.proxy = false;
    onnxEnvironment.wasm.wasmPaths = {
      mjs: new URL(
        "./wasm/ort-wasm-simd-threaded.asyncify.mjs",
        self.location.href,
      ).href,
      wasm: new URL(
        "./wasm/ort-wasm-simd-threaded.asyncify.wasm",
        self.location.href,
      ).href,
    };
  }
  if (onnxEnvironment.webgpu) {
    onnxEnvironment.webgpu.powerPreference = "high-performance";
  }

  const timings: Record<string, number> = {};
  post(
    stage(
      "planner-model",
      `Loading the exact ACE-Step 5 Hz 4B planner from ${request.modelId}.`,
    ),
  );
  const loadStart = performance.now();
  const plannerProgress = progressCallback("music-planner");
  await pruneSupersededPlannerCache(
    request.modelId,
    request.revision,
  );
  await configurePlannerBrowserCache();
  const plannerDtype = "q8";
  await installPlannerDiagnosticGraph(
    request.modelId,
    request.revision,
    plannerDtype,
  );
  const tokenizer = await AutoTokenizer.from_pretrained(request.modelId, {
    revision: request.revision,
    progress_callback: plannerProgress,
  });
  await ensurePlannerChatTemplate(
    tokenizer,
    request.modelId,
    request.revision,
  );
  const model = await Qwen3ForCausalLM.from_pretrained(request.modelId, {
    revision: request.revision,
    device: "webgpu",
    dtype: plannerDtype,
    // The split Q6 body stores packed weights behind ONNX unpack/dequantize
    // subgraphs. ORT's extended optimizer can constant-fold those subgraphs
    // while creating the session, temporarily materializing the full FP16
    // 4B model in the WASM heap and failing with std::bad_alloc. Keep the
    // dequantization on the WebGPU execution path instead.
    session_options: {
      graphOptimizationLevel: "disabled",
      enableCpuMemArena: false,
      enableMemPattern: false,
    },
    progress_callback: plannerProgress,
  });
  let plannerHead: PlannerQ8WebGpuHead;
  try {
    post(
      stage(
        "planner-head",
        "Loading the split Q8 token embedding and language-model head into dedicated WebGPU buffers.",
      ),
    );
    plannerHead = await PlannerQ8WebGpuHead.create(
      request.modelId,
      request.revision,
      (file, loaded, total) => {
        post({
          type: "download",
          assetId: `music-planner:${file}`,
          group: "music-planner",
          label: file,
          loaded,
          total,
          cached: false,
        });
      },
    );
    installPlannerWebGpuHead(model, plannerHead);
  } catch (error) {
    await model.dispose();
    throw error;
  }
  const plannerDiagnostics = installPlannerForwardDiagnostics(model);
  timings["planner-model-load"] = performance.now() - loadStart;
  post({
    type: "timing",
    stage: "planner-model-load",
    milliseconds: timings["planner-model-load"],
  } satisfies TimingUpdate);

  try {
    random.seed(request.seed);
    post(
      stage(
        "planner-metadata",
        "Packing deterministic ACE music metadata without an extra model pass.",
      ),
    );
    const metadataStart = performance.now();
    const messages = [
      { role: "system", content: PLANNER_SYSTEM_PROMPT },
      {
        role: "user",
        content: buildPlannerUserPrompt(request.prompt, request.lyrics),
      },
    ];
    const metadata = deterministicPlannerMetadata(
      request.prompt,
      request.lyrics,
      request.vocalLanguage,
      request.autoDuration
        ? request.recommendedDurationSeconds
        : request.durationSeconds,
    );
    metadata.durationSource = request.autoDuration
      ? "recommended"
      : "requested";
    const reasoning = formatPlannerMetadata(metadata);
    timings["planner-metadata"] = performance.now() - metadataStart;
    post({
      type: "diagnostic",
      key: "Turbo metadata mode",
      value:
        "deterministic defaults · avoids a redundant full-vocabulary autoregressive pass",
    });
    post({
      type: "timing",
      stage: "planner-metadata",
      milliseconds: timings["planner-metadata"],
    } satisfies TimingUpdate);

    const targetCodes = semanticCodeCount(
      metadata.durationSeconds,
    );
    post(
      stage(
        "planner-semantic-codes",
        `Generating exactly ${targetCodes} constrained 5 Hz semantic codes.`,
      ),
    );
    const codesStart = performance.now();
    const formattedPrefix = tokenizer.apply_chat_template(messages, {
      add_generation_prompt: true,
      tokenize: false,
    }) as string;
    const conditionalPrefix =
      `${formattedPrefix}${formatPlannerMetadata(metadata)}\n\n`;
    const unconditionalFormattedPrefix = tokenizer.apply_chat_template(
      [
        { role: "system", content: PLANNER_SYSTEM_PROMPT },
        {
          role: "user",
          content: PLANNER_UNCONDITIONAL_USER_PROMPT,
        },
      ],
      {
        add_generation_prompt: true,
        tokenize: false,
      },
    ) as string;
    const unconditionalPrefix =
      `${unconditionalFormattedPrefix}<think>\n\n</think>\n\n`;
    const originalPaddingSide = tokenizer.padding_side;
    tokenizer.padding_side = "left";
    let codeInputs: Record<string, Tensor>;
    try {
      codeInputs = tokenizer(
        [conditionalPrefix, unconditionalPrefix],
        {
          add_special_tokens: false,
          padding: true,
          return_tensor: true,
        },
      ) as Record<string, Tensor>;
    } finally {
      tokenizer.padding_side = originalPaddingSide;
    }
    const codePromptLength = codeInputs.input_ids.dims.at(-1) ?? 0;
    const codeProcessors = new LogitsProcessorList();
    codeProcessors.push(
      new AudioCodeSamplerProcessor(
        codePromptLength,
        targetCodes,
        DEFAULT_PLANNER_CFG_SCALE,
      ),
    );
    post({
      type: "diagnostic",
      key: "planner inference mode",
      value:
        `official two-row codes CFG ${DEFAULT_PLANNER_CFG_SCALE.toFixed(1)} · ` +
        "training-aligned NO USER INPUT branch · shared sampled tokens",
    });
    plannerDiagnostics.enable();
    const codeOutput = (await model.generate({
      ...codeInputs,
      max_new_tokens: targetCodes + 1,
      // The processor performs deterministic top-p sampling and forces the
      // selected token while restricting output to ACE's 64k audio codebook.
      do_sample: false,
      eos_token_id: PLANNER_EOS_TOKEN,
      pad_token_id: PLANNER_PAD_TOKEN,
      repetition_penalty: 1,
      logits_processor: codeProcessors,
    })) as unknown as {
      data: BigInt64Array | BigUint64Array | Int32Array;
      dims: readonly number[];
      dispose: () => void;
    };
    const codeTokenIds = generatedIds(codeOutput, codePromptLength).filter(
      (tokenId) =>
        tokenId >= AUDIO_CODE_TOKEN_START &&
        tokenId < AUDIO_CODE_TOKEN_END,
    );
    const semanticCodeIds = codeTokenIds.map(tokenIdToAudioCode);
    codeOutput.dispose();
    disposeInputs(codeInputs);
    if (semanticCodeIds.length !== targetCodes) {
      throw new Error(
        `ACE planner returned ${semanticCodeIds.length} semantic codes; expected exactly ${targetCodes}.`,
      );
    }
    const codeDiagnostics =
      analyzePlannerSemanticCodes(semanticCodeIds);
    post({
      type: "diagnostic",
      key: "planner semantic diversity",
      value:
        `${codeDiagnostics.uniqueCount}/${targetCodes} unique · ` +
        `${codeDiagnostics.transitionCount} transitions · ` +
        `${(codeDiagnostics.adjacentRepeatRatio * 100).toFixed(1)}% adjacent repeats · ` +
        `longest run ${codeDiagnostics.longestIdenticalRun} · ` +
        `dominant code ${(codeDiagnostics.dominantCodeRatio * 100).toFixed(1)}%`,
    });
    post({
      type: "diagnostic",
      key: "planner semantic code sample",
      value:
        `${semanticCodeIds.slice(0, 12).join(",")} … ` +
        semanticCodeIds.slice(-12).join(","),
    });
    timings["planner-semantic-codes"] = performance.now() - codesStart;
    post({
      type: "timing",
      stage: "planner-semantic-codes",
      milliseconds: timings["planner-semantic-codes"],
    } satisfies TimingUpdate);

    post({
      type: "plan-complete",
      plannerQuality: "turbo",
      semanticCodeIds,
      metadata,
      reasoning,
      model: request.modelId,
      revision: request.revision,
      seed: request.seed,
      timings,
    } satisfies PlanCompleteUpdate);
  } finally {
    await model.dispose();
    plannerHead.dispose();
  }
};

const planMusic = (request: PlanMusicRequest) =>
  request.plannerQuality === "high-quality"
    ? planMusicHighQuality(request)
    : planMusicTurbo(request);

self.onmessage = (event: MessageEvent<LanguageWorkerRequest>) => {
  const operation =
    event.data.type === "write-lyrics"
      ? writeLyrics(event.data)
      : planMusic(event.data);
  void operation.catch((error: unknown) => {
    post({
      type: "error",
      stage:
        event.data.type === "write-lyrics"
          ? "lyrics-writer"
          : "music-planner",
      message: error instanceof Error ? error.message : String(error),
      stack: error instanceof Error ? error.stack : undefined,
    } satisfies ErrorUpdate);
  });
};
