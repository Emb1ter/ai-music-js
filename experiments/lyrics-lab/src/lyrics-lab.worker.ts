/// <reference lib="webworker" />

import {
  AutoTokenizer,
  Gemma4ForCausalLM,
  Qwen3_5ForCausalLM,
  env,
  random,
} from "@huggingface/transformers";
import {
  cleanLyrics,
  compactLyrics,
  countLyricWords,
  lyricQualityIssues,
} from "../../../lib/lyrics";
import { LANGUAGE_CACHE_NAME } from "../../../lib/language-model-manifest";
import type {
  LyricLabModelDtype,
  LyricLabModelFamily,
} from "./models";
import { missingExactTermIssues } from "./exact-words";
import { buildFactGroundedLyricsPrompt } from "./prompts";

declare const self: DedicatedWorkerGlobalScope;

type Variant = "a" | "b" | "c" | "d" | "e" | "f" | "g" | "h" | "i" | "j" | "k" | "l" | "m" | "o" | "p" | "q";

type GenerateRequest = {
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
  modelDtype: LyricLabModelDtype;
  modelFamily: LyricLabModelFamily;
  modelLabel: string;
  modelLocalOnly: boolean;
  wasmUrl: string;
  wasmModuleUrl: string;
};

type DisposeRequest = {
  type: "dispose";
  requestId: string;
};

type WorkerRequest = GenerateRequest | DisposeRequest;

type LabUpdate =
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

const post = (update: LabUpdate) => self.postMessage(update);

env.cacheKey = LANGUAGE_CACHE_NAME;

type LoadedModel = {
  tokenizer: Awaited<ReturnType<typeof AutoTokenizer.from_pretrained>>;
  model:
    | Awaited<ReturnType<typeof Qwen3_5ForCausalLM.from_pretrained>>
    | Awaited<ReturnType<typeof Gemma4ForCausalLM.from_pretrained>>;
  loadMilliseconds: number;
};

let loadedModel: Promise<LoadedModel> | undefined;

const progressCallback = (variant: Variant) => (event: unknown) => {
  if (!event || typeof event !== "object") return;
  const progress = event as Record<string, unknown>;
  if (progress.status !== "progress") return;
  const loaded = typeof progress.loaded === "number" ? progress.loaded : 0;
  const total = typeof progress.total === "number" ? progress.total : 0;
  if (!(total > 0)) return;
  const file =
    typeof progress.file === "string"
      ? progress.file
      : typeof progress.name === "string"
        ? progress.name
        : "Qwen3.5 model";
  post({ type: "download", variant, file, loaded, total });
};

const configureRuntime = (request: GenerateRequest) => {
  env.allowLocalModels = request.modelLocalOnly;
  env.allowRemoteModels = !request.modelLocalOnly;
  env.localModelPath = "/models/";
  const onnxEnvironment = env.backends.onnx;
  if (onnxEnvironment.wasm) {
    onnxEnvironment.wasm.numThreads = 1;
    onnxEnvironment.wasm.proxy = false;
    onnxEnvironment.wasm.wasmPaths = {
      mjs: request.wasmModuleUrl,
      wasm: request.wasmUrl,
    };
  }
  if (onnxEnvironment.webgpu) {
    onnxEnvironment.webgpu.powerPreference = "high-performance";
  }
};

const loadModel = (request: GenerateRequest) => {
  if (loadedModel) return loadedModel;
  configureRuntime(request);
  loadedModel = (async () => {
    post({
      type: "stage",
      variant: request.variant,
      stage: "model-load",
      detail: `Loading ${request.modelLabel} on WebGPU.`,
    });
    const startedAt = performance.now();
    const modelPromise =
      request.modelFamily === "gemma4"
        ? Gemma4ForCausalLM.from_pretrained(request.modelId, {
            revision: request.modelRevision,
            local_files_only: request.modelLocalOnly,
            device: "webgpu",
            dtype: request.modelDtype,
            progress_callback: progressCallback(request.variant),
          })
        : Qwen3_5ForCausalLM.from_pretrained(request.modelId, {
            revision: request.modelRevision,
            local_files_only: request.modelLocalOnly,
            device: "webgpu",
            dtype: request.modelDtype,
            progress_callback: progressCallback(request.variant),
          });
    const [tokenizer, model] = await Promise.all([
      AutoTokenizer.from_pretrained(request.modelId, {
        revision: request.modelRevision,
        local_files_only: request.modelLocalOnly,
        progress_callback: progressCallback(request.variant),
      }),
      modelPromise,
    ]);
    return {
      tokenizer,
      model,
      loadMilliseconds: performance.now() - startedAt,
    };
  })();
  loadedModel.catch(() => {
    loadedModel = undefined;
  });
  return loadedModel;
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

type GenerateTextOptions = {
  tokenizer: LoadedModel["tokenizer"];
  model: LoadedModel["model"];
  systemPrompt: string;
  userPrompt: string;
  seed: number;
  temperature: number;
  topK: number;
  topP: number;
  repetitionPenalty: number;
  maxNewTokens: number;
};

const generateText = async ({
  tokenizer,
  model,
  systemPrompt,
  userPrompt,
  seed,
  temperature,
  topK,
  topP,
  repetitionPenalty,
  maxNewTokens,
}: GenerateTextOptions) => {
  random.seed(seed);
  const inputs = tokenizer.apply_chat_template(
    [
      { role: "system", content: systemPrompt },
      { role: "user", content: userPrompt },
    ],
    {
      add_generation_prompt: true,
      tokenize: true,
      return_tensor: true,
      return_dict: true,
      // This is a chat-template variable, not a tokenizer encoding option.
      ...({ enable_thinking: false } as { enable_thinking: boolean }),
    },
  );
  const inputTokens = inputs.input_ids.dims.at(-1) ?? 0;
  const startedAt = performance.now();
  const output = (await model.generate({
    ...inputs,
    max_new_tokens: maxNewTokens,
    do_sample: true,
    temperature,
    top_k: topK,
    top_p: topP,
    repetition_penalty: repetitionPenalty,
  })) as unknown as {
    data: BigInt64Array | BigUint64Array | Int32Array;
    dims: readonly number[];
    dispose: () => void;
  };
  const sequenceLength = output.dims.at(-1) ?? output.data.length;
  const generatedIds = Array.from(
    output.data.slice(inputTokens, sequenceLength),
    Number,
  );
  const text = tokenizer.decode(generatedIds, {
    skip_special_tokens: true,
  });
  const milliseconds = performance.now() - startedAt;
  output.dispose();
  disposeInputs(inputs);
  return {
    text,
    inputTokens,
    outputTokens: generatedIds.length,
    milliseconds,
  };
};

const cleanFactSheet = (value: string) =>
  value
    .replace(/<think>[\s\S]*?(?:<\/think>|$)/gi, "")
    .replace(/^```(?:text|markdown|json)?\s*/i, "")
    .replace(/\s*```$/, "")
    .trim();

const generate = async (request: GenerateRequest) => {
  if (!("gpu" in self.navigator) || !self.navigator.gpu) {
    throw new Error(
      "WebGPU is unavailable. Open this page in current desktop Chrome or Edge with hardware acceleration enabled.",
    );
  }
  const { tokenizer, model, loadMilliseconds } = await loadModel(request);
  let factSheet: string | undefined;
  let factRaw: string | undefined;
  let factInputTokens: number | undefined;
  let factOutputTokens: number | undefined;
  let factExtractionMilliseconds: number | undefined;
  let effectiveUserPrompt = request.userPrompt;
  if (request.mode === "two-pass") {
    if (!request.factSystemPrompt?.trim() || !request.factUserPrompt?.trim()) {
      throw new Error("Two-pass mode requires fact-extraction prompts.");
    }
    post({
      type: "stage",
      variant: request.variant,
      stage: "fact-extraction",
      detail: "Pass 1/2: extracting immutable story facts at low temperature.",
    });
    const facts = await generateText({
      tokenizer,
      model,
      systemPrompt: request.factSystemPrompt,
      userPrompt: request.factUserPrompt,
      seed: request.seed,
      temperature: 0.25,
      topK: 10,
      topP: 0.9,
      repetitionPenalty: 1.05,
      maxNewTokens: 180,
    });
    factRaw = facts.text;
    factSheet = cleanFactSheet(facts.text);
    factInputTokens = facts.inputTokens;
    factOutputTokens = facts.outputTokens;
    factExtractionMilliseconds = facts.milliseconds;
    if (!factSheet) {
      throw new Error("Qwen returned an empty fact sheet in pass 1.");
    }
    effectiveUserPrompt = buildFactGroundedLyricsPrompt({
      factSheet,
      durationSeconds: request.durationSeconds,
      maxWords: request.maxWords,
      language: request.language,
    });
  }
  post({
    type: "stage",
    variant: request.variant,
    stage: "generating",
    detail:
      request.mode === "two-pass"
        ? `Pass 2/2: writing grounded lyrics with seed ${request.seed}.`
        : `Running prompt ${request.variant.toUpperCase()} with seed ${request.seed}.`,
  });
  const lyricsResult = await generateText({
    tokenizer,
    model,
    systemPrompt: request.systemPrompt,
    userPrompt: effectiveUserPrompt,
    seed: request.seed,
    temperature: request.temperature,
    topK: request.topK,
    topP: request.topP,
    repetitionPenalty: request.repetitionPenalty,
    maxNewTokens: request.maxNewTokens,
  });
  const raw = lyricsResult.text;
  const cleaned = cleanLyrics(raw);
  const generatedWordCount = countLyricWords(cleaned);
  const processed = compactLyrics(cleaned, request.maxWords);
  const retainedWordCount = countLyricWords(processed);
  const removedWordCount = generatedWordCount - retainedWordCount;
  const issues = [
    ...lyricQualityIssues(processed, request.maxWords),
    ...missingExactTermIssues(request.sourceBrief, processed),
    ...(removedWordCount > 0
      ? [
          `post-processing removed ${removedWordCount} words (${generatedWordCount} generated, ${retainedWordCount} retained; ACE limit ${request.maxWords})`,
        ]
      : []),
  ];
  post({
    type: "result",
    variant: request.variant,
    raw,
    cleaned,
    processed,
    issues,
    inputTokens: lyricsResult.inputTokens,
    outputTokens: lyricsResult.outputTokens,
    wordCount: retainedWordCount,
    generatedWordCount,
    generationMilliseconds: lyricsResult.milliseconds,
    modelLoadMilliseconds: loadMilliseconds,
    factSheet,
    factRaw,
    factInputTokens,
    factOutputTokens,
    factExtractionMilliseconds,
    effectiveUserPrompt,
  });
};

const disposeModel = async (requestId: string) => {
  const modelToDispose = loadedModel;
  loadedModel = undefined;
  if (modelToDispose) {
    try {
      const { model } = await modelToDispose;
      await model.dispose();
    } catch {
      // A failed or partially-created session still needs its Worker stopped.
      // The acknowledgement lets the page finish that cleanup deterministically.
    }
  }
  post({ type: "disposed", requestId });
};

self.onmessage = (event: MessageEvent<WorkerRequest>) => {
  const request = event.data;
  if (request.type === "dispose") {
    void disposeModel(request.requestId);
    return;
  }
  if (request.type !== "generate") return;
  void generate(request).catch((error) => {
    post({
      type: "error",
      variant: request.variant,
      message: error instanceof Error ? error.message : String(error),
      stack: error instanceof Error ? error.stack : undefined,
    });
  });
};

export {};
