import { afterEach, describe, expect, it, vi } from "vitest";
import {
  AceStepWebGpu,
  DEFAULT_GENERATION_DOWNLOAD_BYTES,
  DEFAULT_HIGH_QUALITY_PLANNER_MODEL,
  DEFAULT_HIGH_QUALITY_PLANNER_MODEL_REVISION,
  DEFAULT_LYRICS_MODEL,
  DEFAULT_LYRICS_MODEL_REVISION,
  DEFAULT_MODEL_BASE_URL,
  DEFAULT_VOCAL_PROMPT,
  HIGH_PRECISION_GENERATION_DOWNLOAD_BYTES,
  HIGH_PRECISION_MODEL_FILES,
  LOCAL_MODEL_FILES,
  PIPELINE_BUILD,
  FULL_MODEL_DOWNLOAD_BYTES,
  TOTAL_DOWNLOAD_BYTES,
  getRequiredAssets,
} from "../package-src/index";
import type {
  CacheInventory,
  CompleteUpdate,
  LanguageWorkerRequest,
  WorkerRequest,
  WorkerUpdate,
} from "../lib/worker-protocol";

const cacheInventory: CacheInventory = {
  origin: "https://app.example",
  cacheName: "ai-music-js-test",
  expectedBytes: TOTAL_DOWNLOAD_BYTES,
  storedBytes: 2_703_943_680,
  readyBytes: 2_703_943_680,
  missingBytes: TOTAL_DOWNLOAD_BYTES - 2_703_943_680,
  usageBytes: 2_800_000_000,
  quotaBytes: 10_000_000_000,
  availableBytes: 7_200_000_000,
  persisted: true,
  models: [
    {
      id: "dit",
      label: "ACE-Step XL Turbo 4B DiT · INT4",
      expectedBytes: 2_712_961_096,
      storedBytes: 2_703_943_680,
      complete: false,
      partial: true,
      assets: [],
    },
  ],
};

class FakeAudioBuffer {
  readonly length: number;
  readonly numberOfChannels: number;
  readonly sampleRate: number;
  readonly channels: Float32Array[];

  constructor(options: AudioBufferOptions) {
    this.length = options.length;
    this.numberOfChannels = options.numberOfChannels ?? 1;
    this.sampleRate = options.sampleRate;
    this.channels = Array.from(
      { length: this.numberOfChannels },
      () => new Float32Array(options.length),
    );
  }

  copyToChannel(source: Float32Array, channel: number) {
    this.channels[channel].set(source);
  }
}

class FakeWorker {
  onmessage: ((event: MessageEvent<WorkerUpdate>) => void) | null = null;
  onerror: ((event: ErrorEvent) => void) | null = null;
  readonly requests: Array<WorkerRequest | LanguageWorkerRequest> = [];
  terminated = false;

  postMessage(request: WorkerRequest | LanguageWorkerRequest) {
    this.requests.push(request);
    queueMicrotask(() => {
      if (request.type === "write-lyrics") {
        this.emit({
          type: "lyrics-complete",
          lyrics:
            "[Verse]\nNeon wakes the street\nWe move with the beat\n\n[Chorus]\nSing into the light\nKeep the fire bright",
          model: request.modelId,
          revision: request.revision,
          seed: request.seed,
          durationSeconds: request.durationSeconds,
          maxWords: request.maxWords,
          attempts: 1,
          timings: { "lyrics-generation": 123 },
        });
        return;
      }
      if (request.type === "plan-music") {
        const plannedDurationSeconds = request.autoDuration
          ? request.recommendedDurationSeconds
          : request.durationSeconds;
        this.emit({
          type: "plan-complete",
          plannerQuality: request.plannerQuality,
          semanticCodeIds: Array.from(
            { length: plannedDurationSeconds * 5 },
            (_, index) => index % 64_000,
          ),
          metadata: {
            bpm: 120,
            caption: request.prompt,
            durationSeconds: plannedDurationSeconds,
            durationSource: request.autoDuration
              ? "recommended"
              : "requested",
            keyScale: "C major",
            language: request.vocalLanguage,
            timeSignature: 4,
          },
          reasoning: "<think>\nbpm: 120\n</think>",
          model: request.modelId,
          revision: request.revision,
          seed: request.seed,
          timings: { "planner-semantic-codes": 321 },
        });
        return;
      }
      if (request.type === "clear-cache") {
        this.emit({ type: "cache-cleared" });
        return;
      }
      if (request.type === "list-cache") {
        this.emit({ type: "cache-inventory", inventory: cacheInventory });
        return;
      }
      if (request.type === "remove-cached-model") {
        this.emit({
          type: "cached-model-removed",
          modelId: request.modelId,
          removedBytes: 2_703_943_680,
        });
        this.emit({
          type: "cache-inventory",
          inventory: {
            ...cacheInventory,
            storedBytes: 0,
            readyBytes: 0,
            missingBytes: TOTAL_DOWNLOAD_BYTES,
            models: cacheInventory.models.map((model) => ({
              ...model,
              storedBytes: 0,
              complete: false,
              partial: false,
            })),
          },
        });
        return;
      }
      const left = new Float32Array([0.1, 0.2, 0.3, 0.4]);
      const right = new Float32Array([-0.1, -0.2, -0.3, -0.4]);
      const complete: CompleteUpdate = {
        type: "complete",
        seed: request.seed,
        audioQuality: request.audioQuality,
        sampler: request.sampler,
        instrumental: !request.lyrics,
        wav: new Uint8Array([82, 73, 70, 70]).buffer,
        left: left.buffer,
        right: right.buffer,
        sampleRate: 48_000,
        durationSeconds: request.durationSeconds,
        latentFrames: request.durationSeconds * 25,
        trace: [],
        timings: { dit: 123 },
        estimatedPeakBytes: 456,
      };
      this.emit(complete);
    });
  }

  emit(update: WorkerUpdate) {
    this.onmessage?.({ data: update } as MessageEvent<WorkerUpdate>);
  }

  terminate() {
    this.terminated = true;
  }
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("published browser API", () => {
  it("reports audio, default-generation, and full cache sizes separately", () => {
    const runtime = new AceStepWebGpu({
      workerFactory: () => new FakeWorker() as unknown as Worker,
    });
    expect(runtime.audioDownloadBytes).toBe(TOTAL_DOWNLOAD_BYTES);
    expect(runtime.highPrecisionAudioDownloadBytes).toBe(
      HIGH_PRECISION_GENERATION_DOWNLOAD_BYTES,
    );
    expect(runtime.totalDownloadBytes).toBe(
      DEFAULT_GENERATION_DOWNLOAD_BYTES,
    );
    expect(runtime.totalBrowserModelBytes).toBe(FULL_MODEL_DOWNLOAD_BYTES);
    runtime.dispose();
  });

  it("uses the pinned Hugging Face XL export by default", () => {
    const assets = getRequiredAssets({
      origin: "https://app.example/",
    });
    for (const model of LOCAL_MODEL_FILES) {
      expect(assets.find((asset) => asset.id === model.id)?.url).toBe(
        `${DEFAULT_MODEL_BASE_URL}${model.fileName}?build=${PIPELINE_BUILD}`,
      );
    }
  });

  it("resolves only fresh XL files through modelBaseUrl", () => {
    const assets = getRequiredAssets({
      origin: "https://app.example/",
      modelBaseUrl: "https://cdn.example/ace-xl",
    });
    expect(assets.reduce((sum, asset) => sum + asset.bytes, 0)).toBe(
      TOTAL_DOWNLOAD_BYTES,
    );
    for (const local of LOCAL_MODEL_FILES) {
      expect(assets.find((asset) => asset.id === local.id)?.url).toBe(
        `https://cdn.example/ace-xl/${local.fileName}?build=${PIPELINE_BUILD}`,
      );
    }
    expect(
      assets.find((asset) => asset.id === "text-encoder:graph")?.url,
    ).toMatch(/^https:\/\/huggingface\.co\//);
  });

  it("resolves the complete INT8 high-precision profile", () => {
    const assets = getRequiredAssets({
      origin: "https://app.example/",
      modelBaseUrl: "https://cdn.example/ace-xl",
      audioQuality: "high",
    });
    expect(assets.reduce((sum, asset) => sum + asset.bytes, 0)).toBe(
      HIGH_PRECISION_GENERATION_DOWNLOAD_BYTES,
    );
    for (const local of HIGH_PRECISION_MODEL_FILES) {
      expect(assets.find((asset) => asset.id === local.id)?.url).toBe(
        `https://cdn.example/ace-xl/${local.fileName}?build=${PIPELINE_BUILD}`,
      );
    }
    expect(
      assets.some((asset) =>
        asset.fileName.includes("xl_turbo_q4"),
      ),
    ).toBe(false);
  });

  it("supports a single self-hosted directory for every model asset", () => {
    const assets = getRequiredAssets({
      origin: "https://app.example/",
      allAssetsBaseUrl: "/static/ace/",
    });
    expect(
      assets.every((asset) =>
        asset.url.startsWith("https://app.example/static/ace/"),
      ),
    ).toBe(true);
  });

  it("returns AudioBuffer, WAV and telemetry through a typed Worker wrapper", async () => {
    vi.stubGlobal("AudioBuffer", FakeAudioBuffer);
    const fakeWorker = new FakeWorker();
    const updates: WorkerUpdate[] = [];
    const runtime = new AceStepWebGpu({
      modelBaseUrl: "https://cdn.example/ace-xl/",
      workerFactory: () => fakeWorker as unknown as Worker,
      onUpdate: (update) => updates.push(update),
    });

    const result = await runtime.generate({
      prompt: "  cinematic instrumental  ",
      seed: 7,
      durationSeconds: 10,
    });

    expect(result.audioBuffer).toBeInstanceOf(FakeAudioBuffer);
    expect(result.channels[0]).toEqual(
      new Float32Array([0.1, 0.2, 0.3, 0.4]),
    );
    expect(result.wav.type).toBe("audio/wav");
    expect(result.durationSeconds).toBe(10);
    expect(result.seed).toBe(7);
    expect(result.sampler).toBe("euler");
    expect(result.instrumental).toBe(true);
    expect(result.timings).toEqual(
      expect.objectContaining({
        dit: 123,
        "pipeline:audio-buffer": expect.any(Number),
        "pipeline:audio-worker": expect.any(Number),
        "pipeline:persistent-storage": expect.any(Number),
        "pipeline:total": expect.any(Number),
      }),
    );
    const progressUpdates = updates.filter(
      (update) => update.type === "progress",
    );
    expect(progressUpdates[0]).toMatchObject({
      type: "progress",
      operation: "generate",
      progress: 0,
    });
    expect(progressUpdates.at(-1)).toMatchObject({
      type: "progress",
      operation: "generate",
      progress: 1,
    });
    expect(
      progressUpdates.every(
        (update, index) =>
          update.progress >= 0 &&
          update.progress <= 1 &&
          (index === 0 ||
            update.progress >= progressUpdates[index - 1]!.progress),
      ),
    ).toBe(true);
    expect(runtime.progress).toBe(1);
    expect(updates.some((update) => update.type === "complete")).toBe(true);
    expect(fakeWorker.terminated).toBe(true);
    expect(fakeWorker.requests).toHaveLength(1);
    expect(fakeWorker.requests[0]).toMatchObject({
      type: "start",
      prompt: "cinematic instrumental",
      lyrics: "",
      vocalLanguage: "unknown",
      seed: 7,
      durationSeconds: 10,
      sampler: "euler",
      dcw: {
        enabled: false,
        mode: "double",
        scaler: 0.05,
        highScaler: 0.02,
      },
      assets: {
        modelBaseUrl: "https://cdn.example/ace-xl/",
      },
    });
    const turboRequest = fakeWorker.requests[0];
    expect(
      turboRequest && "semanticCodeIds" in turboRequest
        ? turboRequest.semanticCodeIds
        : undefined,
    ).toBeUndefined();
    expect(result.plan).toBeUndefined();
    expect(updates).toContainEqual({
      type: "diagnostic",
      key: "Turbo model path",
      value:
        "direct XL Turbo text/lyric conditioning · 4B semantic planner skipped",
    });
    runtime.dispose();
  });

  it("hands a qualified precomputed plan directly to the audio Worker", async () => {
    vi.stubGlobal("AudioBuffer", FakeAudioBuffer);
    const fakeWorker = new FakeWorker();
    const runtime = new AceStepWebGpu({
      workerFactory: () => fakeWorker as unknown as Worker,
      languageWorkerFactory: () => {
        throw new Error("Precomputed plans must skip the language Worker.");
      },
    });
    const semanticCodeIds = Array.from(
      { length: 50 },
      (_, index) => index,
    );
    const plannerMetadata = {
      bpm: 100,
      caption: "qualified FP32 synthwave",
      durationSeconds: 10,
      keyScale: "F major",
      language: "en",
      timeSignature: 4 as const,
    };
    const result = await runtime.generate({
      prompt: "synthwave vocal",
      lyrics: "[Verse]\nNeon light",
      vocalLanguage: "en",
      durationSeconds: 10,
      seed: 42,
      semanticCodeIds,
      plannerMetadata,
    });

    expect(fakeWorker.requests).toHaveLength(1);
    expect(fakeWorker.requests[0]).toMatchObject({
      type: "start",
      prompt: plannerMetadata.caption,
      semanticCodeIds,
      plannerMetadata,
    });
    expect(result.plan).toMatchObject({
      model: "external-precomputed-planner",
      semanticCodeCount: 50,
    });
  });

  it("rejects incomplete or malformed precomputed planner conditioning", async () => {
    const runtime = new AceStepWebGpu({
      workerFactory: () => new FakeWorker() as unknown as Worker,
    });
    const metadata = {
      bpm: 100,
      caption: "test",
      durationSeconds: 10,
      keyScale: "F major",
      language: "en",
      timeSignature: 4 as const,
    };
    await expect(
      runtime.generate({
        prompt: "test",
        durationSeconds: 10,
        semanticCodeIds: Array(50).fill(0),
      }),
    ).rejects.toThrow(/supplied together/);
    await expect(
      runtime.generate({
        prompt: "test",
        durationSeconds: 10,
        semanticCodeIds: [0, 1, 64_000],
        plannerMetadata: metadata,
      }),
    ).rejects.toThrow(/exactly 50/);
    runtime.dispose();
  });

  it("passes lyrics, language, Heun and DCW settings to inference", async () => {
    vi.stubGlobal("AudioBuffer", FakeAudioBuffer);
    const fakeWorker = new FakeWorker();
    const runtime = new AceStepWebGpu({
      workerFactory: () => fakeWorker as unknown as Worker,
    });

    const result = await runtime.generate({
      prompt: "upbeat electropop with a bright female lead vocal",
      lyrics: "[Verse]\nWe light the dark\n\n[Chorus]\nSing it again",
      vocalLanguage: "en",
      seed: 99,
      durationSeconds: 30,
      sampler: "heun",
      dcw: {
        enabled: true,
        mode: "double",
        scaler: 0.04,
        highScaler: 0.015,
      },
    });

    expect(result.instrumental).toBe(false);
    expect(result.sampler).toBe("heun");
    expect(
      fakeWorker.requests.find((request) => request.type === "start"),
    ).toMatchObject({
      type: "start",
      lyrics: "[Verse]\nWe light the dark\n\n[Chorus]\nSing it again",
      vocalLanguage: "en",
      sampler: "heun",
      dcw: {
        enabled: true,
        mode: "double",
        scaler: 0.04,
        highScaler: 0.015,
      },
    });
  });

  it("writes lyrics in an isolated Qwen Worker", async () => {
    const fakeWorker = new FakeWorker();
    const runtime = new AceStepWebGpu({
      languageWorkerFactory: () => fakeWorker as unknown as Worker,
    });

    const result = await runtime.writeLyrics({
      prompt: "An upbeat neon synth-pop anthem in English",
      seed: 17,
      durationSeconds: 30,
      maxWords: 66,
    });

    expect(result.lyrics).toContain("[Chorus]");
    expect(result.model).toBe(DEFAULT_LYRICS_MODEL);
    expect(fakeWorker.requests[0]).toEqual({
      type: "write-lyrics",
      prompt: "An upbeat neon synth-pop anthem in English",
      seed: 17,
      durationSeconds: 30,
      maxWords: 66,
      modelId: DEFAULT_LYRICS_MODEL,
      revision: DEFAULT_LYRICS_MODEL_REVISION,
    });
    expect(fakeWorker.terminated).toBe(true);
  });

  it("runs Qwen, the ACE planner, and audio inference in isolated Workers", async () => {
    vi.stubGlobal("AudioBuffer", FakeAudioBuffer);
    const workers: FakeWorker[] = [];
    const runtime = new AceStepWebGpu({
      workerFactory: () => {
        const worker = new FakeWorker();
        workers.push(worker);
        return worker as unknown as Worker;
      },
    });

    const result = await runtime.generate({
      prompt: "upbeat neon synth-pop with a clear lead singer",
      plannerQuality: "high-quality",
      writeLyrics: true,
      vocalLanguage: "en",
      seed: 33,
      durationSeconds: 30,
    });

    expect(workers).toHaveLength(3);
    expect(workers[0]?.requests[0]?.type).toBe("write-lyrics");
    expect(workers[1]?.requests[0]).toMatchObject({
      type: "plan-music",
      lyrics: expect.stringContaining("[Chorus]"),
      seed: 33,
    });
    expect(workers[2]?.requests[0]).toMatchObject({
      type: "start",
      lyrics: expect.stringContaining("[Chorus]"),
      seed: 33,
      semanticCodeIds: expect.any(Array),
      plannerMetadata: expect.objectContaining({
        durationSeconds: 30,
        timeSignature: 4,
      }),
    });
    expect(result.instrumental).toBe(false);
    expect(result.lyrics).toContain("[Chorus]");
  });

  it("selects the INT8-weight / FP32-compute planner explicitly", async () => {
    const fakeWorker = new FakeWorker();
    const runtime = new AceStepWebGpu({
      languageWorkerFactory: () => fakeWorker as unknown as Worker,
    });

    const result = await runtime.planMusic({
      prompt: "polished synthwave song",
      plannerQuality: "high-quality",
      audioQuality: "high",
      seed: 42,
      durationSeconds: 10,
    });

    expect(fakeWorker.requests[0]).toMatchObject({
      type: "plan-music",
      plannerQuality: "high-quality",
      audioQuality: "high",
      highQualityModelId: DEFAULT_HIGH_QUALITY_PLANNER_MODEL,
      highQualityRevision:
        DEFAULT_HIGH_QUALITY_PLANNER_MODEL_REVISION,
      assets: {
        modelBaseUrl: DEFAULT_MODEL_BASE_URL,
      },
    });
    expect(result.plannerQuality).toBe("high-quality");
  });

  it("expands automatic vocal duration before planning and audio inference", async () => {
    vi.stubGlobal("AudioBuffer", FakeAudioBuffer);
    const workers: FakeWorker[] = [];
    const runtime = new AceStepWebGpu({
      workerFactory: () => {
        const worker = new FakeWorker();
        workers.push(worker);
        return worker as unknown as Worker;
      },
    });
    const lyrics = `[Verse]\n${Array.from(
      { length: 66 },
      (_, index) => `word${index}`,
    ).join(" ")}`;

    const result = await runtime.generate({
      prompt: "clear expressive vocal synth-pop",
      plannerQuality: "high-quality",
      lyrics,
      autoDuration: true,
      durationSeconds: 30,
    });

    expect(workers[0]?.requests[0]).toMatchObject({
      type: "plan-music",
      autoDuration: true,
      durationSeconds: 30,
      recommendedDurationSeconds: 60,
    });
    expect(workers[1]?.requests[0]).toMatchObject({
      type: "start",
      durationSeconds: 60,
      plannerMetadata: expect.objectContaining({
        durationSeconds: 60,
        durationSource: "recommended",
      }),
    });
    expect(result.durationSeconds).toBe(60);
  });

  it("rejects contradictory instrumental captions before downloading models", async () => {
    const fakeWorker = new FakeWorker();
    const runtime = new AceStepWebGpu({
      workerFactory: () => fakeWorker as unknown as Worker,
    });
    await expect(
      runtime.generate({
        prompt: "Warm synthwave instrumental with no singing",
        lyrics: "[Verse]\nPlease sing this line",
        vocalLanguage: "en",
      }),
    ).rejects.toThrow(/requests an instrumental track/);
    expect(fakeWorker.requests).toEqual([]);
  });

  it("keeps Euler SDE instrumental-only until vocal quality passes", async () => {
    const fakeWorker = new FakeWorker();
    const runtime = new AceStepWebGpu({
      workerFactory: () => fakeWorker as unknown as Worker,
    });
    await expect(
      runtime.generate({
        prompt: DEFAULT_VOCAL_PROMPT,
        lyrics: "[Chorus]\nPlease sing this line",
        sampler: "euler-sde",
      }),
    ).rejects.toThrow(/limited to instrumental generation/);
    expect(fakeWorker.requests).toEqual([]);
  });

  it("generates a seed batch sequentially and reports progress", async () => {
    vi.stubGlobal("AudioBuffer", FakeAudioBuffer);
    const workers: FakeWorker[] = [];
    const updates: WorkerUpdate[] = [];
    const runtime = new AceStepWebGpu({
      workerFactory: () => {
        const worker = new FakeWorker();
        workers.push(worker);
        return worker as unknown as Worker;
      },
      onUpdate: (update) => updates.push(update),
    });

    const results = await runtime.generateBatch({
      prompt: "ambient piano",
      seeds: [10, 11, 12],
      durationSeconds: 10,
      sampler: "euler-sde",
    });

    expect(results.map((result) => result.seed)).toEqual([10, 11, 12]);
    expect(workers).toHaveLength(3);
    expect(
      workers
        .map((worker) => worker.requests[0])
        .filter((request) => request?.type === "start")
        .map((request) => request.seed),
    ).toEqual([10, 11, 12]);
    expect(
      updates
        .filter((update) => update.type === "batch-progress")
        .map((update) => `${update.status}:${update.seed}`),
    ).toEqual([
      "started:10",
      "complete:10",
      "started:11",
      "complete:11",
      "started:12",
      "complete:12",
    ]);
    const overall = updates.filter(
      (update) => update.type === "progress",
    );
    expect(overall[0]).toMatchObject({
      operation: "generate-batch",
      progress: 0,
    });
    expect(overall.at(-1)).toMatchObject({
      operation: "generate-batch",
      progress: 1,
    });
    expect(
      overall.every(
        (update, index) =>
          index === 0 ||
          update.progress >= overall[index - 1]!.progress,
      ),
    ).toBe(true);
  });

  it("validates every batch seed before starting any generation", async () => {
    const workers: FakeWorker[] = [];
    const runtime = new AceStepWebGpu({
      workerFactory: () => {
        const worker = new FakeWorker();
        workers.push(worker);
        return worker as unknown as Worker;
      },
    });
    await expect(
      runtime.generateBatch({
        prompt: "ambient piano",
        seeds: [1, -1],
      }),
    ).rejects.toThrow(/Every sequential batch seed/);
    expect(workers).toHaveLength(0);
  });

  it("clears the persistent cache through an isolated Worker operation", async () => {
    const fakeWorker = new FakeWorker();
    const runtime = new AceStepWebGpu({
      workerFactory: () => fakeWorker as unknown as Worker,
    });
    await runtime.clearCache();
    expect(fakeWorker.requests).toEqual([{ type: "clear-cache" }]);
    expect(fakeWorker.terminated).toBe(true);
  });

  it("lists cached model components with quota information", async () => {
    const fakeWorker = new FakeWorker();
    const runtime = new AceStepWebGpu({
      workerFactory: () => fakeWorker as unknown as Worker,
    });
    const inventory = await runtime.listCachedModels();
    expect(inventory.origin).toBe("https://app.example");
    expect(inventory.models[0]).toMatchObject({
      id: "dit",
      partial: true,
    });
    expect(fakeWorker.requests[0]).toMatchObject({
      type: "list-cache",
      assets: {
        modelBaseUrl: DEFAULT_MODEL_BASE_URL,
      },
    });
  });

  it("removes one cached model component and returns fresh inventory", async () => {
    const fakeWorker = new FakeWorker();
    const updates: WorkerUpdate[] = [];
    const runtime = new AceStepWebGpu({
      workerFactory: () => fakeWorker as unknown as Worker,
      onUpdate: (update) => updates.push(update),
    });
    const inventory = await runtime.removeCachedModel("dit");
    expect(inventory.models[0]?.storedBytes).toBe(0);
    expect(fakeWorker.requests[0]).toMatchObject({
      type: "remove-cached-model",
      modelId: "dit",
    });
    expect(updates.some((update) => update.type === "cached-model-removed")).toBe(
      true,
    );
  });

  it("accepts the isolated planner and lyric-writer cache identifiers", async () => {
    for (const modelId of [
      "music-planner",
      "music-planner-high-quality",
      "lyrics-writer",
    ]) {
      const fakeWorker = new FakeWorker();
      const runtime = new AceStepWebGpu({
        workerFactory: () => fakeWorker as unknown as Worker,
      });
      await runtime.removeCachedModel(modelId);
      expect(fakeWorker.requests[0]).toMatchObject({
        type: "remove-cached-model",
        modelId,
      });
    }
  });

  it("rejects unknown cache component identifiers before creating a Worker", async () => {
    const fakeWorker = new FakeWorker();
    const runtime = new AceStepWebGpu({
      workerFactory: () => fakeWorker as unknown as Worker,
    });
    await expect(runtime.removeCachedModel("not-a-model")).rejects.toThrow(
      "Unknown cached model component",
    );
    expect(fakeWorker.requests).toEqual([]);
  });
});
