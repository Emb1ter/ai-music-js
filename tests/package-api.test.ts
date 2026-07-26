import { afterEach, describe, expect, it, vi } from "vitest";
import {
  AceStepWebGpu,
  DEFAULT_MODEL_BASE_URL,
  LOCAL_MODEL_FILES,
  TOTAL_DOWNLOAD_BYTES,
  getRequiredAssets,
} from "../package-src/index";
import type {
  CacheInventory,
  CompleteUpdate,
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
  readonly requests: WorkerRequest[] = [];
  terminated = false;

  postMessage(request: WorkerRequest) {
    this.requests.push(request);
    queueMicrotask(() => {
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
  it("uses the pinned Hugging Face XL export by default", () => {
    const assets = getRequiredAssets({
      origin: "https://app.example/",
    });
    for (const model of LOCAL_MODEL_FILES) {
      expect(assets.find((asset) => asset.id === model.id)?.url).toBe(
        `${DEFAULT_MODEL_BASE_URL}${model.fileName}?build=2026-07-25-xl-turbo-q4-chunked`,
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
        `https://cdn.example/ace-xl/${local.fileName}?build=2026-07-25-xl-turbo-q4-chunked`,
      );
    }
    expect(
      assets.find((asset) => asset.id === "text-encoder:graph")?.url,
    ).toMatch(/^https:\/\/huggingface\.co\//);
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
    expect(result.timings).toEqual({ dit: 123 });
    expect(updates.at(-1)?.type).toBe("complete");
    expect(fakeWorker.terminated).toBe(true);
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
    expect(fakeWorker.requests[0]).toMatchObject({
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
      workers.map((worker) => {
        const request = worker.requests[0];
        return request?.type === "start" ? request.seed : undefined;
      }),
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
