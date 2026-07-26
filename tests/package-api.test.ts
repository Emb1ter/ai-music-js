import { afterEach, describe, expect, it, vi } from "vitest";
import {
  AceStepWebGpu,
  DEFAULT_MODEL_BASE_URL,
  LOCAL_MODEL_FILES,
  TOTAL_DOWNLOAD_BYTES,
  getRequiredAssets,
} from "../package-src/index";
import type {
  CompleteUpdate,
  WorkerRequest,
  WorkerUpdate,
} from "../lib/worker-protocol";

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
      const left = new Float32Array([0.1, 0.2, 0.3, 0.4]);
      const right = new Float32Array([-0.1, -0.2, -0.3, -0.4]);
      const complete: CompleteUpdate = {
        type: "complete",
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
    expect(result.timings).toEqual({ dit: 123 });
    expect(updates.at(-1)?.type).toBe("complete");
    expect(fakeWorker.terminated).toBe(true);
    expect(fakeWorker.requests[0]).toMatchObject({
      type: "start",
      prompt: "cinematic instrumental",
      seed: 7,
      durationSeconds: 10,
      assets: {
        modelBaseUrl: "https://cdn.example/ace-xl/",
      },
    });
    runtime.dispose();
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
});
