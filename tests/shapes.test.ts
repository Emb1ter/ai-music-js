import { describe, expect, it } from "vitest";
import {
  DEFAULT_DURATION_SECONDS,
  DIT_PATCH_SIZE,
  HIGH_PRECISION_MODEL_GRAPHS,
  HIGH_QUALITY_TOTAL_DOWNLOAD_BYTES,
  LATENT_CHANNELS,
  LATENT_FRAME_RATE,
  MAX_DURATION_SECONDS,
  MIN_DURATION_SECONDS,
  MODEL_GRAPHS,
  SAMPLE_RATE,
  TOTAL_DOWNLOAD_BYTES,
  VAE_UPSAMPLE_FACTOR,
  buildCaptionPrompt,
  buildLyricPrompt,
  durationToAudioFrames,
  durationToLatentFrames,
  hasVocalPromptConflict,
  validateDurationSeconds,
} from "../lib/model-manifest";
import {
  assertShape,
  padFrameMajorTensor,
} from "../lib/tensor-diagnostics";
import {
  HIGH_QUALITY_PLANNER_MODEL_ASSETS,
  HIGH_QUALITY_PLANNER_MODEL_DOWNLOAD_BYTES,
  LANGUAGE_MODEL_COMPONENTS,
  LANGUAGE_MODEL_DOWNLOAD_BYTES,
  LYRICS_MODEL_DOWNLOAD_BYTES,
  PLANNER_MODEL_DOWNLOAD_BYTES,
} from "../lib/language-model-manifest";

describe("variable-duration tensor contracts", () => {
  it("maps 48 kHz audio to the 25 Hz latent clock exactly", () => {
    for (const duration of [10, 30, 60, 120]) {
      const latentFrames = durationToLatentFrames(duration);
      const audioFrames = durationToAudioFrames(duration);
      expect(latentFrames).toBe(duration * LATENT_FRAME_RATE);
      expect(audioFrames).toBe(duration * SAMPLE_RATE);
      expect(latentFrames * VAE_UPSAMPLE_FACTOR).toBe(audioFrames);
    }
  });

  it("accepts every pipeline boundary shape", () => {
    for (const duration of [10, 30, 60, 120]) {
      const latentFrames = durationToLatentFrames(duration);
      const audioFrames = durationToAudioFrames(duration);
      expect(() =>
        assertShape(
          [1, latentFrames, LATENT_CHANNELS],
          [1, duration * 25, 64],
          "initial latent",
        ),
      ).not.toThrow();
      expect(() =>
        assertShape(
          [1, latentFrames, 128],
          [1, duration * 25, 128],
          "context latent",
        ),
      ).not.toThrow();
      expect(() =>
        assertShape(
          [1, 2, audioFrames],
          [1, 2, duration * 48_000],
          "waveform",
        ),
      ).not.toThrow();
    }
  });

  it("rejects a transposed DiT latent", () => {
    const latentFrames = durationToLatentFrames(60);
    expect(() =>
      assertShape(
        [1, LATENT_CHANNELS, latentFrames],
        [1, latentFrames, LATENT_CHANNELS],
        "DiT latent",
      ),
    ).toThrow(/shape mismatch/);
  });

  it("pads odd AI-selected durations to the DiT patch boundary", () => {
    const frames = durationToLatentFrames(107);
    const latent = new Float32Array(frames * LATENT_CHANNELS).fill(1);
    const padded = padFrameMajorTensor(
      latent,
      frames,
      LATENT_CHANNELS,
      DIT_PATCH_SIZE,
    );

    expect(frames).toBe(2_675);
    expect(padded.frames).toBe(2_676);
    expect(padded.data.length).toBe(2_676 * LATENT_CHANNELS);
    expect(padded.data.slice(0, latent.length)).toEqual(latent);
    expect(padded.data.slice(latent.length).every((value) => value === 0)).toBe(
      true,
    );
  });

  it("does not copy even DiT frame sequences", () => {
    const frames = durationToLatentFrames(30);
    const latent = new Float32Array(frames * LATENT_CHANNELS);
    const padded = padFrameMajorTensor(
      latent,
      frames,
      LATENT_CHANNELS,
      DIT_PATCH_SIZE,
    );

    expect(padded).toEqual({ data: latent, frames });
    expect(padded.data).toBe(latent);
  });

  it("validates the guarded desktop range", () => {
    expect(validateDurationSeconds(DEFAULT_DURATION_SECONDS)).toBe(10);
    expect(validateDurationSeconds(MIN_DURATION_SECONDS)).toBe(10);
    expect(validateDurationSeconds(MAX_DURATION_SECONDS)).toBe(120);
    for (const invalid of [9, 121, 10.5, Number.NaN]) {
      expect(() => validateDurationSeconds(invalid)).toThrow(/Duration/);
    }
  });

  it("packs the selected duration into the text caption", () => {
    expect(buildCaptionPrompt("ambient instrumental", 60)).toContain(
      "duration: 60 seconds",
    );
  });

  it("matches backend semantic-code prompt conditioning", () => {
    const prompt = buildCaptionPrompt("neon vocal synth-pop", 30, {
      bpm: 118,
      keyScale: "F# minor",
      timeSignature: 4,
    });
    expect(prompt).toContain(
      "Generate audio semantic tokens based on the given conditions:",
    );
    expect(prompt).toContain("- bpm: 118");
    expect(prompt).toContain("- timesignature: 4/4");
    expect(prompt).toContain("- keyscale: F# minor");
  });

  it("formats user lyrics exactly as ACE-Step expects", () => {
    expect(buildLyricPrompt("[Verse]\nHello world", "en")).toBe(
      "# Languages\nen\n\n# Lyric\n[Verse]\nHello world<|endoftext|>",
    );
    expect(buildLyricPrompt()).toContain(
      "# Lyric\n[Instrumental]<|endoftext|>",
    );
  });

  it("detects captions that contradict supplied vocal lyrics", () => {
    const lyrics = "[Verse]\nSing this line";
    expect(
      hasVocalPromptConflict(
        "Warm analog synthwave instrumental, polished mix",
        lyrics,
      ),
    ).toBe(true);
    expect(
      hasVocalPromptConflict(
        "Mostly instrumental synthwave with a clear lead vocal",
        lyrics,
      ),
    ).toBe(false);
    expect(
      hasVocalPromptConflict(
        "Warm analog synthwave instrumental",
        "[Instrumental]",
      ),
    ).toBe(false);
  });
});

describe("pinned model manifest", () => {
  it("contains the five audio graphs plus the semantic detokenizer", () => {
    expect(MODEL_GRAPHS.map((graph) => graph.id)).toEqual([
      "text-encoder",
      "lyric-embedding",
      "condition-encoder",
      "audio-code-detokenizer",
      "dit",
      "vae",
    ]);
  });

  it("records the audited cold-download size", () => {
    expect(TOTAL_DOWNLOAD_BYTES).toBe(5_626_494_229);
    expect(HIGH_QUALITY_TOTAL_DOWNLOAD_BYTES).toBe(8_004_092_572);
    expect(PLANNER_MODEL_DOWNLOAD_BYTES).toBe(3_628_429_574);
    expect(HIGH_QUALITY_PLANNER_MODEL_DOWNLOAD_BYTES).toBe(
      4_633_150_982,
    );
    expect(LYRICS_MODEL_DOWNLOAD_BYTES).toBe(489_166_749);
    expect(LANGUAGE_MODEL_DOWNLOAD_BYTES).toBe(8_750_747_305);
    expect(TOTAL_DOWNLOAD_BYTES + LANGUAGE_MODEL_DOWNLOAD_BYTES).toBe(
      14_377_241_534,
    );
    expect(LANGUAGE_MODEL_COMPONENTS.map((model) => model.id)).toEqual([
      "music-planner",
      "music-planner-high-quality",
      "lyrics-writer",
    ]);
  });

  it("uses the browser-qualified FP32 WebGPU VAE export", () => {
    const vae = MODEL_GRAPHS.find((graph) => graph.id === "vae");
    expect(vae?.label).toContain("FP32 WebGPU");
    expect(vae?.graph.fileName).toBe("vae_decoder_fp32.onnx");
    expect(vae?.graph.bytes).toBe(1_076_526);
    expect(vae?.weights).toEqual([
      expect.objectContaining({
        fileName: "vae_decoder_fp32.onnx.data",
        bytes: 337_707_008,
      }),
    ]);
    expect(vae?.graph.url).toContain("/models/vae_decoder_fp32.onnx");
  });

  it("pins the browser-qualified five-shard hybrid planner body", () => {
    const bodyAssets = HIGH_QUALITY_PLANNER_MODEL_ASSETS.filter(
      (asset) => asset.fileName.startsWith("int8-fp32/onnx/"),
    );
    expect(bodyAssets.map((asset) => asset.fileName)).toEqual([
      "int8-fp32/onnx/model.onnx",
      "int8-fp32/onnx/model.onnx.data.0",
      "int8-fp32/onnx/model.onnx.data.1",
      "int8-fp32/onnx/model.onnx.data.2",
      "int8-fp32/onnx/model.onnx.data.3",
      "int8-fp32/onnx/model.onnx.data.4",
    ]);
    expect(bodyAssets.reduce((sum, asset) => sum + asset.bytes, 0)).toBe(
      3_939_389_998,
    );
    expect(
      bodyAssets
        .filter((asset) => asset.role === "weights")
        .every((asset) => asset.bytes < 1_000_000_000),
    ).toBe(true);
  });

  it("keeps every browser-mounted external-data file below 2 GB", () => {
    for (const graph of [
      ...MODEL_GRAPHS,
      ...HIGH_PRECISION_MODEL_GRAPHS,
    ]) {
      for (const weights of graph.weights) {
        expect(weights.bytes).toBeLessThan(2_000_000_000);
      }
    }
    const ditWeights = MODEL_GRAPHS.find(
      (graph) => graph.id === "dit",
    )?.weights;
    expect(ditWeights).toHaveLength(3);
    expect(ditWeights?.every((asset) => asset.bytes < 1_000_000_000)).toBe(
      true,
    );
    const highPrecisionDitWeights = HIGH_PRECISION_MODEL_GRAPHS.find(
      (graph) => graph.id === "dit",
    )?.weights;
    expect(highPrecisionDitWeights).toHaveLength(6);
    expect(
      highPrecisionDitWeights?.every(
        (asset) => asset.bytes < 1_000_000_000,
      ),
    ).toBe(true);
  });
});
