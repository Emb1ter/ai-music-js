import { describe, expect, it } from "vitest";
import {
  DEFAULT_DURATION_SECONDS,
  LATENT_CHANNELS,
  LATENT_FRAME_RATE,
  MAX_DURATION_SECONDS,
  MIN_DURATION_SECONDS,
  MODEL_GRAPHS,
  SAMPLE_RATE,
  TOTAL_DOWNLOAD_BYTES,
  VAE_UPSAMPLE_FACTOR,
  buildCaptionPrompt,
  durationToAudioFrames,
  durationToLatentFrames,
  validateDurationSeconds,
} from "../lib/model-manifest";
import { assertShape } from "../lib/tensor-diagnostics";

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
});

describe("pinned model manifest", () => {
  it("contains the five no-LM graphs", () => {
    expect(MODEL_GRAPHS.map((graph) => graph.id)).toEqual([
      "text-encoder",
      "lyric-embedding",
      "condition-encoder",
      "dit",
      "vae",
    ]);
  });

  it("records the audited cold-download size", () => {
    expect(TOTAL_DOWNLOAD_BYTES).toBe(5_245_621_594);
  });

  it("keeps every browser-mounted external-data file below 2 GB", () => {
    for (const graph of MODEL_GRAPHS) {
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
  });
});
