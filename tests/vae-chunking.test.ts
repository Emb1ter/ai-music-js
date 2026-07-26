import { describe, expect, it } from "vitest";
import {
  VAE_CHUNK_CONTEXT_FRAMES,
  VAE_CHUNK_CORE_FRAMES,
  createVaeDecodePlan,
} from "../lib/vae-chunking";

describe("memory-bounded VAE decode planning", () => {
  it("keeps the verified ten-second path as one exact chunk", () => {
    expect(createVaeDecodePlan(250)).toEqual([
      {
        index: 0,
        coreStartFrame: 0,
        coreEndFrame: 250,
        inputStartFrame: 0,
        inputEndFrame: 250,
        cropStartFrame: 0,
      },
    ]);
  });

  it("covers a 60-second latent exactly once with bounded input windows", () => {
    const chunks = createVaeDecodePlan(1_500);
    expect(chunks).toHaveLength(6);
    expect(chunks[0]).toMatchObject({
      coreStartFrame: 0,
      coreEndFrame: VAE_CHUNK_CORE_FRAMES,
      inputStartFrame: 0,
      inputEndFrame:
        VAE_CHUNK_CORE_FRAMES + VAE_CHUNK_CONTEXT_FRAMES,
    });
    expect(chunks.at(-1)).toMatchObject({
      coreStartFrame: 1_250,
      coreEndFrame: 1_500,
      inputStartFrame: 1_200,
      inputEndFrame: 1_500,
      cropStartFrame: VAE_CHUNK_CONTEXT_FRAMES,
    });

    for (const [index, chunk] of chunks.entries()) {
      expect(chunk.index).toBe(index);
      expect(chunk.coreStartFrame).toBe(index * VAE_CHUNK_CORE_FRAMES);
      expect(chunk.coreEndFrame - chunk.coreStartFrame).toBe(
        VAE_CHUNK_CORE_FRAMES,
      );
      expect(chunk.inputEndFrame - chunk.inputStartFrame).toBeLessThanOrEqual(
        VAE_CHUNK_CORE_FRAMES + VAE_CHUNK_CONTEXT_FRAMES * 2,
      );
      if (index > 0) {
        expect(chunks[index - 1]?.coreEndFrame).toBe(
          chunk.coreStartFrame,
        );
      }
    }
  });

  it("handles a final partial core without output gaps", () => {
    const chunks = createVaeDecodePlan(775);
    expect(chunks.at(-1)).toMatchObject({
      coreStartFrame: 750,
      coreEndFrame: 775,
      inputStartFrame: 700,
      inputEndFrame: 775,
      cropStartFrame: 50,
    });
  });

  it("rejects invalid frame counts", () => {
    expect(() => createVaeDecodePlan(0)).toThrow(RangeError);
    expect(() => createVaeDecodePlan(250, 0)).toThrow(RangeError);
    expect(() => createVaeDecodePlan(250, 250, -1)).toThrow(RangeError);
  });
});
