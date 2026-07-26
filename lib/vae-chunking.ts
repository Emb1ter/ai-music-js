export const VAE_CHUNK_CORE_FRAMES = 250;
export const VAE_CHUNK_CONTEXT_FRAMES = 50;

export type VaeDecodeChunk = {
  index: number;
  coreStartFrame: number;
  coreEndFrame: number;
  inputStartFrame: number;
  inputEndFrame: number;
  cropStartFrame: number;
};

/**
 * Splits a latent sequence into ten-second decode cores with two seconds of
 * convolution context on either side. Context is cropped after each VAE run,
 * so every output sample is written exactly once.
 */
export const createVaeDecodePlan = (
  latentFrames: number,
  coreFrames = VAE_CHUNK_CORE_FRAMES,
  contextFrames = VAE_CHUNK_CONTEXT_FRAMES,
): VaeDecodeChunk[] => {
  for (const [name, value] of [
    ["latentFrames", latentFrames],
    ["coreFrames", coreFrames],
    ["contextFrames", contextFrames],
  ] as const) {
    if (!Number.isInteger(value) || value < (name === "contextFrames" ? 0 : 1)) {
      throw new RangeError(`${name} must be a valid whole-frame count.`);
    }
  }

  const chunks: VaeDecodeChunk[] = [];
  for (
    let coreStartFrame = 0;
    coreStartFrame < latentFrames;
    coreStartFrame += coreFrames
  ) {
    const coreEndFrame = Math.min(
      latentFrames,
      coreStartFrame + coreFrames,
    );
    const inputStartFrame = Math.max(
      0,
      coreStartFrame - contextFrames,
    );
    const inputEndFrame = Math.min(
      latentFrames,
      coreEndFrame + contextFrames,
    );
    chunks.push({
      index: chunks.length,
      coreStartFrame,
      coreEndFrame,
      inputStartFrame,
      inputEndFrame,
      cropStartFrame: coreStartFrame - inputStartFrame,
    });
  }
  return chunks;
};
