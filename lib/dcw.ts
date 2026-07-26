import type {
  ResolvedDcwOptions,
} from "./generation-options";

const INV_SQRT_2 = Math.fround(1 / Math.sqrt(2));

const fadd = (left: number, right: number) =>
  Math.fround(Math.fround(left) + Math.fround(right));
const fsub = (left: number, right: number) =>
  Math.fround(Math.fround(left) - Math.fround(right));
const fmul = (left: number, right: number) =>
  Math.fround(Math.fround(left) * Math.fround(right));

/**
 * Native single-level Haar DCW over a flattened [T,C] ACE-Step latent.
 * This mirrors the official MLX implementation, including right zero-padding
 * for an odd time dimension and float32 arithmetic.
 */
export const applyDcw = (
  xNext: Float32Array,
  denoised: Float32Array,
  timeFrames: number,
  channels: number,
  currentTimestep: number,
  options: ResolvedDcwOptions,
  output = new Float32Array(xNext.length),
) => {
  if (
    xNext.length !== denoised.length ||
    output.length !== xNext.length ||
    xNext.length !== timeFrames * channels
  ) {
    throw new RangeError(
      "DCW tensors must have identical flattened [time, channel] shapes.",
    );
  }
  if (!options.enabled) {
    output.set(xNext);
    return output;
  }

  const timestep = Math.fround(currentTimestep);
  const lowScale = fmul(timestep, options.scaler);
  const highScale = fmul(
    fsub(1, timestep),
    options.mode === "double"
      ? options.highScaler
      : options.scaler,
  );

  if (options.mode === "pix") {
    for (let index = 0; index < xNext.length; index += 1) {
      output[index] = fadd(
        xNext[index],
        fmul(options.scaler, fsub(xNext[index], denoised[index])),
      );
    }
    return output;
  }

  const correctLow =
    (options.mode === "low" || options.mode === "double") &&
    lowScale !== 0;
  const correctHigh =
    (options.mode === "high" || options.mode === "double") &&
    highScale !== 0;
  if (!correctLow && !correctHigh) {
    output.set(xNext);
    return output;
  }

  for (let frame = 0; frame < timeFrames; frame += 2) {
    const hasOdd = frame + 1 < timeFrames;
    for (let channel = 0; channel < channels; channel += 1) {
      const evenIndex = frame * channels + channel;
      const oddIndex = (frame + 1) * channels + channel;
      const xEven = xNext[evenIndex] ?? 0;
      const xOdd = hasOdd ? (xNext[oddIndex] ?? 0) : 0;
      const yEven = denoised[evenIndex] ?? 0;
      const yOdd = hasOdd ? (denoised[oddIndex] ?? 0) : 0;

      let xLow = fmul(fadd(xEven, xOdd), INV_SQRT_2);
      let xHigh = fmul(fsub(xEven, xOdd), INV_SQRT_2);
      const yLow = fmul(fadd(yEven, yOdd), INV_SQRT_2);
      const yHigh = fmul(fsub(yEven, yOdd), INV_SQRT_2);

      if (correctLow) {
        xLow = fadd(xLow, fmul(lowScale, fsub(xLow, yLow)));
      }
      if (correctHigh) {
        xHigh = fadd(
          xHigh,
          fmul(highScale, fsub(xHigh, yHigh)),
        );
      }

      output[evenIndex] = fmul(
        fadd(xLow, xHigh),
        INV_SQRT_2,
      );
      if (hasOdd) {
        output[oddIndex] = fmul(
          fsub(xLow, xHigh),
          INV_SQRT_2,
        );
      }
    }
  }
  return output;
};
