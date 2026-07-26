export type EulerStep = {
  index: number;
  current: number;
  next: number;
  delta: number;
};

export function createTurboSchedule(
  steps = 8,
  shift = 3,
): EulerStep[] {
  if (!Number.isInteger(steps) || steps < 1 || steps > 20) {
    throw new RangeError("Turbo inference steps must be an integer from 1 to 20.");
  }
  if (!Number.isFinite(shift) || shift <= 0) {
    throw new RangeError("Flow shift must be a positive finite number.");
  }

  const shift32 = Math.fround(shift);
  const timesteps = Array.from({ length: steps }, (_, index) => {
    const raw = Math.fround(1 - index / steps);
    if (shift === 1) {
      return raw;
    }
    const numerator = Math.fround(shift32 * raw);
    const denominator = Math.fround(
      1 + Math.fround(Math.fround(shift32 - 1) * raw),
    );
    return Math.fround(numerator / denominator);
  });

  return timesteps.map((current, index) => {
    const next = timesteps[index + 1] ?? 0;
    return {
      index,
      current,
      next,
      delta: Math.fround(current - next),
    };
  });
}

export function eulerFlowStep(
  latent: Float32Array,
  velocity: Float32Array,
  delta: number,
  output = new Float32Array(latent.length),
) {
  if (latent.length !== velocity.length || output.length !== latent.length) {
    throw new RangeError("Euler tensors must have identical flattened shapes.");
  }
  if (!Number.isFinite(delta) || delta < 0) {
    throw new RangeError("Euler delta must be finite and non-negative.");
  }

  for (let index = 0; index < latent.length; index += 1) {
    output[index] = Math.fround(
      Math.fround(latent[index]) -
        Math.fround(Math.fround(velocity[index]) * Math.fround(delta)),
    );
  }
  return output;
}
