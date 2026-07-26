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

export function predictCleanSample(
  latent: Float32Array,
  velocity: Float32Array,
  timestep: number,
  output = new Float32Array(latent.length),
) {
  if (latent.length !== velocity.length || output.length !== latent.length) {
    throw new RangeError(
      "Clean-prediction tensors must have identical flattened shapes.",
    );
  }
  if (!Number.isFinite(timestep) || timestep < 0 || timestep > 1) {
    throw new RangeError("Timestep must be finite and between zero and one.");
  }
  const timestep32 = Math.fround(timestep);
  for (let index = 0; index < latent.length; index += 1) {
    output[index] = Math.fround(
      Math.fround(latent[index]) -
        Math.fround(Math.fround(velocity[index]) * timestep32),
    );
  }
  return output;
}

export function heunFlowStep(
  latent: Float32Array,
  velocity: Float32Array,
  correctorVelocity: Float32Array,
  delta: number,
  output = new Float32Array(latent.length),
) {
  if (
    latent.length !== velocity.length ||
    latent.length !== correctorVelocity.length ||
    output.length !== latent.length
  ) {
    throw new RangeError("Heun tensors must have identical flattened shapes.");
  }
  if (!Number.isFinite(delta) || delta < 0) {
    throw new RangeError("Heun delta must be finite and non-negative.");
  }
  const delta32 = Math.fround(delta);
  for (let index = 0; index < latent.length; index += 1) {
    const averageVelocity = Math.fround(
      Math.fround(
        Math.fround(velocity[index]) +
          Math.fround(correctorVelocity[index]),
      ) * Math.fround(0.5),
    );
    output[index] = Math.fround(
      Math.fround(latent[index]) -
        Math.fround(averageVelocity * delta32),
    );
  }
  return output;
}

/**
 * Official ACE-Step Euler SDE update: reconstruct x0 at t_current, then
 * re-noise it with an independent normal tensor at t_next.
 */
export function eulerSdeFlowStep(
  latent: Float32Array,
  velocity: Float32Array,
  noise: Float32Array,
  currentTimestep: number,
  nextTimestep: number,
  output = new Float32Array(latent.length),
) {
  if (
    latent.length !== velocity.length ||
    latent.length !== noise.length ||
    output.length !== latent.length
  ) {
    throw new RangeError(
      "Euler SDE tensors must have identical flattened shapes.",
    );
  }
  if (
    !Number.isFinite(currentTimestep) ||
    !Number.isFinite(nextTimestep) ||
    currentTimestep < 0 ||
    currentTimestep > 1 ||
    nextTimestep < 0 ||
    nextTimestep > currentTimestep
  ) {
    throw new RangeError(
      "Euler SDE timesteps must be finite, descending, and between zero and one.",
    );
  }
  const clean = predictCleanSample(latent, velocity, currentTimestep);
  const next32 = Math.fround(nextTimestep);
  const cleanScale = Math.fround(1 - next32);
  for (let index = 0; index < latent.length; index += 1) {
    output[index] = Math.fround(
      Math.fround(next32 * Math.fround(noise[index])) +
        Math.fround(cleanScale * Math.fround(clean[index])),
    );
  }
  return output;
}
