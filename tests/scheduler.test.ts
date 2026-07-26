import { describe, expect, it } from "vitest";
import {
  createTurboSchedule,
  eulerFlowStep,
  eulerSdeFlowStep,
  heunFlowStep,
  predictCleanSample,
} from "../lib/scheduler";

describe("ACE-Step Turbo Euler schedule", () => {
  it("matches the Python shift=3 eight-step schedule", () => {
    const schedule = createTurboSchedule(8, 3);
    expect(schedule.map((step) => step.current)).toEqual([
      1,
      Math.fround(21 / 22),
      Math.fround(9 / 10),
      Math.fround(5 / 6),
      0.75,
      Math.fround(9 / 14),
      0.5,
      Math.fround(0.3),
    ]);
    expect(schedule.at(-1)?.next).toBe(0);
    expect(schedule.map((step) => step.delta)).toEqual([
      Math.fround(1 - Math.fround(21 / 22)),
      Math.fround(Math.fround(21 / 22) - Math.fround(9 / 10)),
      Math.fround(Math.fround(9 / 10) - Math.fround(5 / 6)),
      Math.fround(Math.fround(5 / 6) - 0.75),
      Math.fround(0.75 - Math.fround(9 / 14)),
      Math.fround(Math.fround(9 / 14) - 0.5),
      Math.fround(0.5 - Math.fround(0.3)),
      Math.fround(0.3),
    ]);
    expect(schedule.reduce((sum, step) => sum + step.delta, 0)).toBeCloseTo(1, 6);
  });

  it("uses the Python final-step x0 update", () => {
    const schedule = createTurboSchedule(8, 3);
    const latent = new Float32Array([1, -2, 0.5]);
    const velocity = new Float32Array([0.25, -1, 2]);
    const result = eulerFlowStep(
      latent,
      velocity,
      schedule.at(-1)!.delta,
    );
    const expected = [1, -2, 0.5].map((value, index) =>
      Math.fround(
        Math.fround(value) -
          Math.fround(
            Math.fround(velocity[index]) * Math.fround(schedule.at(-1)!.delta),
          ),
      ),
    );
    expect(Array.from(result)).toEqual(expected);
  });

  it("rejects unsupported Turbo step counts", () => {
    expect(() => createTurboSchedule(0, 3)).toThrow(RangeError);
    expect(() => createTurboSchedule(21, 3)).toThrow(RangeError);
  });

  it("matches the Python Heun trapezoidal update", () => {
    const result = heunFlowStep(
      new Float32Array([1, -2]),
      new Float32Array([0.2, -0.5]),
      new Float32Array([0.4, -0.1]),
      0.25,
    );
    expect(Array.from(result)).toEqual([
      Math.fround(1 - Math.fround(Math.fround(0.3) * 0.25)),
      Math.fround(-2 - Math.fround(Math.fround(-0.3) * 0.25)),
    ]);
  });

  it("matches the Python Euler SDE clean-predict and re-noise update", () => {
    const latent = new Float32Array([1, -2]);
    const velocity = new Float32Array([0.5, -0.25]);
    const noise = new Float32Array([0.2, -0.4]);
    const clean = predictCleanSample(latent, velocity, 0.75);
    const result = eulerSdeFlowStep(
      latent,
      velocity,
      noise,
      0.75,
      0.5,
    );
    expect(Array.from(clean)).toEqual([
      Math.fround(1 - Math.fround(0.5 * 0.75)),
      Math.fround(-2 - Math.fround(-0.25 * 0.75)),
    ]);
    expect(Array.from(result)).toEqual(
      Array.from(clean, (value, index) =>
        Math.fround(
          Math.fround(0.5 * noise[index]!) +
            Math.fround(0.5 * value),
        ),
      ),
    );
  });
});
