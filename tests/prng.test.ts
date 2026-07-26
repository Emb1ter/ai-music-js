import { describe, expect, it } from "vitest";
import { deterministicNormal } from "../lib/prng";

describe("deterministic browser latent RNG", () => {
  it("repeats exactly for the same seed", () => {
    expect(deterministicNormal(64, 42)).toEqual(deterministicNormal(64, 42));
  });

  it("changes for a different seed", () => {
    expect(deterministicNormal(16, 42)).not.toEqual(
      deterministicNormal(16, 43),
    );
  });

  it("uses a stable non-zero fallback for seed zero", () => {
    const values = deterministicNormal(4, 0);
    expect(Array.from(values)).toEqual(Array.from(deterministicNormal(4, 0)));
    expect(values.some((value) => value !== 0)).toBe(true);
  });
});
