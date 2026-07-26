import { describe, expect, it } from "vitest";
import { applyDcw } from "../lib/dcw";
import {
  DEFAULT_DCW_OPTIONS,
  resolveDcwOptions,
} from "../lib/generation-options";

describe("ACE-Step Haar DCW", () => {
  it("is an exact no-op by default", () => {
    const latent = new Float32Array([1, 2, 3, 4]);
    const result = applyDcw(
      latent,
      new Float32Array([0, 0, 0, 0]),
      2,
      2,
      1,
      DEFAULT_DCW_OPTIONS,
    );
    expect(result).toEqual(latent);
    expect(result).not.toBe(latent);
  });

  it("matches the direct pixel correction", () => {
    const result = applyDcw(
      new Float32Array([2, -1]),
      new Float32Array([1, 3]),
      2,
      1,
      0.75,
      resolveDcwOptions({
        enabled: true,
        mode: "pix",
        scaler: 0.05,
      }),
    );
    expect(Array.from(result)).toEqual([
      Math.fround(2 + 0.05 * (2 - 1)),
      Math.fround(-1 + 0.05 * (-1 - 3)),
    ]);
  });

  it("matches the official native Haar low/high band equations", () => {
    const x = new Float32Array([2, 4]);
    const denoised = new Float32Array([1, 1]);
    const lowOnly = applyDcw(
      x,
      denoised,
      2,
      1,
      1,
      resolveDcwOptions({
        enabled: true,
        mode: "low",
        scaler: 0.05,
      }),
    );
    const highOnly = applyDcw(
      x,
      denoised,
      2,
      1,
      0,
      resolveDcwOptions({
        enabled: true,
        mode: "high",
        scaler: 0.05,
      }),
    );
    expect(Array.from(lowOnly)).toEqual([
      expect.closeTo(2.1, 5),
      expect.closeTo(4.1, 5),
    ]);
    expect(Array.from(highOnly)).toEqual([
      expect.closeTo(1.95, 5),
      expect.closeTo(4.05, 5),
    ]);
  });

  it("zero-pads and crops an odd time dimension", () => {
    const result = applyDcw(
      new Float32Array([1, 2, 3]),
      new Float32Array([0, 0, 0]),
      3,
      1,
      0.5,
      resolveDcwOptions({
        enabled: true,
        mode: "double",
        scaler: 0.05,
        highScaler: 0.02,
      }),
    );
    expect(result).toHaveLength(3);
    expect(Array.from(result).every(Number.isFinite)).toBe(true);
  });

  it("validates public DCW configuration", () => {
    expect(() =>
      resolveDcwOptions({ scaler: 0.101 }),
    ).toThrow(/scaler/);
    expect(() =>
      resolveDcwOptions({ mode: "bogus" as "low" }),
    ).toThrow(/mode/);
  });
});
