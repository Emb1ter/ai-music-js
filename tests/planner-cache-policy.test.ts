import { describe, expect, it } from "vitest";
import {
  decidePlannerAssetCache,
  missingAssetBytes,
} from "../lib/planner-cache-policy";

describe("high-quality planner cache policy", () => {
  it("caches a large planner shard when only the actually missing audio is reserved", () => {
    const decision = decidePlannerAssetCache({
      availableBytes: 10_740_000_000,
      missingAudioBytes: 1_880_000_000,
      assetBytes: 947_312_640,
    });

    expect(decision.cache).toBe(true);
    expect(decision.requiredBytes).toBe(
      1_880_000_000 + 947_312_640 * 2,
    );
  });

  it("can skip a large shard but retain a smaller shard", () => {
    const common = {
      availableBytes: 6_600_000_000,
      missingAudioBytes: 5_460_000_000,
    };
    expect(
      decidePlannerAssetCache({
        ...common,
        assetBytes: 947_312_640,
      }).cache,
    ).toBe(false);
    expect(
      decidePlannerAssetCache({
        ...common,
        assetBytes: 83_886_080,
      }).cache,
    ).toBe(true);
  });

  it("reserves a fixed minimum transaction margin for small files", () => {
    const decision = decidePlannerAssetCache({
      availableBytes: 1_000_000_000,
      missingAudioBytes: 800_000_000,
      assetBytes: 10_000_000,
    });

    expect(decision.writeHeadroomBytes).toBe(128_000_000);
    expect(decision.requiredBytes).toBe(938_000_000);
    expect(decision.cache).toBe(true);
  });

  it("counts only uncached audio assets", () => {
    expect(
      missingAssetBytes([
        { bytes: 100, cached: true },
        { bytes: 200, cached: false },
        { bytes: 300, cached: true },
        { bytes: 400, cached: false },
      ]),
    ).toBe(600);
  });
});
