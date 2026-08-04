import { describe, expect, it } from "vitest";
import {
  FP32_AUDIO_CODE_COUNT,
  FP32_PLANNER_HIDDEN_SIZE,
  Fp32PlannerEmbeddingTable,
  fp32AudioHeadRowLocation,
} from "../lib/planner-fp32-webgpu";

describe("FP32 planner embedding cache", () => {
  it("loads a persistent row without a network range request", async () => {
    const expected = new Float32Array(FP32_PLANNER_HIDDEN_SIZE).fill(0.25);
    const table = new Fp32PlannerEmbeddingTable(
      "https://example.test/model.onnx_data",
      () => undefined,
      {
        load: async (tokenId) => (tokenId === 42 ? expected : undefined),
        save: async () => undefined,
      },
    );

    const tensor = await table.embed([42], [1, 1]);
    expect((tensor.data as Float32Array)[0]).toBe(0.25);
    expect(table.stats()).toMatchObject({
      persistentHits: 1,
      fetchedRows: 0,
      rangeRequests: 0,
    });
    tensor.dispose();
  });

  it("accepts an exact generated-token row already read from the head", async () => {
    const table = new Fp32PlannerEmbeddingTable(
      "https://example.test/model.onnx_data",
    );
    const row = new Float32Array(FP32_PLANNER_HIDDEN_SIZE).fill(-0.5);
    table.setRow(151_669, row);

    const tensor = await table.embed([151_669, 151_669], [2, 1]);
    expect((tensor.data as Float32Array)[FP32_PLANNER_HIDDEN_SIZE]).toBe(-0.5);
    expect(table.stats()).toMatchObject({
      injectedRows: 1,
      memoryHits: 1,
      rangeRequests: 0,
    });
    tensor.dispose();
  });
});

describe("FP32 planner audio-head row mapping", () => {
  it.each([
    [0, 0, 0],
    [8_191, 0, 8_191],
    [8_192, 1, 0],
    [FP32_AUDIO_CODE_COUNT - 1, 7, 6_655],
  ])("maps code %i to shard %i row %i", (code, shard, row) => {
    expect(fp32AudioHeadRowLocation(code)).toEqual({
      shardIndex: shard,
      rowInShard: row,
    });
  });

  it("rejects values outside the 64k audio codebook", () => {
    expect(() => fp32AudioHeadRowLocation(-1)).toThrow(RangeError);
    expect(() => fp32AudioHeadRowLocation(FP32_AUDIO_CODE_COUNT)).toThrow(
      RangeError,
    );
  });
});
