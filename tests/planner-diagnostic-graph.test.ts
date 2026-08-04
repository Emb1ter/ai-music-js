import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

const graphPath = resolve(
  import.meta.dirname,
  "../planner-diagnostics/model_q4f16.onnx",
);

describe("ACE planner WebGPU diagnostic graph", () => {
  it("pins the graph containing per-node non-finite probes", () => {
    const graph = readFileSync(graphPath);
    expect(graph.byteLength).toBe(2_592_937);
    expect(createHash("sha256").update(graph).digest("hex")).toBe(
      "74dee7fcc1dffde1cea49568e6c444aaa3f710a34a9f694ccdc16cdfdcff6a30",
    );
    expect(graph.includes("planner_diag.0006")).toBe(true);
    expect(graph.includes("NotFinite")).toBe(true);
    expect(graph.includes("ReduceCount")).toBe(true);
  });
});
