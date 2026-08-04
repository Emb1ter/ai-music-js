import { describe, expect, it } from "vitest";
import { PlannerProfiler } from "../lib/planner-profile";

describe("production planner profiler", () => {
  it("aggregates per-call min/average/max and separates embedding phases", () => {
    const profiler = new PlannerProfiler();
    profiler.record("semantic-body-session-run", 40);
    profiler.record("semantic-body-session-run", 20);
    profiler.record("semantic-body-total", 70);
    profiler.record("semantic-head-forward-total", 5);
    profiler.record("semantic-cfg", 2);
    profiler.record("semantic-sampling", 3);
    profiler.record("semantic-total", 100);

    const report = profiler.report({
      completedSemanticSteps: 2,
      targetSemanticSteps: 150,
      embeddingSource: "https://example.test/model.onnx_data",
      embeddingAfterMetadata: {
        requestedRows: 10,
        fetchedRows: 8,
        memoryHits: 2,
        persistentHits: 0,
        persistentWrites: 0,
        injectedRows: 0,
        fetchedBytes: 80,
        rangeRequests: 8,
      },
      embeddingTotal: {
        requestedRows: 14,
        fetchedRows: 9,
        memoryHits: 5,
        persistentHits: 2,
        persistentWrites: 1,
        injectedRows: 1,
        fetchedBytes: 90,
        rangeRequests: 9,
      },
    });

    const run = report.metrics.find(
      (metric) => metric.id === "semantic-body-session-run",
    );
    expect(run).toMatchObject({
      totalMilliseconds: 60,
      calls: 2,
      averageMilliseconds: 30,
      minimumMilliseconds: 20,
      maximumMilliseconds: 40,
    });
    expect(report.embedding.semantic).toEqual({
      requestedRows: 4,
      fetchedRows: 1,
      memoryHits: 3,
      persistentHits: 2,
      persistentWrites: 1,
      injectedRows: 1,
      fetchedBytes: 10,
      rangeRequests: 1,
    });
    expect(
      report.metrics.find(
        (metric) => metric.id === "semantic-unattributed",
      )?.totalMilliseconds,
    ).toBe(20);
  });
});
