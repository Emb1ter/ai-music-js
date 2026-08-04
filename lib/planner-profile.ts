export type PlannerProfilePhase =
  | "setup"
  | "metadata"
  | "semantic"
  | "cleanup";

export type PlannerTimingMetric = {
  id: string;
  label: string;
  phase: PlannerProfilePhase;
  includesChildren: boolean;
  totalMilliseconds: number;
  calls: number;
  averageMilliseconds: number;
  minimumMilliseconds: number;
  maximumMilliseconds: number;
};

export type PlannerEmbeddingStats = {
  requestedRows: number;
  fetchedRows: number;
  memoryHits: number;
  persistentHits: number;
  persistentWrites: number;
  injectedRows: number;
  fetchedBytes: number;
  rangeRequests: number;
};

export type PlannerInputFingerprint = {
  batchSize: number;
  paddedTokens: number;
  realTokens: number[];
  conditionalInputIdsSha256: string;
  unconditionalInputIdsSha256: string;
  conditionalAttentionMaskSha256: string;
  unconditionalAttentionMaskSha256: string;
  promptSha256: string;
  lyricsSha256: string;
  metadataReasoningSha256: string;
};

export type PlannerProfileReport = {
  completedSemanticSteps: number;
  targetSemanticSteps: number;
  final: boolean;
  metrics: PlannerTimingMetric[];
  embeddingSource: string;
  embedding: {
    total: PlannerEmbeddingStats;
    metadata: PlannerEmbeddingStats;
    semantic: PlannerEmbeddingStats;
  };
  input?: PlannerInputFingerprint;
};

type MetricDefinition = {
  label: string;
  phase: PlannerProfilePhase;
  includesChildren?: boolean;
};

type MetricAggregate = {
  totalMilliseconds: number;
  calls: number;
  minimumMilliseconds: number;
  maximumMilliseconds: number;
};

export const PLANNER_PROFILE_DEFINITIONS = {
  "worker-total": {
    label: "High-quality planner wall time",
    phase: "setup",
    includesChildren: true,
  },
  "cache-prune": { label: "Cache policy + pruning", phase: "setup" },
  "asset-loader-create": { label: "Asset loader creation", phase: "setup" },
  "tokenizer-load": { label: "Tokenizer load", phase: "setup" },
  "body-assets-load": {
    label: "Transformer graph + body asset reads",
    phase: "setup",
  },
  "body-session-create": {
    label: "Transformer session creation",
    phase: "setup",
  },
  "semantic-tokenization": {
    label: "Semantic prompt formatting + tokenization",
    phase: "setup",
  },
  "head-load": {
    label: "FP32 audio-code head load",
    phase: "setup",
    includesChildren: true,
  },
  "head-adapter-device": {
    label: "Head adapter + device",
    phase: "setup",
  },
  "head-pipeline-create": {
    label: "Head shader pipeline creation",
    phase: "setup",
  },
  "head-weight-asset-load": {
    label: "Head weight cache/network reads",
    phase: "setup",
  },
  "head-weight-gpu-upload": {
    label: "Head weight GPU uploads",
    phase: "setup",
  },
  "metadata-total": {
    label: "ACE metadata reasoning wall time",
    phase: "metadata",
    includesChildren: true,
  },
  "metadata-body-total": {
    label: "Transformer body total",
    phase: "metadata",
    includesChildren: true,
  },
  "metadata-embedding-total": {
    label: "Sparse embedding total",
    phase: "metadata",
    includesChildren: true,
  },
  "metadata-embedding-range-fetch": {
    label: "Embedding HTTP range requests",
    phase: "metadata",
  },
  "metadata-embedding-cache-read": {
    label: "Persistent embedding-cache reads",
    phase: "metadata",
  },
  "metadata-embedding-cache-write": {
    label: "Persistent embedding-cache writes",
    phase: "metadata",
  },
  "metadata-embedding-prefetch": {
    label: "Metadata embedding preload",
    phase: "metadata",
  },
  "metadata-embedding-pack": {
    label: "Embedding tensor packing",
    phase: "metadata",
  },
  "metadata-body-feed-prep": {
    label: "Body feed tensor preparation",
    phase: "metadata",
  },
  "metadata-body-session-run": {
    label: "ORT WebGPU body session.run",
    phase: "metadata",
  },
  "metadata-body-output-cache": {
    label: "Body output + KV-cache handoff",
    phase: "metadata",
  },
  "metadata-sparse-score": {
    label: "Constrained tied-head scoring",
    phase: "metadata",
  },
  "metadata-sampling": {
    label: "Constrained top-p sampling",
    phase: "metadata",
  },
  "metadata-unattributed": {
    label: "Metadata remainder",
    phase: "metadata",
  },
  "semantic-total": {
    label: "Semantic loop wall time",
    phase: "semantic",
    includesChildren: true,
  },
  "semantic-body-total": {
    label: "Transformer body total",
    phase: "semantic",
    includesChildren: true,
  },
  "semantic-embedding-total": {
    label: "Sparse embedding total",
    phase: "semantic",
    includesChildren: true,
  },
  "semantic-embedding-range-fetch": {
    label: "Embedding HTTP range requests",
    phase: "semantic",
  },
  "semantic-embedding-cache-read": {
    label: "Persistent embedding-cache reads",
    phase: "semantic",
  },
  "semantic-embedding-cache-write": {
    label: "Persistent embedding-cache writes",
    phase: "semantic",
  },
  "semantic-embedding-prefetch": {
    label: "Semantic prompt embedding preload",
    phase: "semantic",
  },
  "semantic-embedding-pack": {
    label: "Embedding tensor packing",
    phase: "semantic",
  },
  "semantic-body-feed-prep": {
    label: "Body feed tensor preparation",
    phase: "semantic",
  },
  "semantic-body-session-run": {
    label: "ORT WebGPU body session.run",
    phase: "semantic",
  },
  "semantic-body-output-cache": {
    label: "Body output + KV-cache handoff",
    phase: "semantic",
  },
  "semantic-head-forward-total": {
    label: "FP32 head forward total",
    phase: "semantic",
    includesChildren: true,
  },
  "semantic-head-hidden-copy": {
    label: "Head CPU hidden-row extraction",
    phase: "semantic",
  },
  "semantic-head-command-encode": {
    label: "Head GPU upload + command encoding",
    phase: "semantic",
  },
  "semantic-head-gpu-compute-readback": {
    label: "Head GPU compute + readback wait",
    phase: "semantic",
  },
  "semantic-head-result-copy": {
    label: "Head mapped-result copy",
    phase: "semantic",
  },
  "semantic-head-embedding-readback": {
    label: "Selected-token embedding readback",
    phase: "semantic",
  },
  "semantic-cfg": { label: "CPU CFG blend", phase: "semantic" },
  "semantic-first-ranking": {
    label: "First-step ranking diagnostics",
    phase: "semantic",
  },
  "semantic-sampling": {
    label: "CPU 64k top-p sampling",
    phase: "semantic",
  },
  "semantic-next-input-prep": {
    label: "Next-token mask/position preparation",
    phase: "semantic",
  },
  "semantic-unattributed": {
    label: "Semantic loop remainder",
    phase: "semantic",
  },
  "resource-release": {
    label: "Session and GPU resource release",
    phase: "cleanup",
  },
} as const satisfies Record<string, MetricDefinition>;

export type PlannerProfileTimingId =
  keyof typeof PLANNER_PROFILE_DEFINITIONS;

const emptyEmbeddingStats = (): PlannerEmbeddingStats => ({
  requestedRows: 0,
  fetchedRows: 0,
  memoryHits: 0,
  persistentHits: 0,
  persistentWrites: 0,
  injectedRows: 0,
  fetchedBytes: 0,
  rangeRequests: 0,
});

export const subtractEmbeddingStats = (
  value: PlannerEmbeddingStats,
  baseline: PlannerEmbeddingStats,
): PlannerEmbeddingStats => ({
  requestedRows: value.requestedRows - baseline.requestedRows,
  fetchedRows: value.fetchedRows - baseline.fetchedRows,
  memoryHits: value.memoryHits - baseline.memoryHits,
  persistentHits: value.persistentHits - baseline.persistentHits,
  persistentWrites:
    value.persistentWrites - baseline.persistentWrites,
  injectedRows: value.injectedRows - baseline.injectedRows,
  fetchedBytes: value.fetchedBytes - baseline.fetchedBytes,
  rangeRequests: value.rangeRequests - baseline.rangeRequests,
});

export class PlannerProfiler {
  private readonly aggregates = new Map<
    PlannerProfileTimingId,
    MetricAggregate
  >();

  record(id: PlannerProfileTimingId, milliseconds: number) {
    if (!Number.isFinite(milliseconds) || milliseconds < 0) return;
    const aggregate = this.aggregates.get(id);
    if (aggregate) {
      aggregate.totalMilliseconds += milliseconds;
      aggregate.calls += 1;
      aggregate.minimumMilliseconds = Math.min(
        aggregate.minimumMilliseconds,
        milliseconds,
      );
      aggregate.maximumMilliseconds = Math.max(
        aggregate.maximumMilliseconds,
        milliseconds,
      );
      return;
    }
    this.aggregates.set(id, {
      totalMilliseconds: milliseconds,
      calls: 1,
      minimumMilliseconds: milliseconds,
      maximumMilliseconds: milliseconds,
    });
  }

  total(id: PlannerProfileTimingId) {
    return this.aggregates.get(id)?.totalMilliseconds ?? 0;
  }

  report(options: {
    completedSemanticSteps: number;
    targetSemanticSteps: number;
    embeddingSource: string;
    embeddingTotal: PlannerEmbeddingStats;
    embeddingAfterMetadata?: PlannerEmbeddingStats;
    input?: PlannerInputFingerprint;
    final?: boolean;
  }): PlannerProfileReport {
    const aggregates = new Map(this.aggregates);
    const metadataRemainder = Math.max(
      0,
      this.total("metadata-total") -
        this.total("metadata-body-total") -
        this.total("metadata-sparse-score") -
        this.total("metadata-sampling") -
        this.total("metadata-embedding-prefetch"),
    );
    if (metadataRemainder > 0) {
      aggregates.set("metadata-unattributed", {
        totalMilliseconds: metadataRemainder,
        calls: 1,
        minimumMilliseconds: metadataRemainder,
        maximumMilliseconds: metadataRemainder,
      });
    }
    const semanticRemainder = Math.max(
      0,
      this.total("semantic-total") -
        this.total("semantic-body-total") -
        this.total("semantic-head-forward-total") -
        this.total("semantic-cfg") -
        this.total("semantic-first-ranking") -
        this.total("semantic-sampling") -
        this.total("semantic-next-input-prep") -
        this.total("semantic-embedding-prefetch") -
        this.total("semantic-head-embedding-readback"),
    );
    if (semanticRemainder > 0) {
      aggregates.set("semantic-unattributed", {
        totalMilliseconds: semanticRemainder,
        calls: 1,
        minimumMilliseconds: semanticRemainder,
        maximumMilliseconds: semanticRemainder,
      });
    }
    const metadata = options.embeddingAfterMetadata ?? emptyEmbeddingStats();
    return {
      completedSemanticSteps: options.completedSemanticSteps,
      targetSemanticSteps: options.targetSemanticSteps,
      final: options.final ?? false,
      embeddingSource: options.embeddingSource,
      embedding: {
        total: { ...options.embeddingTotal },
        metadata: { ...metadata },
        semantic: subtractEmbeddingStats(options.embeddingTotal, metadata),
      },
      ...(options.input ? { input: { ...options.input } } : {}),
      metrics: Object.entries(PLANNER_PROFILE_DEFINITIONS).flatMap(
        ([id, definition]) => {
          const aggregate = aggregates.get(id as PlannerProfileTimingId);
          if (!aggregate) return [];
          return [
            {
              id,
              label: definition.label,
              phase: definition.phase,
              includesChildren:
                "includesChildren" in definition
                  ? definition.includesChildren
                  : false,
              totalMilliseconds: aggregate.totalMilliseconds,
              calls: aggregate.calls,
              averageMilliseconds:
                aggregate.totalMilliseconds / aggregate.calls,
              minimumMilliseconds: aggregate.minimumMilliseconds,
              maximumMilliseconds: aggregate.maximumMilliseconds,
            } satisfies PlannerTimingMetric,
          ];
        },
      ),
    };
  }
}
