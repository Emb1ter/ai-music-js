export type PlannerQuality = "turbo" | "high-quality";

export const DEFAULT_PLANNER_QUALITY: PlannerQuality = "turbo";

export const DEFAULT_HIGH_QUALITY_PLANNER_MODEL =
  "emb1ter/ACE-Step-v1.5-5Hz-LM-4B-FP32-ONNX-WebGPU";

// Kept separate from the direct-Turbo revision so either conversion can be
// updated without silently changing the other planner path.
export const DEFAULT_HIGH_QUALITY_PLANNER_MODEL_REVISION =
  "ad1eba6d99ea99d7cd2db7f7fb14275634016777";

export const HIGH_QUALITY_PLANNER_GRAPH_FILE =
  "int8-fp32/onnx/model.onnx";

export const HIGH_QUALITY_PLANNER_BODY_FILES = [
  "int8-fp32/onnx/model.onnx.data.0",
  "int8-fp32/onnx/model.onnx.data.1",
  "int8-fp32/onnx/model.onnx.data.2",
  "int8-fp32/onnx/model.onnx.data.3",
  "int8-fp32/onnx/model.onnx.data.4",
] as const;

export const HIGH_QUALITY_PLANNER_HEAD_FILES = [
  "fp32-head/audio_head_0.bin",
  "fp32-head/audio_head_1.bin",
  "fp32-head/audio_head_2.bin",
  "fp32-head/audio_head_3.bin",
  "fp32-head/audio_head_4.bin",
  "fp32-head/audio_head_5.bin",
  "fp32-head/audio_head_6.bin",
  "fp32-head/audio_head_7.bin",
] as const;

/** Sparse range source for prompt/metadata rows; generated rows reuse the head. */
export const HIGH_QUALITY_PLANNER_EMBEDDING_FILE =
  "fp32-embedding/model.onnx_data";

export const HIGH_QUALITY_PLANNER_EMBEDDING_ROW_CACHE_PARAMETER =
  "ai-music-js-fp32-embedding-row";

export const HIGH_QUALITY_PLANNER_BODY_BYTES = 3_938_887_680;
export const HIGH_QUALITY_PLANNER_HEAD_BYTES = 655_360_000;

export const isPlannerQuality = (
  value: unknown,
): value is PlannerQuality =>
  value === "turbo" || value === "high-quality";
