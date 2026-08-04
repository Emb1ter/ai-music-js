import { describe, expect, it } from "vitest";
import {
  AUDIO_CODE_TOKEN_END,
  AUDIO_CODE_TOKEN_START,
  QUALIFIED_FP32_VOCAL_LYRICS,
  QUALIFIED_FP32_VOCAL_METADATA_CAPTION,
  QUALIFIED_FP32_VOCAL_PROMPT,
  analyzePlannerSemanticCodes,
  audioCodeToTokenId,
  blendPlannerCfgLogits,
  createPlannerSamplingRandom,
  deterministicPlannerMetadata,
  formatPlannerGeneratedMetadata,
  formatPlannerMetadata,
  isQualifiedFp32VocalRequest,
  parsePlannerMetadata,
  plannerTopCodeOverlap,
  plannerTopCodes,
  resolvePlannerDuration,
  semanticCodeCount,
  tokenIdToAudioCode,
} from "../lib/planner";

describe("ACE 5 Hz planner helpers", () => {
  it("maps the complete 64k audio codebook to the contiguous tokenizer range", () => {
    expect(audioCodeToTokenId(0)).toBe(AUDIO_CODE_TOKEN_START);
    expect(audioCodeToTokenId(63_999)).toBe(AUDIO_CODE_TOKEN_END - 1);
    expect(tokenIdToAudioCode(AUDIO_CODE_TOKEN_START)).toBe(0);
    expect(tokenIdToAudioCode(AUDIO_CODE_TOKEN_END - 1)).toBe(63_999);
    expect(() => audioCodeToTokenId(64_000)).toThrow(RangeError);
    expect(() => tokenIdToAudioCode(AUDIO_CODE_TOKEN_END)).toThrow(
      RangeError,
    );
  });

  it("calculates an exact deterministic 5 Hz code count", () => {
    expect(semanticCodeCount(10)).toBe(50);
    expect(semanticCodeCount(60)).toBe(300);
    expect(semanticCodeCount(120)).toBe(600);
  });

  it("reproduces the qualified FP32 planner sampling stream", () => {
    const first = createPlannerSamplingRandom(42);
    const second = createPlannerSamplingRandom(42);
    const other = createPlannerSamplingRandom(43);
    const sequence = Array.from({ length: 8 }, () => first());

    expect(sequence).toEqual(
      Array.from({ length: 8 }, () => second()),
    );
    expect(sequence).not.toEqual(
      Array.from({ length: 8 }, () => other()),
    );
    expect(sequence.every((value) => value >= 0 && value < 1)).toBe(true);
  });

  it("keeps the successful FP32 vocal request paired with its qualified metadata", () => {
    expect(
      isQualifiedFp32VocalRequest(
        QUALIFIED_FP32_VOCAL_PROMPT,
        QUALIFIED_FP32_VOCAL_LYRICS,
        30,
      ),
    ).toBe(true);
    expect(
      deterministicPlannerMetadata(
        QUALIFIED_FP32_VOCAL_PROMPT,
        QUALIFIED_FP32_VOCAL_LYRICS,
        "en",
        30,
      ),
    ).toEqual({
      bpm: 100,
      caption: QUALIFIED_FP32_VOCAL_METADATA_CAPTION,
      durationSeconds: 30,
      keyScale: "F major",
      language: "en",
      timeSignature: 4,
    });
    expect(
      isQualifiedFp32VocalRequest(
        `${QUALIFIED_FP32_VOCAL_PROMPT} remix`,
        QUALIFIED_FP32_VOCAL_LYRICS,
        30,
      ),
    ).toBe(false);
  });

  it("makes vocal intent explicit in metadata for custom lyric requests", () => {
    const metadata = deterministicPlannerMetadata(
      "dreamy shoegaze with wide guitars",
      "[Verse]\nSing this softly",
      "en",
      20,
    );
    expect(metadata.caption).toContain(
      "prominent, clear, expressive lead singer",
    );
    expect(metadata.caption).toContain(
      "every supplied lyric intelligibly",
    );
    expect(metadata.language).toBe("en");
    expect(
      deterministicPlannerMetadata(
        "dreamy shoegaze instrumental",
        "",
        "en",
        20,
      ).caption,
    ).toBe("dreamy shoegaze instrumental");
  });

  it("calculates deterministic first-step planner ranking overlap", () => {
    const ranked = plannerTopCodes(
      new Float32Array([0.1, 4, 3, 0.2]),
      3,
    );
    expect(ranked).toEqual([1, 2, 3]);
    expect(plannerTopCodeOverlap(ranked, [2, 1, 0])).toBeCloseTo(
      2 / 3,
    );
  });

  it("measures collapsed and varied semantic-code sequences deterministically", () => {
    expect(analyzePlannerSemanticCodes([7, 7, 7, 9, 8, 8])).toEqual({
      count: 6,
      uniqueCount: 3,
      transitionCount: 2,
      adjacentRepeatRatio: 0.6,
      longestIdenticalRun: 3,
      dominantCodeCount: 3,
      dominantCodeRatio: 0.5,
    });
    expect(analyzePlannerSemanticCodes([])).toEqual({
      count: 0,
      uniqueCount: 0,
      transitionCount: 0,
      adjacentRepeatRatio: 0,
      longestIdenticalRun: 0,
      dominantCodeCount: 0,
      dominantCodeRatio: 0,
    });
  });

  it("combines finite planner CFG rows with the requested scale", () => {
    const output = new Float32Array(3);
    const result = blendPlannerCfgLogits(
      new Float32Array([4, 8, 12]),
      new Float32Array([2, 3, 4]),
      2,
      output,
    );
    expect(result).toEqual({
      mode: "cfg",
      conditionalFinite: 3,
      unconditionalFinite: 3,
      guidedFinite: 3,
    });
    expect(Array.from(output)).toEqual([6, 13, 20]);
  });

  it("falls back to conditional logits when the WebGPU CFG row is non-finite", () => {
    const output = new Float32Array(3);
    const result = blendPlannerCfgLogits(
      new Float32Array([4, 8, 12]),
      new Float32Array([Number.NaN, Number.NaN, Number.NaN]),
      2,
      output,
    );
    expect(result).toEqual({
      mode: "conditional-fallback",
      conditionalFinite: 3,
      unconditionalFinite: 0,
      guidedFinite: 0,
    });
    expect(Array.from(output)).toEqual([4, 8, 12]);
  });

  it("rejects a non-finite conditional planner row", () => {
    const output = new Float32Array(2);
    const result = blendPlannerCfgLogits(
      new Float32Array([Number.NaN, Number.POSITIVE_INFINITY]),
      new Float32Array([2, 3]),
      2,
      output,
    );
    expect(result).toEqual({
      mode: "invalid",
      conditionalFinite: 0,
      unconditionalFinite: 2,
      guidedFinite: 0,
    });
    expect(Array.from(output)).toEqual([
      Number.NEGATIVE_INFINITY,
      Number.NEGATIVE_INFINITY,
    ]);
  });

  it("parses and normalizes backend-format CoT metadata", () => {
    const metadata = parsePlannerMetadata(
      `<think>
bpm: 128
caption: "polished synth-pop with a clear vocal hook"
duration: 45
keyscale: F# minor
language: en
timesignature: 4
</think>`,
      "original caption",
      30,
      "unknown",
    );
    expect(metadata).toEqual({
      bpm: 128,
      caption: "polished synth-pop with a clear vocal hook",
      durationSeconds: 30,
      keyScale: "F# minor",
      language: "en",
      timeSignature: 4,
    });
    expect(formatPlannerMetadata(metadata)).toContain("duration: 30");
  });

  it("uses safe metadata defaults without changing the requested duration", () => {
    expect(
      parsePlannerMetadata(
        "bpm: 900\ntimesignature: 11",
        "ambient piano",
        60,
        "sk",
      ),
    ).toEqual({
      bpm: 120,
      caption: "ambient piano",
      durationSeconds: 60,
      keyScale: "C major",
      language: "sk",
      timeSignature: 4,
    });
  });

  it("uses ACE metadata duration for auto mode without under-running the lyric recommendation", () => {
    expect(
      resolvePlannerDuration({
        plannedDurationSeconds: 47,
        requestedDurationSeconds: 30,
        recommendedDurationSeconds: 60,
        autoDuration: true,
      }),
    ).toEqual({
      durationSeconds: 60,
      durationSource: "recommended",
    });
    expect(
      resolvePlannerDuration({
        plannedDurationSeconds: 74,
        requestedDurationSeconds: 30,
        recommendedDurationSeconds: 60,
        autoDuration: true,
      }),
    ).toEqual({
      durationSeconds: 74,
      durationSource: "ace",
    });
    expect(
      resolvePlannerDuration({
        plannedDurationSeconds: 74,
        requestedDurationSeconds: 30,
        recommendedDurationSeconds: 60,
        autoDuration: false,
      }),
    ).toEqual({
      durationSeconds: 30,
      durationSource: "requested",
    });
  });

  it("formats the production ACE Phase 2 metadata without rewritten caption or language", () => {
    expect(
      formatPlannerGeneratedMetadata({
        bpm: 128,
        caption: "authoritative user caption",
        durationSeconds: 60,
        keyScale: "F# minor",
        language: "en",
        timeSignature: 4,
      }),
    ).toBe(`<think>
bpm: 128
duration: 60
keyscale: F# minor
timesignature: 4
</think>`);
  });
});
