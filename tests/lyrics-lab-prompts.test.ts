import { describe, expect, it } from "vitest";
import {
  COMPACT_FIDELITY_LYRICS_SYSTEM_PROMPT,
  FACT_EXTRACTION_SYSTEM_PROMPT,
  MINIMAL_FIDELITY_LYRICS_SYSTEM_PROMPT,
  NARRATIVE_FIDELITY_LYRICS_SYSTEM_PROMPT,
  RHYMEAI_FACT_DENSE_LYRICS_SYSTEM_PROMPT,
  RHYMEAI_MINIMAL_LYRICS_SYSTEM_PROMPT,
  RHYMEAI_NATURAL_LANGUAGE_SYSTEM_PROMPT,
  RHYMEAI_EXACT_WORDS_SYSTEM_PROMPT,
  RHYMEAI_LITERAL_SENSE_SYSTEM_PROMPT,
  SONGCRAFT_BALANCED_LYRICS_SYSTEM_PROMPT,
  STORY_EDITOR_LYRICS_SYSTEM_PROMPT,
  buildFactExtractionPrompt,
  buildFactGroundedLyricsPrompt,
  buildRhymeAiLooseLongFormLyricsPrompt,
  buildRhymeAiLiteralLongFormLyricsPrompt,
  buildRhymeAiMinimalLyricsPrompt,
  buildStrictBudgetLyricsPrompt,
  rhymeAiSafeGenerationWordLimit,
} from "../experiments/lyrics-lab/src/prompts";
import {
  DEFAULT_LYRIC_LAB_MODEL_ID,
  LYRIC_LAB_MODELS,
  lyricLabModelById,
} from "../experiments/lyrics-lab/src/models";
import {
  DEFAULT_TEST_BRIEF_ID,
  LYRIC_TEST_BRIEFS,
  lyricTestBriefById,
} from "../experiments/lyrics-lab/src/test-briefs";
import {
  PROMPT_VARIANTS,
  buildBatchMatrix,
  serializeBatchResultsCsv,
  type BatchResultRow,
} from "../experiments/lyrics-lab/src/batch-evaluation";
import {
  missingExactTermIssues,
  requiredExactTerms,
} from "../experiments/lyrics-lab/src/exact-words";

describe("lyric-lab model selection", () => {
  it("uses the pinned 2B Q4F16 export by default", () => {
    const model = lyricLabModelById(DEFAULT_LYRIC_LAB_MODEL_ID);

    expect(model.modelId).toBe("onnx-community/Qwen3.5-2B-ONNX-OPT");
    expect(model.dtype).toBe("q4f16");
    expect(model.family).toBe("qwen35");
    expect(model.revision).toMatch(/^[a-f0-9]{40}$/);
    expect(model.downloadBytes).toBeGreaterThan(1_400_000_000);
  });

  it("offers the pinned local RhymeAI Gemma 4 E2B export", () => {
    const model = lyricLabModelById("rhymeai-gemma4-e2b-q4f16");

    expect(model.modelId).toBe("Shayde182/rhymeai-gemma-4-gguf");
    expect(model.revision).toBe(
      "27492c6696f2f868bbbe3aeafb633009ef932598",
    );
    expect(model.family).toBe("gemma4");
    expect(model.dtype).toBe("q4f16");
    expect(model.localOnly).toBe(true);
    expect(model.downloadBytes).toBe(3_130_549_798);
  });

  it("keeps the production 0.8B model as a baseline", () => {
    expect(LYRIC_LAB_MODELS.map((model) => model.id)).toContain(
      "qwen35-08b-q4",
    );
  });

  it("offers the pinned 4B Q4F16 browser export", () => {
    const model = lyricLabModelById("qwen35-4b-q4f16");

    expect(model.modelId).toBe("onnx-community/Qwen3.5-4B-ONNX-OPT");
    expect(model.localOnly).toBe(false);
    expect(model.dtype).toBe("q4f16");
    expect(model.revision).toBe(
      "57b13b4dce7be073be0df3eaf1c842a6bbb2e0a7",
    );
    expect(model.downloadBytes).toBe(2_820_843_621);
  });

  it("offers the pinned local HauhauCS Q4F16 experiment", () => {
    const model = lyricLabModelById("qwen35-2b-hauhaucs-q4f16");

    expect(model.modelId).toBe(
      "DreamFast/Qwen3.5-2B-Uncensored-HauhauCS-Aggressive-Safetensor-Benchmark",
    );
    expect(model.localOnly).toBe(true);
    expect(model.dtype).toBe("q4f16");
    expect(model.revision).toMatch(/^[a-f0-9]{40}$/);
  });
});

describe("strict lyric-lab prompt arithmetic", () => {
  it("turns the 30-second backend budget into six bounded lines", () => {
    const prompt = buildStrictBudgetLyricsPrompt({
      brief: "A funny song about a rainy mountain holiday.",
      durationSeconds: 30,
      maxWords: 40,
      language: "English",
    });

    expect(prompt).toContain("Absolute maximum: 40 sung words");
    expect(prompt).toContain("Safer target: no more than 36 sung words");
    expect(prompt).toContain("Per-line maximum: 6 words");
    expect(prompt).toContain("Total lyric lines: exactly 6");
    expect(prompt).toContain("[Verse]: exactly 3 lyric lines");
    expect(prompt).toContain("[Chorus]: exactly 3 lyric lines");
    expect(prompt).toContain("Stop after line 6");
  });

  it("leaves safety headroom in the 60-second budget", () => {
    const prompt = buildStrictBudgetLyricsPrompt({
      brief: "A hopeful song about starting again.",
      durationSeconds: 60,
      maxWords: 70,
      language: "English",
    });

    expect(prompt).toContain("Safer target: no more than 60 sung words");
    expect(prompt).toContain("Per-line maximum: 5 words");
    expect(prompt).toContain("Total lyric lines: exactly 12");
  });

  it("prioritizes immutable facts and explicit tone over rhyme", () => {
    expect(NARRATIVE_FIDELITY_LYRICS_SYSTEM_PROMPT).toContain(
      "Treat those facts as immutable",
    );
    expect(NARRATIVE_FIDELITY_LYRICS_SYSTEM_PROMPT).toContain(
      "Do not censor, sanitize, euphemize, or avoid them",
    );
    expect(NARRATIVE_FIDELITY_LYRICS_SYSTEM_PROMPT).toContain(
      "Do not create nonsense words",
    );
  });

  it("offers three story-agnostic one-pass prompt experiments", () => {
    const prompts = [
      MINIMAL_FIDELITY_LYRICS_SYSTEM_PROMPT,
      STORY_EDITOR_LYRICS_SYSTEM_PROMPT,
      SONGCRAFT_BALANCED_LYRICS_SYSTEM_PROMPT,
    ];

    expect(new Set(prompts).size).toBe(3);
    for (const prompt of prompts) {
      expect(prompt).toMatch(/cause|causality/);
      expect(prompt).toMatch(/tone|emotional/);
      expect(prompt).toContain("Return only");
      expect(prompt).not.toMatch(/Patricia|motorhome|Alps/i);
    }
    expect(MINIMAL_FIDELITY_LYRICS_SYSTEM_PROMPT.length).toBeLessThan(
      NARRATIVE_FIDELITY_LYRICS_SYSTEM_PROMPT.length,
    );
  });

  it("keeps the compact small-model candidate general and constraint-led", () => {
    expect(COMPACT_FIDELITY_LYRICS_SYSTEM_PROMPT).toContain(
      "who does each action",
    );
    expect(COMPACT_FIDELITY_LYRICS_SYSTEM_PROMPT).toContain(
      "why it happens",
    );
    expect(COMPACT_FIDELITY_LYRICS_SYSTEM_PROMPT).toContain(
      "remove decoration before changing the main event",
    );
    expect(COMPACT_FIDELITY_LYRICS_SYSTEM_PROMPT).toContain(
      "exact language, section order, line counts, per-line limit, and total word limit",
    );
    expect(COMPACT_FIDELITY_LYRICS_SYSTEM_PROMPT).not.toMatch(
      /Patricia|motorhome|Alps|Danube|birthday cake|green radio/i,
    );
    expect(COMPACT_FIDELITY_LYRICS_SYSTEM_PROMPT.length).toBeLessThan(
      STORY_EDITOR_LYRICS_SYSTEM_PROMPT.length,
    );
  });

  it("keeps the RhymeAI candidate fact-dense and story-agnostic", () => {
    expect(RHYMEAI_FACT_DENSE_LYRICS_SYSTEM_PROMPT).toContain(
      "state the central action or event",
    );
    expect(RHYMEAI_FACT_DENSE_LYRICS_SYSTEM_PROMPT).toContain(
      "state its cause or consequence",
    );
    expect(RHYMEAI_FACT_DENSE_LYRICS_SYSTEM_PROMPT).toContain(
      "Rhyme is optional",
    );
    expect(RHYMEAI_FACT_DENSE_LYRICS_SYSTEM_PROMPT).toContain(
      "Every line must either preserve a source fact",
    );
    expect(RHYMEAI_FACT_DENSE_LYRICS_SYSTEM_PROMPT).not.toMatch(
      /Patricia|motorhome|Alps|Danube|birthday cake|green radio/i,
    );
  });

  it("offers a genuinely minimal RhymeAI recipe", () => {
    const userPrompt = buildRhymeAiMinimalLyricsPrompt({
      brief: "Sam missed the train because the clock was wrong.",
      durationSeconds: 30,
      maxWords: 40,
      language: "English",
    });

    expect(RHYMEAI_MINIMAL_LYRICS_SYSTEM_PROMPT.length).toBeLessThan(
      RHYMEAI_FACT_DENSE_LYRICS_SYSTEM_PROMPT.length / 2,
    );
    expect(RHYMEAI_MINIMAL_LYRICS_SYSTEM_PROMPT).toContain(
      "Do not invent or change facts to make a rhyme",
    );
    expect(userPrompt).toContain("Sam missed the train");
    expect(userPrompt).toContain("Maximum: 40 sung words");
    expect(userPrompt).toContain("[Verse]");
    expect(userPrompt).toContain("[Chorus]");
  });

  it("keeps the proven K prompts byte-for-byte stable", () => {
    expect(RHYMEAI_MINIMAL_LYRICS_SYSTEM_PROMPT).toBe(
      `Write short, singable lyrics faithful to the user's story.

Keep the real people, place, main event, reason, feeling, and tone. Do not invent or change facts to make a rhyme. Use direct, natural language. Let the verse tell what happened and the chorus deliver the real emotional payoff.

Follow the requested language and word limit. Return only lyrics with section tags.`,
    );
    expect(
      buildRhymeAiMinimalLyricsPrompt({
        brief: "Sam missed the train because the clock was wrong.",
        durationSeconds: 30,
        maxWords: 40,
        language: "English",
      }),
    ).toBe(`Write a short song from this story:

Sam missed the train because the clock was wrong.

Language: English
Duration: 30 seconds
Maximum: 40 sung words

Use a [Verse] to tell what happened and a [Chorus] for the emotional payoff. An [Outro] is optional only if it fits the word limit. Keep the important names, action, cause, and feeling. Return lyrics only.`);
  });

  it("keeps O identical to K at 30 seconds", () => {
    const options = {
      brief: "Sam missed the train because the clock was wrong.",
      durationSeconds: 30,
      maxWords: 40,
      language: "English",
    };

    expect(buildRhymeAiLooseLongFormLyricsPrompt(options)).toBe(
      buildRhymeAiMinimalLyricsPrompt(options),
    );
  });

  it("changes only K's structure sentence for loose long-form output", () => {
    const options = {
      brief: "Sam missed the train because the clock was wrong.",
      durationSeconds: 60,
      maxWords: 70,
      language: "English",
    };
    const baseline = buildRhymeAiMinimalLyricsPrompt(options);
    const longForm = buildRhymeAiLooseLongFormLyricsPrompt(options);
    const baselineStructure =
      "Use a [Verse] to tell what happened and a [Chorus] for the emotional payoff. An [Outro] is optional only if it fits the word limit.";
    const longFormStructure =
      "Use [Verse 1], [Chorus], then [Verse 2] to continue the same story.";

    expect(longForm).toBe(
      baseline.replace(baselineStructure, longFormStructure),
    );
    expect(longForm).not.toMatch(/exactly|target:|use most|do not stop/i);
  });

  it("makes every phrase's literal source referent Q's highest priority", () => {
    expect(RHYMEAI_LITERAL_SENSE_SYSTEM_PROMPT).toContain(
      "Every line and phrase must make literal sense",
    );
    expect(RHYMEAI_LITERAL_SENSE_SYSTEM_PROMPT).toContain(
      "silently ask what it refers to",
    );
    expect(RHYMEAI_LITERAL_SENSE_SYSTEM_PROMPT).toContain(
      "Never invent an object, surface, location, action, metaphor, or vague poetic phrase",
    );
    expect(RHYMEAI_LITERAL_SENSE_SYSTEM_PROMPT).not.toMatch(
      /Patricia|motorhome|Alps|slippery lip/i,
    );
  });

  it("gives long-form generation 20 percent counting headroom", () => {
    expect(rhymeAiSafeGenerationWordLimit(30, 40)).toBe(40);
    expect(rhymeAiSafeGenerationWordLimit(60, 70)).toBe(56);
    expect(rhymeAiSafeGenerationWordLimit(90, 105)).toBe(84);
    expect(rhymeAiSafeGenerationWordLimit(120, 140)).toBe(112);

    const prompt = buildRhymeAiLiteralLongFormLyricsPrompt({
      brief: "Sam missed the train because the clock was wrong.",
      durationSeconds: 90,
      maxWords: 105,
      language: "English",
    });
    expect(prompt).toContain("Maximum: 84 sung words");
    expect(prompt).toContain("[Bridge]");
  });

  it("makes semantic correctness a hard gate for RhymeAI rhyme", () => {
    expect(RHYMEAI_NATURAL_LANGUAGE_SYSTEM_PROMPT).toContain(
      "Natural meaning is mandatory. Rhyme is optional.",
    );
    expect(RHYMEAI_NATURAL_LANGUAGE_SYSTEM_PROMPT).toContain(
      "An accurate unrhymed line is always better",
    );
    expect(RHYMEAI_NATURAL_LANGUAGE_SYSTEM_PROMPT).toContain(
      "Never use a malformed phrase",
    );
    expect(RHYMEAI_NATURAL_LANGUAGE_SYSTEM_PROMPT).not.toMatch(
      /Patricia|motorhome|Alps|oak voice|Danube/i,
    );
  });

  it("makes explicitly requested wording a verbatim contract", () => {
    expect(RHYMEAI_EXACT_WORDS_SYSTEM_PROMPT).toContain(
      "must appear verbatim in a sung lyric line at least once",
    );
    expect(RHYMEAI_EXACT_WORDS_SYSTEM_PROMPT).toContain(
      "Do not omit, censor, soften, euphemize, translate, or replace it",
    );
    expect(RHYMEAI_EXACT_WORDS_SYSTEM_PROMPT).not.toContain(
      'Use the word "shit"',
    );
  });

  it("detects explicitly required quoted text omitted by a model", () => {
    const brief =
      'Tell the memory without changing it. Use the word "shit".';

    expect(requiredExactTerms(brief)).toEqual(["shit"]);
    expect(missingExactTermIssues(brief, "We hid beneath the trees.")).toEqual([
      'missing required exact text: "shit"',
    ]);
    expect(
      missingExactTermIssues(brief, "The shit tank was full that night."),
    ).toEqual([]);
  });

  it("does not mistake ordinary quoted story details for requirements", () => {
    expect(
      requiredExactTerms('Write a song titled "Home" about an old friend.'),
    ).toEqual([]);
  });

  it("provides diverse story presets instead of one favored test case", () => {
    expect(LYRIC_TEST_BRIEFS).toHaveLength(6);
    expect(new Set(LYRIC_TEST_BRIEFS.map((item) => item.id)).size).toBe(6);
    expect(lyricTestBriefById(DEFAULT_TEST_BRIEF_ID).brief).toContain(
      "because",
    );
    expect(LYRIC_TEST_BRIEFS.map((item) => item.focus).join(" ")).toMatch(
      /grief|emotion/i,
    );
  });

  it("separates fact extraction from budget-bound lyric writing", () => {
    const extraction = buildFactExtractionPrompt({
      brief: "Alex missed the train because the station clock was wrong.",
      language: "English",
    });
    const grounded = buildFactGroundedLyricsPrompt({
      factSheet:
        "PEOPLE: Alex\nEVENT: missed the train\nCAUSE: the station clock was wrong",
      durationSeconds: 30,
      maxWords: 40,
      language: "English",
    });

    expect(FACT_EXTRACTION_SYSTEM_PROMPT).toContain("CAUSE:");
    expect(extraction).toContain("station clock was wrong");
    expect(grounded).toContain("Immutable fact sheet from pass 1");
    expect(grounded).toContain("Absolute maximum: 40 sung words");
    expect(grounded).toContain("CAUSE: the station clock was wrong");
  });
});

describe("lyric-lab batch evaluation", () => {
  it("offers the isolated long-form variants", () => {
    expect(PROMPT_VARIANTS).toHaveLength(16);
    expect(PROMPT_VARIANTS.at(-2)).toEqual({
      id: "p",
      label: "P · L loose long-form",
    });
    expect(PROMPT_VARIANTS.at(-1)).toEqual({
      id: "q",
      label: "Q · literal long-form",
    });
  });

  it("builds a model-major Cartesian matrix", () => {
    const stories = LYRIC_TEST_BRIEFS.slice(0, 2);
    const jobs = buildBatchMatrix({
      modelIds: ["model-1", "model-2"],
      variants: ["a", "g"],
      stories,
    });

    expect(jobs).toHaveLength(8);
    expect(jobs.slice(0, 4).every((job) => job.modelId === "model-1")).toBe(
      true,
    );
    expect(jobs.map((job) => `${job.modelId}/${job.story.id}/${job.variant}`)).toEqual([
      `model-1/${stories[0]!.id}/a`,
      `model-1/${stories[0]!.id}/g`,
      `model-1/${stories[1]!.id}/a`,
      `model-1/${stories[1]!.id}/g`,
      `model-2/${stories[0]!.id}/a`,
      `model-2/${stories[0]!.id}/g`,
      `model-2/${stories[1]!.id}/a`,
      `model-2/${stories[1]!.id}/g`,
    ]);
  });

  it("exports multiline model output as safe, quoted UTF-8 CSV", () => {
    const row: BatchResultRow = {
      runIndex: 1,
      totalRuns: 1,
      status: "success",
      startedAt: "2026-08-13T10:00:00.000Z",
      finishedAt: "2026-08-13T10:00:01.000Z",
      modelId: "model-1",
      modelLabel: "Test model",
      modelRepository: "owner/model",
      modelRevision: "abc",
      modelDtype: "q4f16",
      modelFamily: "qwen35",
      variant: "g",
      storyId: "story-1",
      storyLabel: "Story, with comma",
      storyFocus: "causality",
      sourceBrief: "One line\nSecond line",
      language: "English",
      durationSeconds: 30,
      maxWords: 40,
      seed: 42,
      temperature: 0.55,
      topK: 20,
      topP: 1,
      repetitionPenalty: 1.15,
      maxNewTokens: 500,
      systemPrompt: "System",
      effectiveUserPrompt: "User",
      factSystemPrompt: "",
      factSheet: "",
      factRaw: "",
      rawOutput: '=HYPERLINK("bad")',
      cleanedOutput: "[Verse]\nA lyric",
      processedLyrics: "[Verse]\nA lyric",
      issues: "",
      wordCount: 2,
      generatedWordCount: 2,
      inputTokens: 10,
      outputTokens: 5,
      modelLoadMilliseconds: 100,
      generationMilliseconds: 200,
      factInputTokens: "",
      factOutputTokens: "",
      factExtractionMilliseconds: "",
      error: "",
      stack: "",
    };

    const csv = serializeBatchResultsCsv([row]);
    expect(csv.startsWith("\uFEFF")).toBe(true);
    expect(csv).toContain('"Story, with comma"');
    expect(csv).toContain('"One line\nSecond line"');
    expect(csv).toContain('"\'=HYPERLINK(""bad"")"');
  });
});
