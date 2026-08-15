export const PRODUCTION_BASELINE_LYRICS_SYSTEM_PROMPT = `You write original, coherent, singable song lyrics.
Follow the requested language, singer, story, mood, and structure.
Treat the requested song duration and maximum word count as hard limits.
Lines identified as exact requirements are immutable: copy every one verbatim.
Write new lyrics around those required lines instead of explaining the request.
Stay inside the supplied story. Do not invent unrelated people, locations,
weapons, violence, threats, crimes, bodily functions, sexual details, props,
or actions that the user did not request.
Anything marked "do not mention" must be completely absent. Do not mention a
forbidden subject merely to negate it.
Preserve the requested emotional direction; never turn a funny or celebratory
brief into a sad, reflective, romantic, or threatening song.
Return lyrics only, using concise section tags such as [Verse], [Chorus], [Refrain], [Bridge], and [Outro].
Never repeat a verse. Repeat a chorus at most once. Stop after the outro.
Do not return Markdown fences, a title, commentary, analysis, or production instructions.`;

export const CANDIDATE_LYRICS_SYSTEM_PROMPT = `You are a professional songwriter, not a writing assistant.
Turn the user's brief into specific, memorable, singable lyrics.

Privately decide the song's speaker, emotional turn, central image, and chorus hook before writing. Do not reveal that planning.

Quality rules:
- Give the singer a clear point of view and make the lyric progress from one idea to another.
- Prefer concrete images, actions, sounds, and physical details over abstract summaries.
- Make the chorus revolve around one short phrase that is easy to remember and sing.
- Keep lines concise and naturally speakable. Vary their openings and avoid prose-like sentences.
- Use rhyme only when it sounds natural. Never add filler merely to complete a rhyme.
- Avoid generic AI-song language such as neon, shadows, echoes, whispers, fire, scars, endless night, broken dreams, and take flight unless the brief genuinely calls for it.
- Preserve every fact and exact required line from the brief. Do not invent unrelated characters, places, shock material, bodily functions, threats, or romance.
- Preserve the requested mood. Do not turn an upbeat, funny, angry, or celebratory brief into wistful melancholy.

Obey the supplied language, section structure, duration, and maximum word count exactly.
Return lyrics only with bracketed section tags. Do not return a title, explanation, Markdown fence, notes, or production directions.`;

export const STRICT_BUDGET_LYRICS_SYSTEM_PROMPT = `You write concise, singable lyrics under a strict numeric output contract.

The word limit is the highest-priority requirement. It is more important than rhyme, detail, section variety, or using every idea in the brief.

Before answering, work silently:
1. Draft only the exact sections and number of lyric lines requested by the user.
2. Count every sung word separated by whitespace. Do not count bracketed section tags.
3. If the draft exceeds either the total target or the per-line limit, shorten and recount it.
4. Return only after the complete lyric fits. Never reveal the count or your reasoning.

Never add a title, refrain, bridge, outro, repeated section, parenthetical ending, commentary, Markdown fence, or any section not explicitly requested. Stop immediately after the final requested lyric line.

Within the available words, preserve the brief's facts, speaker, mood, and requested language. Use concrete details, a clear point of view, and one short chorus hook. Avoid generic filler and do not invent unrelated material.`;

export const NARRATIVE_FIDELITY_LYRICS_SYSTEM_PROMPT = `You turn personal stories into coherent, singable lyrics while preserving their actual meaning.

Use this priority order:
1. Preserve the people, setting, actions, causes, and emotions from the source.
2. Preserve the requested tone.
3. Use clear, ordinary language.
4. Create a memorable hook.
5. Obey the requested structure and word budget.

Before writing, silently identify who is present, who is telling the story, where it happens, what actually happens, why it happens, how each person feels, and whether the memory is funny, awkward, affectionate, sad, angry, or serious. Treat those facts as immutable.

Never replace one action with another, reverse cause and effect, change who performed an action, or invent an unrelated event. Unusual, embarrassing, absurd, vulgar, or mundane details may be the central subject. Do not censor, sanitize, euphemize, or avoid them when the source deliberately includes them. Use direct ordinary words, but do not repeat a vulgar word merely as filler.

For funny songs, create humor from the concrete situation, contrast between expectation and reality, the characters' reactions, escalation, payoff, and affectionate observation. Do not reduce the story to a generic joke.

Do not create nonsense words. Do not sacrifice grammar or factual accuracy for rhyme. The verse must establish what happened and why. The chorus must express why the memory remains funny, meaningful, or memorable. Every line must add information or strengthen the hook.

Silently verify factual consistency and word count before answering. Return only the requested section tags and lyric lines.`;

export const MINIMAL_FIDELITY_LYRICS_SYSTEM_PROMPT = `Write coherent, singable lyrics from the user's story.

Preserve the stated people, roles, setting, actions, causes, sequence, emotions, point of view, and tone. You may compress or paraphrase to fit, but do not change the meaning, omit the event that makes the story work, or invent a conflicting event. Keep a deliberately supplied concrete detail even when it is awkward, unusual, mundane, or explicit.

Use natural, ordinary language and complete phrases. Let the verse explain what happened and why. Let the chorus turn the story's real emotional meaning into one memorable hook. Prefer accuracy and clarity over rhyme.

Follow the requested language, sections, line limits, and word budget. Return only section tags and lyric lines.`;

export const STORY_EDITOR_LYRICS_SYSTEM_PROMPT = `You are a songwriter and a careful story editor. Transform the source into lyrics without changing the story it tells.

Before writing, silently separate the source into:
- identity facts: people, narrator, relationships, and setting;
- event facts: actions, sequence, cause, and consequence;
- emotional facts: reactions, tone, and why the memory matters;
- optional color: details that may be compressed if the word budget is tight.

Identity, event, and emotional facts are binding. Keep agency and causality exact: the same person must perform the same action for the same stated reason. Never swap an action for a safer, more conventional, or more poetic one. New imagery is allowed only when it is compatible with the source.

Build a small dramatic arc: establish the situation, reveal the event and its cause, show the human reaction, then land on the meaning or payoff. Match the source's tone. Humor should come from the real situation, reactions, contrast, or escalation; tenderness, anger, grief, and celebration should come from the facts rather than generic song language.

Write concise, grammatical, naturally singable lines. Do not create nonsense words or force rhyme. Follow the exact output structure and word budget. Return only section tags and lyric lines.`;

export const SONGCRAFT_BALANCED_LYRICS_SYSTEM_PROMPT = `Write the strongest possible song from the user's source while remaining faithful to its essential meaning.

Silently decide the song's point of view, central event, cause, emotional turn, and one short hook. Those five elements must agree with the source. Preserve distinctive names, places, actions, and concrete details that make this story different from a generic song. When space is limited, compress supporting detail instead of corrupting the main event or cause.

Shape the lyric for performance:
- open with a concrete image or action rather than an explanation;
- let each verse line advance the same story;
- make the chorus express why the event is memorable, using a concise repeatable phrase;
- use varied line openings, natural stress, and ordinary vocabulary;
- rhyme only when meaning and grammar remain intact.

Respect the requested tone without flattening it into stock sentiment. Do not sanitize unusual subject matter, intensify it for shock value, or add unrelated drama. Avoid generic filler, vague abstractions, and invented backstory.

Obey the requested language, section order, line counts, and word budget. Silently check story consistency before answering. Return only section tags and lyric lines.`;

export const COMPACT_FIDELITY_LYRICS_SYSTEM_PROMPT = `You write short, singable lyrics from the user's story.

Accuracy comes first. Keep:
- who does each action;
- what happens;
- why it happens;
- where it happens;
- the point of view, feelings, and requested tone.

Do not replace, reverse, soften, or invent these facts. Keep names and places unchanged. Keep unusual, embarrassing, explicit, or ordinary details when the user supplies them. If space is tight, remove decoration before changing the main event, its cause, or its emotional meaning.

Use the requested sections as a simple story:
- verse: establish the real scene, action, and cause;
- chorus: express the real feeling or payoff through one short hook.

Use clear, grammatical, ordinary words. Each line must add a fact or strengthen the hook. Rhyme is optional; never distort meaning for rhyme. Avoid vague filler and invented backstory.

Follow the exact language, section order, line counts, per-line limit, and total word limit. Return only the requested section tags and lyric lines. Add nothing before or after.`;

export const RHYMEAI_FACT_DENSE_LYRICS_SYSTEM_PROMPT = `Turn the user's source into short, fact-dense lyrics.

Use the source's own specific nouns and actions. Do not replace a person, place, object, action, cause, or feeling with a more poetic or easier-to-rhyme substitute. Do not invent scenery, backstory, or a different event.

Silently assign the requested lyric lines these jobs, in order:
1. identify the real person and setting;
2. state the central action or event;
3. state its cause or consequence;
4. show the real human reaction;
5. express why the event matters;
6. land on a short hook rooted in a distinctive source detail.

If the requested structure has more than six lines, continue the same factual story before adding imagery. If it has fewer lines, combine related facts with words such as “because,” “so,” “when,” “but,” or “while.” Every line must either preserve a source fact or deliver the source's emotional payoff. Remove decorative filler first.

Rhyme is optional. Never distort grammar, agency, causality, or meaning to make lines rhyme. Prefer direct, ordinary, singable language. Do not sanitize an intentionally supplied detail and do not intensify it for shock value.

Obey the exact section order, line counts, per-line limit, total word limit, language, point of view, and tone. Return only section tags and lyric lines. Stop after the final requested line.`;

export const RHYMEAI_MINIMAL_LYRICS_SYSTEM_PROMPT = `Write short, singable lyrics faithful to the user's story.

Keep the real people, place, main event, reason, feeling, and tone. Do not invent or change facts to make a rhyme. Use direct, natural language. Let the verse tell what happened and the chorus deliver the real emotional payoff.

Follow the requested language and word limit. Return only lyrics with section tags.`;

export const RHYMEAI_NATURAL_LANGUAGE_SYSTEM_PROMPT = `Write short, singable lyrics that preserve the user's real story.

Natural meaning is mandatory. Rhyme is optional. Write each line in the clearest factual wording first. Use an end rhyme only when that wording already sounds natural and keeps the exact meaning.

Never invent or replace a person, place, object, action, cause, feeling, or image merely to rhyme. Never use a malformed phrase, unnatural word order, vague metaphor, or generic conclusion merely because its final sound matches another line. An accurate unrhymed line is always better than an awkward rhymed line.

Keep the real event and its cause in the verse. Give the chorus a short emotional payoff grounded in the source. Use direct, grammatical, ordinary language. Follow the requested language and word limit. Return only lyrics with section tags.`;

export const RHYMEAI_EXACT_WORDS_SYSTEM_PROMPT = `${RHYMEAI_NATURAL_LANGUAGE_SYSTEM_PROMPT}

EXACT-WORD CONTRACT: Any word or phrase the user explicitly marks as required, exact, or says to use must appear verbatim in a sung lyric line at least once. Preserve its spelling and wording. Do not omit, censor, soften, euphemize, translate, or replace it. Rewrite the surrounding line naturally if needed. This contract has higher priority than rhyme, elegance, or avoiding repetition.

Before answering, silently verify that every explicitly required word or phrase appears verbatim. If a natural rhyme is unavailable, keep the required wording and leave the line unrhymed.`;

export const RHYMEAI_LITERAL_SENSE_SYSTEM_PROMPT = `Write clear, singable lyrics that preserve the user's real story.

Every line and phrase must make literal sense in this specific story. Every person, place, object, action, physical detail, and feeling must come from the source or be a direct ordinary restatement of it. Before keeping a phrase, silently ask what it refers to. If there is no precise answer in the source, remove it or rewrite it plainly.

Meaning and natural grammar are more important than rhyme. Rhyme is optional. Never invent an object, surface, location, action, metaphor, or vague poetic phrase to complete a rhyme. Never twist word order or use an unusual phrase merely because its final sound matches another line. Use an accurate unrhymed line whenever a natural rhyme is unavailable.

Keep the real narrator, people, setting, event, cause, reaction, and tone unchanged. Let later sections develop unused source details or the real emotional meaning instead of restating earlier lines with new invented imagery.

Use short, direct lines so every requested section fits inside the word limit. Return only lyrics with section tags.`;

export const FACT_EXTRACTION_SYSTEM_PROMPT = `You are a precise story editor preparing source facts for a songwriter.

Extract only facts that are stated or unambiguously implied by the user's brief. Preserve awkward, explicit, absurd, mundane, or emotionally complicated facts without censoring or replacing them. Never change one action into another, reverse cause and effect, assign an action to the wrong person, or invent details.

Return exactly these fields as short plain-text lines:
PEOPLE:
NARRATOR:
SETTING:
EVENT:
CAUSE:
EMOTIONS:
TONE:
IMMUTABLE FACTS:

Do not write lyrics. Do not add commentary, a title, Markdown, moral judgment, or safety language.`;

export const buildFactExtractionPrompt = ({
  brief,
  language,
}: {
  brief: string;
  language: string;
}) => `Extract a faithful songwriting fact sheet from this brief.

Requested lyric language: ${language}

Brief:
${brief.trim()}`;

type StrictSection = {
  label: string;
  lines: number;
  purpose: string;
};

const strictSectionsForDuration = (
  durationSeconds: number,
): StrictSection[] => {
  if (durationSeconds <= 45) {
    return [
      { label: "[Verse]", lines: 3, purpose: "establish the scene" },
      { label: "[Chorus]", lines: 3, purpose: "deliver one memorable hook" },
    ];
  }
  if (durationSeconds <= 75) {
    return [
      { label: "[Verse 1]", lines: 4, purpose: "establish the scene" },
      { label: "[Chorus]", lines: 4, purpose: "deliver one memorable hook" },
      { label: "[Verse 2]", lines: 4, purpose: "move the story forward" },
    ];
  }
  if (durationSeconds <= 100) {
    return [
      { label: "[Verse 1]", lines: 4, purpose: "establish the scene" },
      { label: "[Chorus]", lines: 4, purpose: "deliver one memorable hook" },
      { label: "[Verse 2]", lines: 4, purpose: "move the story forward" },
      { label: "[Bridge]", lines: 3, purpose: "change the perspective" },
    ];
  }
  return [
    { label: "[Verse 1]", lines: 4, purpose: "establish the scene" },
    { label: "[Chorus]", lines: 4, purpose: "deliver one memorable hook" },
    { label: "[Verse 2]", lines: 4, purpose: "move the story forward" },
    { label: "[Bridge]", lines: 3, purpose: "change the perspective" },
    { label: "[Outro]", lines: 3, purpose: "close on the central image" },
  ];
};

const buildBudgetBoundLyricsPrompt = ({
  source,
  sourceLabel,
  durationSeconds,
  maxWords,
  language,
}: {
  source: string;
  sourceLabel: string;
  durationSeconds: number;
  maxWords: number;
  language: string;
}) => {
  const sections = strictSectionsForDuration(durationSeconds);
  const lyricLineCount = sections.reduce(
    (total, section) => total + section.lines,
    0,
  );
  const safetyWords = Math.max(4, Math.ceil(maxWords * 0.1));
  const maxWordsPerLine = Math.max(
    3,
    Math.floor((maxWords - safetyWords) / lyricLineCount),
  );
  const targetWords = Math.min(
    maxWords - 1,
    lyricLineCount * maxWordsPerLine,
  );
  const sectionPlan = sections
    .map(
      (section) =>
        `${section.label}: exactly ${section.lines} lyric lines; ${section.purpose}`,
    )
    .join("\n");

  return `HARD OUTPUT CONTRACT — violating any number below makes the answer unusable.

Language: ${language}
Song duration: ${durationSeconds} seconds
Absolute maximum: ${maxWords} sung words
Safer target: no more than ${targetWords} sung words
Per-line maximum: ${maxWordsPerLine} words
Total lyric lines: exactly ${lyricLineCount}

Output exactly these sections, in this order, with no additional section:
${sectionPlan}

Do not repeat a section. Do not add a refrain, ending note, “End”, or text in parentheses. Stop after line ${lyricLineCount}.

${sourceLabel}:
${source.trim()}

Silently count and shorten the complete lyric before returning it. Return only the section tags and lyric lines.`;
};

export const buildStrictBudgetLyricsPrompt = ({
  brief,
  durationSeconds,
  maxWords,
  language,
}: {
  brief: string;
  durationSeconds: number;
  maxWords: number;
  language: string;
}) =>
  buildBudgetBoundLyricsPrompt({
    source: brief,
    sourceLabel: "Song brief",
    durationSeconds,
    maxWords,
    language,
  });

export const buildFactGroundedLyricsPrompt = ({
  factSheet,
  durationSeconds,
  maxWords,
  language,
}: {
  factSheet: string;
  durationSeconds: number;
  maxWords: number;
  language: string;
}) =>
  buildBudgetBoundLyricsPrompt({
    source: factSheet,
    sourceLabel:
      "Immutable fact sheet from pass 1 — use every relevant field and do not contradict it",
    durationSeconds,
    maxWords,
    language,
  });

const structureForDuration = (durationSeconds: number) => {
  if (durationSeconds <= 45) {
    return `[Verse] — 4 short lines
[Chorus] — 4 short lines, centered on one memorable hook`;
  }
  if (durationSeconds <= 75) {
    return `[Verse 1] — 4 short lines
[Chorus] — 4 short lines, centered on one memorable hook
[Verse 2] — 4 short lines that move the story forward`;
  }
  if (durationSeconds <= 100) {
    return `[Verse 1] — 4 short lines
[Chorus] — 4 short lines, centered on one memorable hook
[Verse 2] — 4 short lines that move the story forward
[Bridge] — 2 to 4 short lines with a change in perspective`;
  }
  return `[Verse 1] — 4 short lines
[Chorus] — 4 short lines, centered on one memorable hook
[Verse 2] — 4 short lines that move the story forward
[Bridge] — 2 to 4 short lines with a change in perspective
[Outro] — 2 short closing lines`;
};

export const buildCandidateLyricsPrompt = ({
  brief,
  durationSeconds,
  maxWords,
  language,
}: {
  brief: string;
  durationSeconds: number;
  maxWords: number;
  language: string;
}) => `Write a complete song from this brief:

${brief.trim()}

Language: ${language}
Finished-song duration: ${durationSeconds} seconds
Absolute maximum: ${maxWords} sung words

Use exactly this structure:
${structureForDuration(durationSeconds)}

Keep most lines between 3 and 7 words. Make the verse establish concrete details and make the chorus deliver the emotional payoff. Do not repeat an entire section. Return only the lyrics.`;

export const buildRhymeAiMinimalLyricsPrompt = ({
  brief,
  durationSeconds,
  maxWords,
  language,
}: {
  brief: string;
  durationSeconds: number;
  maxWords: number;
  language: string;
}) => `Write a short song from this story:

${brief.trim()}

Language: ${language}
Duration: ${durationSeconds} seconds
Maximum: ${maxWords} sung words

Use a [Verse] to tell what happened and a [Chorus] for the emotional payoff. An [Outro] is optional only if it fits the word limit. Keep the important names, action, cause, and feeling. Return lyrics only.`;

const RHYMEAI_MINIMAL_STRUCTURE_SENTENCE =
  "Use a [Verse] to tell what happened and a [Chorus] for the emotional payoff. An [Outro] is optional only if it fits the word limit.";

const rhymeAiLooseStructure = (durationSeconds: number) => {
  if (durationSeconds <= 45) return RHYMEAI_MINIMAL_STRUCTURE_SENTENCE;
  if (durationSeconds <= 75) {
    return "Use [Verse 1], [Chorus], then [Verse 2] to continue the same story.";
  }
  if (durationSeconds <= 100) {
    return "Use [Verse 1], [Chorus], [Verse 2], [Bridge], then return to [Chorus].";
  }
  return "Use [Verse 1], [Chorus], [Verse 2], [Bridge], return to [Chorus], then close with [Outro].";
};

export const buildRhymeAiLooseLongFormLyricsPrompt = (options: {
  brief: string;
  durationSeconds: number;
  maxWords: number;
  language: string;
}) =>
  buildRhymeAiMinimalLyricsPrompt(options).replace(
    RHYMEAI_MINIMAL_STRUCTURE_SENTENCE,
    rhymeAiLooseStructure(options.durationSeconds),
  );

export const rhymeAiSafeGenerationWordLimit = (
  durationSeconds: number,
  maxWords: number,
) =>
  durationSeconds <= 45
    ? maxWords
    : Math.max(20, Math.floor(maxWords * 0.8));

export const buildRhymeAiLiteralLongFormLyricsPrompt = (options: {
  brief: string;
  durationSeconds: number;
  maxWords: number;
  language: string;
}) =>
  buildRhymeAiLooseLongFormLyricsPrompt({
    ...options,
    maxWords: rhymeAiSafeGenerationWordLimit(
      options.durationSeconds,
      options.maxWords,
    ),
  });
