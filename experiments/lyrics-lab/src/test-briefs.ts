export type LyricTestBrief = {
  id: string;
  label: string;
  focus: string;
  brief: string;
};

export const DEFAULT_TEST_BRIEF_ID = "awkward-memory";

export const LYRIC_TEST_BRIEFS: readonly LyricTestBrief[] = [
  {
    id: "awkward-memory",
    label: "Funny awkward memory",
    focus: "Explicit detail, agency, cause, affectionate humor",
    brief:
      "A funny memory about my girlfriend Patricia and me in the Alps. Patricia was shitting under a tree outside our motorhome. She was very uncomfortable doing it outside, but she did it because the motorhome shit tank was already full. Tell it affectionately from my point of view without changing what happened.",
  },
  {
    id: "quiet-grief",
    label: "Quiet grief",
    focus: "Emotional restraint, concrete objects, no invented tragedy",
    brief:
      "A gentle folk song about clearing out my grandfather's workshop after he died. His cracked green radio still plays the Sunday station he loved. I keep one bent screwdriver because he taught me to repair things with it. The feeling is sad but grateful, not hopeless.",
  },
  {
    id: "cause-and-effect",
    label: "Cause-and-effect mishap",
    focus: "Sequence, who did what, comic escalation",
    brief:
      "An energetic pop-rock song about Maya bringing a birthday cake onto a crowded bus because her car would not start. The driver braked for a cyclist, the cake slid into a stranger's lap, and the stranger started laughing instead of getting angry. Maya arrived late with an empty cake box and a new friend.",
  },
  {
    id: "mundane-victory",
    label: "Mundane victory",
    focus: "Make an ordinary event specific without invented drama",
    brief:
      "A triumphant dance song about finally fixing the kitchen drawer that jammed for three years. The narrator finds a missing teaspoon stuck behind it, calls their sister to celebrate, and enjoys opening the drawer smoothly over and over. Play the tiny victory completely straight.",
  },
  {
    id: "conflicted-anger",
    label: "Conflicted anger",
    focus: "Mixed emotion, factual conflict, no generic breakup clichés",
    brief:
      "A tense indie song addressed to a close friend who shared my private news before I was ready. They apologized sincerely, and I believe them, but trust has not returned yet. The song should hold anger and affection at the same time without turning the friend into a villain.",
  },
  {
    id: "place-and-language",
    label: "Place-specific celebration",
    focus: "Names, setting, sensory detail, upbeat tone",
    brief:
      "A joyful summer song about cousins cooking late at night beside the Danube in Bratislava after a storm. The wet tables shine, someone burns the first batch of sausages, and everyone sings an old family melody while the city lights return. The chorus should celebrate being together again.",
  },
];

export const lyricTestBriefById = (id: string) => {
  const item = LYRIC_TEST_BRIEFS.find((candidate) => candidate.id === id);
  if (!item) throw new Error(`Unknown lyric test brief: ${id}`);
  return item;
};
