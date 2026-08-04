import { describe, expect, it } from "vitest";
import {
  assessLyricDuration,
  cleanLyrics,
  compactLyrics,
  countLyricWords,
  defaultMaxLyricWords,
  lyricQualityIssues,
  recommendDurationForLyrics,
} from "../lib/lyrics";

describe("browser lyric post-processing", () => {
  it("removes thinking/fences and normalizes Markdown section headings", () => {
    expect(
      cleanLyrics(
        "<think>draft notes</think>\n```markdown\n## Verse\nLine one\n\n## Chorus\nLine two\n```",
      ),
    ).toBe("[Verse]\nLine one\n[Chorus]\nLine two");
  });

  it("compacts sung words without charging section tags to the budget", () => {
    const result = compactLyrics(
      "[Verse]\none two three\nfour five six\nseven eight nine\n\n[Chorus]\nten eleven twelve",
      9,
    );
    expect(result).toBe(
      "[Verse]\none two three\nfour five six\nseven eight nine\n\n[Chorus]",
    );
  });

  it("detects unusably short and repeated drafts deterministically", () => {
    const issues = lyricQualityIssues(
      "[Verse]\nSame line\nSame line\nSame line\nSame line",
      40,
    );
    expect(issues).toContain("remove the repeated verse loop");
  });

  it("matches the backend vocal word budget", () => {
    expect(defaultMaxLyricWords(10)).toBe(40);
    expect(defaultMaxLyricWords(30)).toBe(40);
    expect(defaultMaxLyricWords(60)).toBe(70);
    expect(defaultMaxLyricWords(120)).toBe(140);
  });

  it("counts only sung words and recommends the first fitting duration", () => {
    const lyrics =
      "[Verse]\n" +
      Array.from({ length: 66 }, (_, index) => `word${index}`).join(" ");
    expect(countLyricWords(lyrics)).toBe(66);
    expect(recommendDurationForLyrics(lyrics)).toBe(60);
    expect(assessLyricDuration(lyrics, 30)).toEqual({
      wordCount: 66,
      selectedDurationSeconds: 30,
      selectedWordBudget: 40,
      recommendedDurationSeconds: 60,
      recommendedWordBudget: 70,
      fits: false,
      exceedsByWords: 26,
    });
  });
});
