const EXACT_TERM_PATTERN =
  /\b(?:use|include|say|contain|must\s+(?:use|include|contain)|exact(?:\s+word|\s+phrase)?)\b[^\n"“”]{0,36}["“]([^"“”\n]{1,80})["”]/gi;

export const requiredExactTerms = (sourceBrief: string) => {
  const terms = new Set<string>();

  for (const match of sourceBrief.matchAll(EXACT_TERM_PATTERN)) {
    const term = match[1]?.trim();
    if (term) terms.add(term);
  }

  return [...terms];
};

export const missingExactTermIssues = (
  sourceBrief: string,
  lyrics: string,
) => {
  const normalizedLyrics = lyrics.toLocaleLowerCase();

  return requiredExactTerms(sourceBrief)
    .filter(
      (term) => !normalizedLyrics.includes(term.toLocaleLowerCase()),
    )
    .map((term) => `missing required exact text: "${term}"`);
};
