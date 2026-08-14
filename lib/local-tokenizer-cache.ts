export interface TokenizerCache {
  match(key: string): Promise<Response | undefined>;
  put(key: string, response: Response): Promise<void>;
  delete?(key: string): Promise<boolean>;
}

const isTokenizerConfig = async (response: Response): Promise<boolean> => {
  if (!response.ok) return false;
  try {
    const value = (await response.json()) as unknown;
    return Boolean(value && typeof value === "object" && !Array.isArray(value));
  } catch {
    return false;
  }
};

/**
 * Transformers.js 4.2 probes tokenizer metadata before loading tokenizer
 * files, but that probe skips an HTTP(S) env.localModelPath. Priming this
 * same-origin key makes a self-hosted tokenizer discoverable without a Hub
 * request.
 */
export const primeLocalTokenizerConfig = async (
  modelBaseUrl: string,
  modelId: string,
  cache: TokenizerCache,
  fetcher: typeof fetch = fetch,
): Promise<string> => {
  const baseUrl = new URL(
    modelBaseUrl.endsWith("/") ? modelBaseUrl : `${modelBaseUrl}/`,
  );
  const modelPath = modelId
    .split("/")
    .filter(Boolean)
    .map(encodeURIComponent)
    .join("/");
  const tokenizerConfigUrl = new URL(
    `${modelPath}/tokenizer_config.json`,
    baseUrl,
  ).href;

  const cached = await cache.match(tokenizerConfigUrl);
  if (cached && (await isTokenizerConfig(cached.clone()))) {
    return tokenizerConfigUrl;
  }
  if (cached) await cache.delete?.(tokenizerConfigUrl);

  const response = await fetcher(tokenizerConfigUrl, { cache: "no-store" });
  if (!(await isTokenizerConfig(response.clone()))) {
    throw new Error(
      `The self-hosted lyric tokenizer configuration is unavailable (${response.status}).`,
    );
  }
  await cache.put(tokenizerConfigUrl, response);
  return tokenizerConfigUrl;
};
