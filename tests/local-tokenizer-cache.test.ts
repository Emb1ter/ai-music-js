import { describe, expect, it, vi } from "vitest";
import {
  primeLocalTokenizerConfig,
  type TokenizerCache,
} from "../lib/local-tokenizer-cache";

const createCache = () => {
  const entries = new Map<string, Response>();
  const cache: TokenizerCache = {
    async match(key) {
      return entries.get(key)?.clone();
    },
    async put(key, response) {
      entries.set(key, response.clone());
    },
    async delete(key) {
      return entries.delete(key);
    },
  };
  return { cache, entries };
};

describe("local tokenizer discovery cache", () => {
  it("primes the same-origin key expected by Transformers.js", async () => {
    const { cache, entries } = createCache();
    const fetcher = vi.fn(async () =>
      Response.json({ tokenizer_class: "GemmaTokenizer" }),
    ) as unknown as typeof fetch;

    const key = await primeLocalTokenizerConfig(
      "https://song.test/local-lyrics-models/",
      "Shayde182/rhymeai-gemma-4-gguf",
      cache,
      fetcher,
    );

    expect(key).toBe(
      "https://song.test/local-lyrics-models/Shayde182/rhymeai-gemma-4-gguf/tokenizer_config.json",
    );
    expect(entries.has(key)).toBe(true);
    expect(fetcher).toHaveBeenCalledOnce();

    await primeLocalTokenizerConfig(
      "https://song.test/local-lyrics-models/",
      "Shayde182/rhymeai-gemma-4-gguf",
      cache,
      fetcher,
    );
    expect(fetcher).toHaveBeenCalledOnce();
  });

  it("rejects an unavailable tokenizer configuration", async () => {
    const { cache } = createCache();
    const fetcher = vi.fn(async () =>
      new Response("missing", { status: 404 }),
    ) as unknown as typeof fetch;

    await expect(
      primeLocalTokenizerConfig(
        "https://song.test/local-lyrics-models/",
        "missing/model",
        cache,
        fetcher,
      ),
    ).rejects.toThrow("configuration is unavailable (404)");
  });
});
