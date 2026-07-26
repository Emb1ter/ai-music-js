import { createHash } from "node:crypto";
import { readFileSync, statSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import {
  ORT_WASM_BYTES,
  ORT_WASM_FILE,
  ORT_WASM_MODULE_BYTES,
  ORT_WASM_MODULE_FILE,
  ORT_WASM_MODULE_SHA256,
  ORT_WASM_SHA256,
} from "../lib/model-manifest";

describe("pinned ONNX Runtime Web WASM fallback", () => {
  it("vendors the exact runtime binary used by the worker", () => {
    const path = resolve(process.cwd(), "public", ORT_WASM_FILE.slice(1));
    expect(statSync(path).size).toBe(ORT_WASM_BYTES);
    expect(
      createHash("sha256").update(readFileSync(path)).digest("hex"),
    ).toBe(ORT_WASM_SHA256);
  });

  it("vendors the matching Emscripten module used by npm consumers", () => {
    const path = resolve(
      process.cwd(),
      "public",
      ORT_WASM_MODULE_FILE.slice(1),
    );
    expect(statSync(path).size).toBe(ORT_WASM_MODULE_BYTES);
    expect(
      createHash("sha256").update(readFileSync(path)).digest("hex"),
    ).toBe(ORT_WASM_MODULE_SHA256);
  });
});
