import { readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import type { Plugin } from "vite";

const VIRTUAL_PREFIX = "\0ai-music-js-inline-onnx:";

export const inlineOnnxPlugin = (): Plugin => ({
  name: "ai-music-js-inline-onnx",
  enforce: "pre",
  resolveId(source, importer) {
    if (!source.endsWith(".onnx?inline") || !importer) return null;
    return `${VIRTUAL_PREFIX}${resolve(dirname(importer), source.slice(0, -7))}`;
  },
  async load(id) {
    if (!id.startsWith(VIRTUAL_PREFIX)) return null;
    const path = id.slice(VIRTUAL_PREFIX.length);
    const contents = await readFile(path);
    const dataUrl =
      "data:application/octet-stream;base64," + contents.toString("base64");
    return `export default ${JSON.stringify(dataUrl)};`;
  },
});
