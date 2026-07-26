import { copyFile, mkdir, stat, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const projectRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const wasmFile = "ort-wasm-simd-threaded.asyncify.wasm";
const wasmSource = resolve(projectRoot, "public", "wasm", wasmFile);
const wasmTarget = resolve(projectRoot, "dist", "wasm", wasmFile);
const expectedWasmBytes = 24_254_953;
const wasmModuleFile = "ort-wasm-simd-threaded.asyncify.mjs";
const wasmModuleSource = resolve(
  projectRoot,
  "public",
  "wasm",
  wasmModuleFile,
);
const wasmModuleTarget = resolve(
  projectRoot,
  "dist",
  "wasm",
  wasmModuleFile,
);
const expectedWasmModuleBytes = 47_507;

await mkdir(dirname(wasmTarget), { recursive: true });
const sourceStat = await stat(wasmSource);
if (sourceStat.size !== expectedWasmBytes) {
  throw new Error(
    `Pinned WASM size mismatch: ${sourceStat.size} != ${expectedWasmBytes}`,
  );
}
await copyFile(wasmSource, wasmTarget);
const moduleSourceStat = await stat(wasmModuleSource);
if (moduleSourceStat.size !== expectedWasmModuleBytes) {
  throw new Error(
    `Pinned WASM module size mismatch: ${moduleSourceStat.size} != ${expectedWasmModuleBytes}`,
  );
}
await copyFile(wasmModuleSource, wasmModuleTarget);

const library = await import(
  new URL(`../dist/index.js?build=${Date.now()}`, import.meta.url)
);
const manifest = {
  package: "ai-music-js",
  pipelineBuild: library.PIPELINE_BUILD,
  totalDownloadBytes: library.TOTAL_DOWNLOAD_BYTES,
  npmPackageEmbedsModelWeights: false,
  localModelFiles: library.LOCAL_MODEL_FILES,
};
await writeFile(
  resolve(projectRoot, "dist", "model-assets.json"),
  `${JSON.stringify(manifest, null, 2)}\n`,
);
