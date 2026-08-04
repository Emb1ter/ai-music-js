import { resolve } from "node:path";
import { defineConfig } from "vite";
import { inlineOnnxPlugin } from "./build-plugins/inline-onnx";

export default defineConfig({
  plugins: [inlineOnnxPlugin()],
  publicDir: false,
  resolve: {
    alias: {
      "@": resolve(import.meta.dirname),
    },
    conditions: ["onnxruntime-web-use-extern-wasm"],
    dedupe: ["onnxruntime-web"],
  },
  define: {
    "process.env.NODE_ENV": JSON.stringify("production"),
  },
  build: {
    target: "es2022",
    outDir: "dist",
    emptyOutDir: false,
    sourcemap: true,
    lib: {
      entry: resolve(import.meta.dirname, "workers/language.worker.ts"),
      formats: ["es"],
      fileName: "language.worker",
    },
    rolldownOptions: {
      output: {
        codeSplitting: false,
        entryFileNames: "language.worker.js",
        assetFileNames: "assets/[name]-[hash][extname]",
      },
    },
  },
});
