import { resolve } from "node:path";
import { defineConfig } from "vite";

export default defineConfig({
  publicDir: false,
  resolve: {
    alias: {
      "@": resolve(import.meta.dirname),
    },
    conditions: ["onnxruntime-web-use-extern-wasm"],
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
      entry: resolve(import.meta.dirname, "workers/ace-step.worker.ts"),
      formats: ["es"],
      fileName: "ace-step.worker",
    },
    rolldownOptions: {
      output: {
        codeSplitting: false,
        entryFileNames: "ace-step.worker.js",
        assetFileNames: "assets/[name]-[hash][extname]",
      },
    },
  },
});
