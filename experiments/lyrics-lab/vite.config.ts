import { resolve } from "node:path";
import { defineConfig } from "vite";

const packageRoot = resolve(import.meta.dirname, "../..");

export default defineConfig({
  root: import.meta.dirname,
  resolve: {
    conditions: ["onnxruntime-web-use-extern-wasm"],
    dedupe: ["onnxruntime-web"],
  },
  define: {
    "process.env.NODE_ENV": JSON.stringify("production"),
  },
  server: {
    port: 3001,
    allowedHosts: [".ngrok-free.app"],
    fs: {
      allow: [packageRoot],
    },
  },
  preview: {
    port: 3001,
    allowedHosts: [".ngrok-free.app"],
  },
  build: {
    outDir: resolve(import.meta.dirname, "dist"),
    emptyOutDir: true,
    target: "es2022",
  },
  worker: {
    format: "es",
  },
});
