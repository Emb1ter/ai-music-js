import { resolve } from "node:path";
import { defineConfig } from "vite";

export default defineConfig({
  root: import.meta.dirname,
  server: {
    allowedHosts: [".ngrok-free.app"],
  },
  preview: {
    allowedHosts: [".ngrok-free.app"],
  },
  build: {
    outDir: resolve(import.meta.dirname, "dist"),
    emptyOutDir: true,
  },
});
