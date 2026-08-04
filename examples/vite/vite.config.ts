import { resolve } from "node:path";
import { defineConfig } from "vite";

const packageRoot = resolve(import.meta.dirname, "../..");

export default defineConfig({
  root: import.meta.dirname,
  server: {
    allowedHosts: [".ngrok-free.app"],
    ...(process.env["VITE_ACE_MODEL_FS_ROOT"]
      ? {
          fs: {
            allow: [
              packageRoot,
              process.env["VITE_ACE_MODEL_FS_ROOT"],
            ],
          },
        }
      : {}),
  },
  preview: {
    allowedHosts: [".ngrok-free.app"],
  },
  build: {
    outDir: resolve(import.meta.dirname, "dist"),
    emptyOutDir: true,
  },
});
