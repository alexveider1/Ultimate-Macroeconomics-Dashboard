/// <reference types="vitest/config" />
import { fileURLToPath, URL } from "node:url";

import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The BFF (read-only backend-for-frontend) is the single API origin. In dev we
// proxy `/api/*` to it so the browser talks same-origin (no CORS); in prod nginx
// does the same. The BFF routes are un-prefixed (`/worldbank`, `/config`, …), so
// the `/api` prefix is stripped before forwarding.
const BFF_TARGET = process.env.BFF_TARGET ?? "http://localhost:8005";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: BFF_TARGET,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
  test: {
    globals: true,
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
    css: false,
    exclude: ["**/node_modules/**", "**/e2e/**", "**/dist/**"],
  },
});
