import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// During `vite dev`, proxy API calls to the backend so the dev server behaves
// like the nginx production setup (same-origin /api). In the built image nginx
// does this instead (see nginx.conf).
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": {
        target: process.env.DRIFTLY_API_URL || "http://localhost:8003",
        changeOrigin: true,
      },
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test/setup.ts"],
  },
});
