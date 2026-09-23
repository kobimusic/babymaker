import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// relative base, so dist/ works from any path (a subfolder, GitHub Pages, file server)
export default defineConfig({
  base: "./",
  plugins: [react()],
  worker: { format: "es" },
});
