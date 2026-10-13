import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // Proxy /api requests to the FastAPI backend during development
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
  define: {
    // Fallback for VITE_API_URL when using the proxy
    "import.meta.env.VITE_API_URL": JSON.stringify(
      process.env.VITE_API_URL ?? ""
    ),
  },
});
