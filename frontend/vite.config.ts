import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  // Same proxy for dev and preview so `vite preview` behaves like `vite dev`.
  server: {
    port: 5173,
    // The backend is the single source of truth for config schema, analysis and
    // export. Proxying in dev keeps the frontend origin-agnostic so the same
    // build works when FastAPI serves it in production.
    proxy: { '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true } },
  },
  preview: {
    port: 5173,
    proxy: { '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true } },
  },
  build: { outDir: 'dist', sourcemap: true },
})
