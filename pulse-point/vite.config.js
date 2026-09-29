import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

const ISOLATION_HEADERS = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
};

export default defineConfig({
  plugins: [react()],
  base: '/',
  // Same isolation headers as production (vercel.json) so multi-threaded WASM works in dev too.
  server: { headers: ISOLATION_HEADERS },
  preview: { headers: ISOLATION_HEADERS },
  optimizeDeps: {
    // Pre-bundling rewrites import.meta.url, which ONNX Runtime uses to locate its .wasm file.
    exclude: ['onnxruntime-web'],
  },
  worker: {
    format: 'es',
  },
});
