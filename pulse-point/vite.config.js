import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  base: '/',
  optimizeDeps: {
    // Pre-bundling rewrites import.meta.url, which ONNX Runtime uses to locate its .wasm file.
    exclude: ['onnxruntime-web'],
  },
  worker: {
    format: 'es',
  },
});
