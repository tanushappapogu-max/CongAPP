// Runs the metric depth model off the main thread so detection, haptics and speech never stall.
import * as ort from 'onnxruntime-web/webgpu';
import { sampleBoxDepth } from './depthSample.js';

ort.env.wasm.numThreads = 1;
ort.env.logLevel = 'error';

// Same graph, two encodings: fp16 for the GPU, uint8 for the CPU (uint8 ops don't run on WebGPU).
const MODELS = {
  webgpu: '/depth-indoor-small.fp16.onnx',
  wasm: '/depth-indoor-small.uint8.onnx',
};
const OPTIONS = { graphOptimizationLevel: 'all' };
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

let sessionPromise = null;
let backend = null;
let inputBuf = null;

async function fetchModel(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Failed to fetch depth model: ${response.status}`);
  return new Uint8Array(await response.arrayBuffer());
}

async function createSession() {
  if (self.navigator?.gpu) {
    try {
      const session = await ort.InferenceSession.create(await fetchModel(MODELS.webgpu), {
        ...OPTIONS,
        executionProviders: ['webgpu', 'wasm'],
      });
      // Shader compile failures only show up on the first run, so find out now rather than mid-scan.
      await session.run({ pixel_values: new ort.Tensor('float32', new Float32Array(3 * 518 * 518), [1, 3, 518, 518]) });
      backend = 'webgpu';
      return session;
    } catch (err) {
      console.warn('Depth: WebGPU unavailable, using WASM', err);
    }
  }
  const session = await ort.InferenceSession.create(await fetchModel(MODELS.wasm), {
    ...OPTIONS,
    executionProviders: ['wasm'],
  });
  backend = 'wasm';
  return session;
}

function getSession() {
  if (!sessionPromise) {
    sessionPromise = createSession().catch(err => {
      sessionPromise = null;
      throw err;
    });
  }
  return sessionPromise;
}

function toTensor(pixels, width, height) {
  const rgba = new Uint8ClampedArray(pixels);
  const n = width * height;
  if (!inputBuf || inputBuf.length !== 3 * n) inputBuf = new Float32Array(3 * n);
  const buf = inputBuf;
  for (let i = 0; i < n; i++) {
    buf[i]         = (rgba[i * 4]     / 255 - MEAN[0]) / STD[0];
    buf[n + i]     = (rgba[i * 4 + 1] / 255 - MEAN[1]) / STD[1];
    buf[2 * n + i] = (rgba[i * 4 + 2] / 255 - MEAN[2]) / STD[2];
  }
  return new ort.Tensor('float32', buf, [1, 3, height, width]);
}

self.onmessage = async ({ data }) => {
  const { id, type } = data;
  try {
    const session = await getSession();
    if (type === 'load') {
      self.postMessage({ id, ok: true, backend });
      return;
    }
    const t0 = performance.now();
    const out = await session.run({ pixel_values: toTensor(data.pixels, data.width, data.height) });
    const depth = out.predicted_depth;
    const [, dh, dw] = depth.dims;
    const meters = sampleBoxDepth(depth.data, dw, dh, data.boxRel);
    self.postMessage({ id, ok: true, meters, backend, ms: Math.round(performance.now() - t0) });
  } catch (err) {
    self.postMessage({ id, ok: false, error: String(err?.message || err) });
  }
};
