// Runs the metric depth model off the main thread so detection, haptics and speech never stall.
import * as ort from 'onnxruntime-web';
import { sampleBoxDepth } from './depthSample.js';

ort.env.wasm.wasmPaths = '/';
ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = true;

const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

let sessionPromise = null;

function getSession(url) {
  if (!sessionPromise) {
    sessionPromise = ort.InferenceSession.create(url, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    }).catch(err => {
      sessionPromise = null;
      throw err;
    });
  }
  return sessionPromise;
}

function toTensor(pixels, width, height) {
  const rgba = new Uint8ClampedArray(pixels);
  const n = width * height;
  const buf = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    buf[i]         = (rgba[i * 4]     / 255 - MEAN[0]) / STD[0];
    buf[n + i]     = (rgba[i * 4 + 1] / 255 - MEAN[1]) / STD[1];
    buf[2 * n + i] = (rgba[i * 4 + 2] / 255 - MEAN[2]) / STD[2];
  }
  return new ort.Tensor('float32', buf, [1, 3, height, width]);
}

self.onmessage = async ({ data }) => {
  const { id, type, url } = data;
  try {
    const session = await getSession(url);
    if (type === 'load') {
      self.postMessage({ id, ok: true });
      return;
    }
    const t0 = performance.now();
    const out = await session.run({ pixel_values: toTensor(data.pixels, data.width, data.height) });
    const depth = out.predicted_depth;
    const [, dh, dw] = depth.dims;
    const meters = sampleBoxDepth(depth.data, dw, dh, data.boxRel);
    self.postMessage({ id, ok: true, meters, ms: Math.round(performance.now() - t0) });
  } catch (err) {
    self.postMessage({ id, ok: false, error: String(err?.message || err) });
  }
};
