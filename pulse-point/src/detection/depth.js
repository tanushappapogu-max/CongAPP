// Metric depth on the GPU, sharing the detector's ONNX Runtime and GPU device. A separate worker
// meant a second runtime and GPU context, which phones can't afford, and CPU depth takes seconds
// per frame, so devices without WebGPU fall back to width-based distance instead.
import * as ort from 'onnxruntime-web/webgpu';
import { hasWebGPU, runExclusive } from './engine.js';
import { sampleBoxDepth } from './depthSample.js';

const MODEL_URL = '/depth-indoor-small.fp16.onnx';
const SHORT_SIDE = 518; // the model's training size; smaller inputs drift 10–25% in meters
const PATCH = 14;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

let sessionPromise = null;
let session = null;
let failed = false;
let busy = false;
let canvas = null;
let ctx = null;
let input = null;

function load() {
  if (!sessionPromise) {
    sessionPromise = (async () => {
      if (!(await hasWebGPU())) throw new Error('Depth needs WebGPU');
      const response = await fetch(MODEL_URL);
      if (!response.ok) throw new Error(`Failed to fetch depth model: ${response.status}`);
      const s = await ort.InferenceSession.create(new Uint8Array(await response.arrayBuffer()), {
        graphOptimizationLevel: 'all',
        executionProviders: ['webgpu', 'wasm'],
      });
      try {
        await runExclusive(() => s.run({
          pixel_values: new ort.Tensor('float32', new Float32Array(3 * SHORT_SIDE * SHORT_SIDE), [1, 3, SHORT_SIDE, SHORT_SIDE]),
        }));
      } catch (err) {
        await s.release().catch(() => {});
        throw err;
      }
      session = s;
      return s;
    })().catch(err => {
      failed = true;
      console.warn('Depth unavailable; using width-based distance', err);
      throw err;
    });
  }
  return sessionPromise;
}

export function isDepthBusy() {
  return busy;
}

/** 'webgpu' once the depth model is running, else null. */
export function getDepthBackend() {
  return session ? 'webgpu' : null;
}

/**
 * Measure how far the object in `bbox` is. The first call only starts loading the model (it is
 * ~50 MB, so it downloads after something has actually been found); later calls measure.
 * @param {HTMLVideoElement} video
 * @param {[number, number, number, number]} bbox  video pixels
 * @returns {Promise<number|null>} meters, or null when loading/busy/unavailable
 */
export async function measureDepth(video, bbox) {
  if (failed || busy) return null;
  if (!session) {
    load().catch(() => {});
    return null;
  }
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  if (!vw || !vh) return null;

  const scale = SHORT_SIDE / Math.min(vw, vh);
  const width = Math.max(PATCH, Math.round((vw * scale) / PATCH) * PATCH);
  const height = Math.max(PATCH, Math.round((vh * scale) / PATCH) * PATCH);
  if (!ctx) {
    canvas = document.createElement('canvas');
    ctx = canvas.getContext('2d', { willReadFrequently: true });
  }
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  ctx.drawImage(video, 0, 0, width, height);
  const rgba = ctx.getImageData(0, 0, width, height).data;
  const n = width * height;
  if (!input || input.length !== 3 * n) input = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    input[i]         = (rgba[i * 4]     / 255 - MEAN[0]) / STD[0];
    input[n + i]     = (rgba[i * 4 + 1] / 255 - MEAN[1]) / STD[1];
    input[2 * n + i] = (rgba[i * 4 + 2] / 255 - MEAN[2]) / STD[2];
  }
  const [x, y, w, h] = bbox;

  busy = true;
  try {
    const out = await runExclusive(() => session.run({ pixel_values: new ort.Tensor('float32', input, [1, 3, height, width]) }));
    const depth = out.predicted_depth;
    const [, dh, dw] = depth.dims;
    return sampleBoxDepth(depth.data, dw, dh, [x / vw, y / vh, w / vw, h / vh]);
  } catch {
    return null;
  } finally {
    busy = false;
  }
}
