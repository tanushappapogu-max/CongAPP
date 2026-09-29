// Main-thread client for the metric depth worker. One measurement at a time; callers skip
// frames while it's busy and fall back to width-based distance when depth is unavailable.

const SHORT_SIDE = 518; // the model's training size; smaller inputs drift 10–25% in meters
const PATCH = 14;

let worker = null;
let failed = false;
let backend = null;
let busy = false;
let nextId = 1;
const pending = new Map();
let canvas = null;

function rejectAll(error) {
  for (const { reject } of pending.values()) reject(error);
  pending.clear();
}

function getWorker() {
  if (!worker) {
    worker = new Worker(new URL('./depth.worker.js', import.meta.url), { type: 'module' });
    worker.onmessage = ({ data }) => {
      const entry = pending.get(data.id);
      if (!entry) return;
      pending.delete(data.id);
      if (data.backend) backend = data.backend;
      if (data.ok) entry.resolve(data);
      else entry.reject(new Error(data.error));
    };
    worker.onerror = (event) => {
      failed = true;
      rejectAll(new Error(event.message || 'Depth worker crashed'));
    };
  }
  return worker;
}

function call(message, transfer = []) {
  return new Promise((resolve, reject) => {
    const id = nextId++;
    pending.set(id, { resolve, reject });
    getWorker().postMessage({ ...message, id }, transfer);
  });
}

export function isDepthAvailable() {
  return !failed && typeof Worker !== 'undefined';
}

export function isDepthBusy() {
  return busy;
}

/** 'webgpu' or 'wasm' once the worker has loaded a model, else null. */
export function getDepthBackend() {
  return backend;
}

/** Start downloading and compiling the depth model in the background. Never throws. */
export function preloadDepth() {
  if (!isDepthAvailable()) return Promise.resolve(false);
  return call({ type: 'load' }).then(() => true).catch(() => {
    failed = true;
    return false;
  });
}

/**
 * Measure how far the object in `bbox` is. The frame is captured synchronously, so the
 * reading matches the box even though the model finishes seconds later.
 * @param {HTMLVideoElement} video
 * @param {[number, number, number, number]} bbox  video pixels
 * @returns {Promise<number|null>} meters, or null when busy/unavailable/failed
 */
export async function measureDepth(video, bbox) {
  if (busy || !isDepthAvailable()) return null;
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  if (!vw || !vh) return null;

  const scale = SHORT_SIDE / Math.min(vw, vh);
  const width = Math.max(PATCH, Math.round((vw * scale) / PATCH) * PATCH);
  const height = Math.max(PATCH, Math.round((vh * scale) / PATCH) * PATCH);
  canvas ||= document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) return null;
  ctx.drawImage(video, 0, 0, width, height);
  const pixels = ctx.getImageData(0, 0, width, height).data.buffer;
  const [x, y, w, h] = bbox;

  busy = true;
  try {
    const result = await call(
      { type: 'run', pixels, width, height, boxRel: [x / vw, y / vh, w / vw, h / vh] },
      [pixels],
    );
    return result.meters ?? null;
  } catch {
    return null;
  } finally {
    busy = false;
  }
}
