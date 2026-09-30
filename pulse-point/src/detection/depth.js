import { FLAGS } from '../lib/flags.js';
import { centerSquare, boxInCrop } from './depthSample.js';

// Main-thread client for the metric depth worker. One measurement at a time; callers skip
// frames while it's busy and fall back to width-based distance when depth is unavailable.
//
// Depth runs on a square crop from the middle of the frame (the target is roughly centered by
// the time we measure). On the CPU path (phones) the worker is closed after every reading so its
// memory is handed back instead of sitting next to the detector for the whole session.

const CROP_SIZE = FLAGS.depthSize;
// Measured bias of the crop vs a full-frame 518 px reading on indoor photos.
const CROP_CORRECTION = { 518: 1.03, 392: 1.15 }[CROP_SIZE];
const RELEASE_AFTER_READING = FLAGS.forceCpu;

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

/** Close the worker; a closed worker returns all of its memory, which a live one never does. */
function releaseWorker() {
  if (!worker) return;
  worker.terminate();
  worker = null;
  rejectAll(new Error('Depth worker released'));
}

function call(message, transfer = []) {
  return new Promise((resolve, reject) => {
    const id = nextId++;
    pending.set(id, { resolve, reject });
    getWorker().postMessage({ ...message, id }, transfer);
  });
}

export function isDepthAvailable() {
  return !FLAGS.noDepth && !failed && typeof Worker !== 'undefined';
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
  // When the worker is released after every reading there is nothing worth keeping warm.
  if (!isDepthAvailable() || RELEASE_AFTER_READING) return Promise.resolve(false);
  return call({ type: 'load', forceCpu: FLAGS.forceCpu }).then(() => true).catch(() => {
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
  const crop = centerSquare(vw, vh);
  const boxRel = boxInCrop(bbox, crop);
  if (!boxRel) return null;

  const width = CROP_SIZE;
  const height = CROP_SIZE;
  canvas ||= document.createElement('canvas');
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) return null;
  ctx.drawImage(video, crop.left, crop.top, crop.side, crop.side, 0, 0, width, height);
  const pixels = ctx.getImageData(0, 0, width, height).data.buffer;

  busy = true;
  try {
    const result = await call(
      { type: 'run', forceCpu: FLAGS.forceCpu, pixels, width, height, boxRel },
      [pixels],
    );
    return result.meters == null ? null : result.meters / CROP_CORRECTION;
  } catch {
    return null;
  } finally {
    if (RELEASE_AFTER_READING) releaseWorker();
    busy = false;
  }
}
