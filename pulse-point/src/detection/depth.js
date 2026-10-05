import { FLAGS } from '../lib/flags.js';
import { centerSquare, boxInCrop } from './depthSample.js';
import { isRemoteConfigured, isRemoteUp, remoteDepth } from './remote.js';

// Main-thread client for the metric depth worker. One measurement at a time; callers skip
// frames while it's busy and fall back to width-based distance when depth is unavailable.
//
// Depth runs on a square crop from the middle of the frame (the target is roughly centered by
// the time we measure), which halves its peak memory. One worker stays alive for the session:
// closing and reopening it every reading still crashed iPhone Safari.

const CROP_SIZE = FLAGS.depthSize;
// Measured bias of the crop vs a full-frame 518 px reading on indoor photos.
const CROP_CORRECTION = { 518: 1.03, 392: 1.15 }[CROP_SIZE];

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
  return !FLAGS.noDepth && !failed && typeof Worker !== 'undefined';
}

export function isDepthBusy() {
  return busy;
}

/** 'server', or 'webgpu'/'wasm' once the on-device worker has loaded a model, else null. */
export function getDepthBackend() {
  if (isRemoteUp() && isDepthAvailable()) return 'server';
  return backend;
}

/** Start downloading and compiling the depth model in the background. Never throws. */
export function preloadDepth() {
  // With a vision server, depth runs there; the on-device worker only loads if the server fails.
  if (!isDepthAvailable() || isRemoteConfigured()) return Promise.resolve(false);
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

  if (isRemoteUp()) {
    busy = true;
    try {
      return await remoteDepth(canvas, boxRel);
    } catch {
      return null;
    } finally {
      busy = false;
    }
  }

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
    busy = false;
  }
}
