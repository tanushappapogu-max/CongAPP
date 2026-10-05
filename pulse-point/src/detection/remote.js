// Client for the vision server (server/vision): YOLOE detection and Depth Anything run there, so
// the phone loads no models at all. On iPhone, running both models in the browser kept getting the
// tab killed for memory; with the server the page only handles camera, guidance and feedback.
// Anything that fails here falls back to on-device inference.
import { FLAGS } from '../lib/flags.js';

// The Modal deployment (server/vision/modal_vision.py). VITE_VISION_URL overrides it; ?server=off disables it.
const DEFAULT_URL = import.meta.env.VITE_VISION_URL || 'https://tanush-appapogu--pulse-point-vision-fast.us-east.modal.direct';
const BASE = FLAGS.serverOff ? '' : (FLAGS.server || DEFAULT_URL).replace(/\/$/, '');
const JPEG_QUALITY = 0.8;
const REQUEST_TIMEOUT_MS = 6000;
const RETRY_AFTER_MS = 15000;

let state = BASE ? 'unknown' : 'off'; // off | unknown | checking | up | down
let checkPromise = null;
let downSince = 0;

export function isRemoteConfigured() {
  return Boolean(BASE);
}

export function isRemoteUp() {
  if (state === 'down' && Date.now() - downSince > RETRY_AFTER_MS) void checkRemote();
  return state === 'up';
}

function markDown() {
  state = 'down';
  downSince = Date.now();
}

/**
 * Health check. A sleeping server answers 503 for the ~20–60 s it takes to start, so keep asking
 * until it's up or the time runs out.
 */
export function checkRemote(timeoutMs = 90000) {
  if (!BASE) return Promise.resolve(false);
  if (!checkPromise) {
    state = 'checking';
    checkPromise = waitForHealth(Date.now() + timeoutMs)
      .then(ok => {
        if (ok) state = 'up';
        else markDown();
        return ok;
      })
      .finally(() => {
        checkPromise = null;
      });
  }
  return checkPromise;
}

async function waitForHealth(deadline) {
  while (Date.now() < deadline) {
    try {
      const r = await fetch(`${BASE}/health`, { signal: AbortSignal.timeout(Math.max(1000, deadline - Date.now())) });
      if (r.ok) return true;
      if (r.status !== 503) return false;
    } catch {
      // The proxy's 503 may lack CORS headers, which surfaces here as a network error: keep waiting.
    }
    await new Promise(resolve => setTimeout(resolve, 2000));
  }
  return false;
}

// toDataURL is synchronous (~5 ms for 640×640); toBlob runs as a low-priority task and took ~1 s
// in testing, which would cap detection at about one frame per second.
function toJpeg(canvas) {
  const b64 = canvas.toDataURL('image/jpeg', JPEG_QUALITY).split(',')[1];
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Blob([bytes], { type: 'image/jpeg' });
}

async function post(path, fields, canvas) {
  const form = new FormData();
  for (const [k, v] of Object.entries(fields)) form.append(k, v);
  form.append('file', toJpeg(canvas), 'frame.jpg');
  try {
    const r = await fetch(`${BASE}${path}`, { method: 'POST', body: form, signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS) });
    if (r.status === 429 || r.status === 503) return null; // rate limited / container restarting: skip this frame
    if (!r.ok) throw new Error(`${path} ${r.status}`);
    return await r.json();
  } catch (err) {
    markDown();
    throw err;
  }
}

/**
 * @param {HTMLCanvasElement} letterboxed  the 640×640 letterboxed frame the on-device detector would use
 * @returns {Promise<Array<{class: string, score: number, bbox: number[]}>|null>} boxes in 640 px input space
 */
export async function remoteDetect(letterboxed, target) {
  const data = await post('/v1/detect', { target }, letterboxed);
  return data ? data.detections : null;
}

/** @returns {Promise<number|null>} meters (crop correction already applied by the server) */
export async function remoteDepth(crop, boxRel) {
  const data = await post('/v1/depth', { box: boxRel.map(v => v.toFixed(5)).join(',') }, crop);
  return data ? data.meters : null;
}
