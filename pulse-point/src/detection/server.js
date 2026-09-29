/**
 * Visual grounding via the LocateAnything-3B server (Modal GPU, see server/modal_app.py).
 *
 * Only active when VITE_SERVER_URL is set. Grounding models never run in the browser: they are
 * hundreds of MB and block the main thread for seconds per frame, which froze scanning.
 *
 *   detectWithServer(video, target) → result object | null
 *   isServerAvailable()            → bool
 */

const SERVER_URL = import.meta.env.VITE_SERVER_URL || '';
const RECHECK_MS = 30_000;

let _serverOk = null;  // null = unknown, true/false = last health check
let _serverCheck = 0;

async function _checkServer() {
  _serverCheck = Date.now();
  try {
    const r = await fetch(`${SERVER_URL}/health`, { signal: AbortSignal.timeout(4000) });
    _serverOk = r.ok;
  } catch {
    _serverOk = false;
  }
  return _serverOk;
}

if (SERVER_URL) _checkServer();

export function isServerAvailable() {
  if (!SERVER_URL) return false;
  if (Date.now() - _serverCheck > RECHECK_MS) void _checkServer();
  return _serverOk === true;
}

export async function detectWithServer(video, target = '') {
  if (!SERVER_URL || !video || video.readyState < 2 || !target) return null;

  const W = video.videoWidth  || 640;
  const H = video.videoHeight || 480;
  const canvas = document.createElement('canvas');
  canvas.width  = W;
  canvas.height = H;
  canvas.getContext('2d').drawImage(video, 0, 0, W, H);

  try {
    const blob = await new Promise(res => canvas.toBlob(res, 'image/jpeg', 0.85));
    const form = new FormData();
    form.append('file', blob, 'frame.jpg');
    form.append('target', target);

    const t0 = performance.now();
    const r = await fetch(`${SERVER_URL}/detect`, {
      method: 'POST',
      body: form,
      signal: AbortSignal.timeout(8000),
    });
    if (!r.ok) return null;
    const data = await r.json();
    if (!data.detected) return null;
    const { x, y, width, height } = data.boundingBox;
    return {
      class:        data.name || target,
      score:        data.confidence ?? 0.90,
      bbox:         [x * W, y * H, width * W, height * H],
      fromServer:   true,
      model:        'LocateAnything',
      alternatives: [],
      latency_ms:   Math.round(performance.now() - t0),
    };
  } catch (e) {
    _serverOk = false;
    console.warn('[Ground] Server error:', e.message);
    return null;
  }
}
