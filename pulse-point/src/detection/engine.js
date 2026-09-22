import * as ort from 'onnxruntime-web';

ort.env.wasm.wasmPaths = '/';
ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = true;

const INPUT_W = 640;
const INPUT_H = 640;
const CONF_THRESH = 0.25;
const IOU_THRESH  = 0.45;
const NUM_CLASSES = 80;
const NUM_ANCHORS = 8400;
const MODEL_URL   = '/net.onnx';

const CLASSES = [
  'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
  'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat',
  'dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack',
  'umbrella','handbag','tie','suitcase','frisbee','skis','snowboard',
  'sports ball','kite','baseball bat','baseball glove','skateboard','surfboard',
  'tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl',
  'banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza',
  'donut','cake','chair','couch','potted plant','bed','dining table','toilet',
  'tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven',
  'toaster','sink','refrigerator','book','clock','vase','scissors',
  'teddy bear','hair drier','toothbrush',
];

// ── Singleton session & preload state ────────────────────────────────────────
let _session = null;
let _preloadPromise = null;  // shared in-flight Promise<ArrayBuffer>
let _modelBuffer = null;     // resolved ArrayBuffer once downloaded
let _loadPromise = null;     // shared in-flight Promise<session>
let _warmedUp = false;

// ── Progress helpers ──────────────────────────────────────────────────────────
/**
 * Report a progress event. Shape:
 *   { step, loaded?, total?, percent?, fromCache?, message? }
 * @param {Function|null} onProgress
 * @param {object} data
 */
function report(onProgress, data) {
  if (typeof onProgress === 'function') {
    try { onProgress(data); } catch { /* never block boot on listener errors */ }
  }
}

// ── Model byte fetch with streaming progress ──────────────────────────────────
/**
 * Download the ONNX model with streaming progress. Returns an ArrayBuffer.
 * Detects whether the response came from the service-worker cache via
 * the response timestamp header that the SW adds, or by checking Cache API.
 *
 * @param {{ onProgress?: Function, signal?: AbortSignal }} opts
 * @returns {Promise<ArrayBuffer>}
 */
async function _fetchModelBuffer({ onProgress, signal } = {}) {
  // Detect cache hit by peeking into Cache Storage first (best-effort).
  let fromCache = false;
  try {
    const caches_ = typeof caches !== 'undefined' ? caches : null;
    if (caches_) {
      const cache = await caches_.open('pulse-point-model-v2');
      const cached = await cache.match(MODEL_URL);
      if (cached) fromCache = true;
    }
  } catch { /* ignore — browsers without cache API still work */ }

  report(onProgress, { step: 'download', loaded: 0, total: 0, percent: 0, fromCache, message: fromCache ? 'Loading from offline cache…' : 'Downloading neural weights…' });

  const response = await fetch(MODEL_URL, { signal });
  if (!response.ok) throw new Error(`Failed to fetch model: ${response.status} ${response.statusText}`);

  const contentLength = Number(response.headers.get('content-length')) || 0;
  const reader = response.body?.getReader();

  // Fallback: no streaming reader → download whole blob at once
  if (!reader) {
    report(onProgress, { step: 'download', loaded: 0, total: contentLength, percent: 50, fromCache, message: fromCache ? 'Loading from cache…' : 'Downloading (streaming unavailable)…' });
    const buffer = await response.arrayBuffer();
    report(onProgress, { step: 'download', loaded: buffer.byteLength, total: buffer.byteLength, percent: 100, fromCache, message: 'Weights loaded.' });
    return buffer;
  }

  const chunks = [];
  let loaded = 0;
  // eslint-disable-next-line no-constant-condition
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    const percent = contentLength > 0 ? Math.min(99, Math.round((loaded / contentLength) * 100)) : 0;
    report(onProgress, { step: 'download', loaded, total: contentLength, percent, fromCache, message: fromCache ? 'Loading from cache…' : `Downloading neural weights… ${percent}%` });
  }

  // Concat all chunks into a single ArrayBuffer
  const totalBytes = chunks.reduce((acc, c) => acc + c.byteLength, 0);
  const result = new Uint8Array(totalBytes);
  let offset = 0;
  for (const chunk of chunks) {
    result.set(chunk, offset);
    offset += chunk.byteLength;
  }
  report(onProgress, { step: 'download', loaded: totalBytes, total: totalBytes, percent: 100, fromCache, message: 'Weights ready.' });
  return result.buffer;
}

// ── Preload — call early (e.g. on mount) to warm the cache in the background ──
/**
 * Kick off a background download of the model bytes. Safe to call multiple times.
 * Returns a Promise that resolves when the buffer is ready.
 * @param {{ onProgress?: Function, signal?: AbortSignal }} opts
 */
export function preloadModel(opts = {}) {
  if (_modelBuffer) {
    report(opts.onProgress, { step: 'download', loaded: 1, total: 1, percent: 100, fromCache: true, message: 'Loaded from offline cache.' });
    return Promise.resolve(_modelBuffer);
  }
  if (!_preloadPromise) {
    _preloadPromise = _fetchModelBuffer(opts).then(buf => {
      _modelBuffer = buf;
      return buf;
    }).catch(err => {
      _preloadPromise = null; // allow retry on next call
      throw err;
    });
  }
  // If a preload is already in flight, we can't attach new progress listeners
  // to the same stream. Just return the shared promise (caller will get the buffer).
  return _preloadPromise;
}

/** Returns true if the model session is already loaded and warmed up. */
export function isModelReady() {
  return _session !== null && _warmedUp;
}

// ── Warm-up: run one dummy inference to pre-compile JIT kernels ────────────────
async function _warmupSession(session, onProgress) {
  report(onProgress, { step: 'warmup', percent: 0, message: 'Warming up inference engine…' });
  const dummyBuf = new Float32Array(3 * INPUT_W * INPUT_H); // all zeros
  const dummyInput = new ort.Tensor('float32', dummyBuf, [1, 3, INPUT_H, INPUT_W]);
  try {
    await session.run({ images: dummyInput });
  } catch {
    // Warm-up failure is non-fatal; the real first inference may be slower but will work.
  }
  _warmedUp = true;
  report(onProgress, { step: 'warmup', percent: 100, message: 'Engine ready.' });
}

// ── Primary loadModel with per-step progress callbacks ────────────────────────
/**
 * Load (or return the cached) ONNX inference session.
 *
 * Progress callbacks receive objects with:
 *   { step: 'download'|'compile'|'warmup', percent, loaded?, total?, fromCache?, message }
 *
 * @param {{ onProgress?: Function, signal?: AbortSignal, skipWarmup?: boolean }} opts
 * @returns {Promise<InferenceSession>}
 */
export async function loadModel({ onProgress, signal, skipWarmup = false } = {}) {
  if (_session && _warmedUp) return _session;
  if (_session && skipWarmup) return _session;

  // Serialize concurrent load calls
  if (_loadPromise) return _loadPromise;

  _loadPromise = _doLoad({ onProgress, signal, skipWarmup }).finally(() => {
    _loadPromise = null;
  });
  return _loadPromise;
}

async function _doLoad({ onProgress, signal, skipWarmup }) {
  if (_session && (_warmedUp || skipWarmup)) return _session;

  // ── Step 1: Download/fetch model bytes ────────────────────────────────────
  let buffer;
  if (_modelBuffer) {
    // Already preloaded
    report(onProgress, { step: 'download', loaded: _modelBuffer.byteLength, total: _modelBuffer.byteLength, percent: 100, fromCache: true, message: 'Loaded from offline cache.' });
    buffer = _modelBuffer;
  } else {
    buffer = await _fetchModelBuffer({ onProgress, signal });
    _modelBuffer = buffer;
  }
  if (signal?.aborted) throw new DOMException('Aborted', 'AbortError');

  // ── Step 2: Compile ONNX graph ────────────────────────────────────────────
  report(onProgress, { step: 'compile', percent: 0, message: 'Compiling WASM SIMD graph…' });
  if (!_session) {
    _session = await ort.InferenceSession.create(buffer, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });
  }
  report(onProgress, { step: 'compile', percent: 100, message: 'Graph compiled.' });
  if (signal?.aborted) throw new DOMException('Aborted', 'AbortError');

  // ── Step 3: Warmup ────────────────────────────────────────────────────────
  if (!skipWarmup && !_warmedUp) {
    await _warmupSession(_session, onProgress);
  }
  if (signal?.aborted) throw new DOMException('Aborted', 'AbortError');

  return _session;
}

// ── Inference ─────────────────────────────────────────────────────────────────
export async function runInference(video) {
  const session = _session || await loadModel();

  const vw = video.videoWidth  || 640;
  const vh = video.videoHeight || 480;

  const scale = Math.min(INPUT_W / vw, INPUT_H / vh);
  const sw = Math.round(vw * scale);
  const sh = Math.round(vh * scale);
  const pad_x = Math.round((INPUT_W - sw) / 2);
  const pad_y = Math.round((INPUT_H - sh) / 2);

  const canvas = document.createElement('canvas');
  canvas.width  = INPUT_W;
  canvas.height = INPUT_H;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#808080';
  ctx.fillRect(0, 0, INPUT_W, INPUT_H);
  ctx.drawImage(video, pad_x, pad_y, sw, sh);

  const px = ctx.getImageData(0, 0, INPUT_W, INPUT_H).data;
  const n  = INPUT_W * INPUT_H;
  const buf = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    buf[i]         = px[i * 4]     / 255;
    buf[n + i]     = px[i * 4 + 1] / 255;
    buf[2 * n + i] = px[i * 4 + 2] / 255;
  }

  const input = new ort.Tensor('float32', buf, [1, 3, INPUT_H, INPUT_W]);
  const out   = await session.run({ images: input });
  const raw   = out[Object.keys(out)[0]].data;

  const hits = [];
  for (let i = 0; i < NUM_ANCHORS; i++) {
    let best = 0, cls = 0;
    for (let c = 0; c < NUM_CLASSES; c++) {
      const s = raw[(4 + c) * NUM_ANCHORS + i];
      if (s > best) { best = s; cls = c; }
    }
    if (best < CONF_THRESH) continue;

    const cx = raw[0 * NUM_ANCHORS + i];
    const cy = raw[1 * NUM_ANCHORS + i];
    const bw = raw[2 * NUM_ANCHORS + i];
    const bh = raw[3 * NUM_ANCHORS + i];

    const x = ((cx - bw / 2) - pad_x) / scale;
    const y = ((cy - bh / 2) - pad_y) / scale;
    const w = bw / scale;
    const h = bh / scale;

    hits.push({ class: CLASSES[cls], score: best, bbox: [x, y, w, h] });
  }

  return _nms(hits);
}

function _nms(dets) {
  const sorted = [...dets].sort((a, b) => b.score - a.score);
  const kept = [];
  for (const d of sorted) {
    if (!kept.some(k => _iou(d.bbox, k.bbox) > IOU_THRESH)) kept.push(d);
  }
  return kept;
}

function _iou([ax, ay, aw, ah], [bx, by, bw, bh]) {
  const ix1 = Math.max(ax, bx), iy1 = Math.max(ay, by);
  const ix2 = Math.min(ax + aw, bx + bw), iy2 = Math.min(ay + ah, by + bh);
  const inter = Math.max(0, ix2 - ix1) * Math.max(0, iy2 - iy1);
  return inter / (aw * ah + bw * bh - inter || 1);
}
