import * as ort from 'onnxruntime-web/webgpu';
import { COCO_LABELS } from './coco.js';
import { FLAGS } from '../lib/flags.js';

ort.env.wasm.numThreads = 1;
ort.env.logLevel = 'error';

const INPUT_W = 640;
const INPUT_H = 640;
const CONF_THRESH = 0.25;
const IOU_THRESH  = 0.45;
const NUM_ANCHORS = 8400;
const PROMPT_DIM  = 512;

// YOLOE finds anything in the prompt pack; YOLO11n only knows the 80 COCO classes but is about
// 4× lighter on the CPU. With a GPU we use YOLOE alone. Without one, YOLO11n handles COCO
// targets and YOLOE loads on demand for everything else. If YOLOE fails to load, YOLO11n is
// the backup.
const models = {
  yoloe: makeSlot('/yoloe-11s.onnx'),
  yolo: makeSlot('/net.onnx'),
};

function makeSlot(url) {
  return { url, bytes: null, bytesPromise: null, session: null, sessionPromise: null, backend: null, failed: false };
}

let _gpuPromise = null;

function hasWebGPU() {
  if (!_gpuPromise) {
    _gpuPromise = (async () => {
      try {
        if (FLAGS.forceCpu) return false;
        return typeof navigator !== 'undefined' && !!navigator.gpu && !!(await navigator.gpu.requestAdapter());
      } catch {
        return false;
      }
    })();
  }
  return _gpuPromise;
}

function fetchBytes(slot) {
  if (slot.bytes) return Promise.resolve(slot.bytes);
  if (!slot.bytesPromise) {
    slot.bytesPromise = fetch(slot.url)
      .then(r => {
        if (!r.ok) throw new Error(`Failed to fetch ${slot.url}: ${r.status}`);
        return r.arrayBuffer();
      })
      .then(buf => (slot.bytes = buf))
      .catch(err => {
        slot.bytesPromise = null;
        throw err;
      });
  }
  return slot.bytesPromise;
}

async function warmup(slot, session) {
  const feeds = { images: new ort.Tensor('float32', new Float32Array(3 * INPUT_W * INPUT_H), [1, 3, INPUT_H, INPUT_W]) };
  if (slot === models.yoloe) {
    feeds.pe = new ort.Tensor('float32', new Float32Array(PROMPT_DIM), [1, 1, PROMPT_DIM]);
  }
  await session.run(feeds);
}

function openSession(slot, useGpu) {
  if (slot.session) return Promise.resolve(slot.session);
  if (!slot.sessionPromise) {
    slot.sessionPromise = (async () => {
      const bytes = await fetchBytes(slot);
      const options = { graphOptimizationLevel: 'all' };
      let session = null;
      if (useGpu) {
        try {
          session = await ort.InferenceSession.create(bytes, { ...options, executionProviders: ['webgpu', 'wasm'] });
          await warmup(slot, session);
          slot.backend = 'webgpu';
        } catch (err) {
          console.warn(`WebGPU unavailable for ${slot.url}, using WASM`, err);
          session = null;
        }
      }
      if (!session) {
        session = await ort.InferenceSession.create(bytes, { ...options, executionProviders: ['wasm'] });
        await warmup(slot, session);
        slot.backend = 'wasm';
      }
      slot.session = session;
      slot.bytes = null; // ORT has its own copy now; don't hold tens of MB twice
      slot.bytesPromise = null;
      return session;
    })().catch(err => {
      slot.sessionPromise = null;
      slot.failed = true;
      throw err;
    });
  }
  return slot.sessionPromise;
}

/** Start downloading whichever detector this device will use first. */
export async function preloadModel() {
  if (FLAGS.noDetect) return null;
  const gpu = await hasWebGPU();
  return fetchBytes(gpu ? models.yoloe : models.yolo);
}

export function isModelReady() {
  return !!(models.yoloe.session || models.yolo.session);
}

/** Which detectors are loaded and on what backend, for debugging. */
export function getDetectorInfo() {
  return { yoloe: models.yoloe.backend, yolo: models.yolo.backend };
}

export async function loadModel({ signal } = {}) {
  if (FLAGS.noDetect) return true;
  const gpu = await hasWebGPU();
  if (gpu) {
    try {
      await openSession(models.yoloe, true);
    } catch (err) {
      console.warn('YOLOE failed to load; using the YOLO11n backup', err);
      await openSession(models.yolo, true);
    }
  } else {
    await openSession(models.yolo, false);
  }
  if (signal?.aborted) throw new DOMException('Aborted', 'AbortError');
  return true;
}

function chooseDetector(prompts) {
  const { yoloe, yolo } = models;
  if (yoloe.session && yoloe.backend === 'webgpu') return 'yoloe';
  if (prompts.cocoLabel && yolo.session) return 'yolo';
  if (yoloe.session) return 'yoloe';
  if (!yoloe.failed) {
    // CPU device asking for something outside COCO: fetch YOLOE in the background. Local
    // inference returns no boxes until it is ready; the optional server probe is managed by App.
    openSession(yoloe, false).catch(err => console.warn('YOLOE failed to load', err));
  } else if (!yolo.session && !yolo.failed) {
    openSession(yolo, false).catch(err => console.warn('YOLO11n failed to load', err));
  }
  return null;
}

let _peCache = { key: null, tensor: null };

// One canvas and one input buffer for every frame. Allocating them per frame (~8 MB) makes
// iOS Safari hit its canvas-memory cap and kill the tab within seconds.
let _canvas = null;
let _ctx = null;
const _input = new Float32Array(3 * INPUT_W * INPUT_H);

function frameContext() {
  if (!_ctx) {
    _canvas = document.createElement('canvas');
    _canvas.width = INPUT_W;
    _canvas.height = INPUT_H;
    _ctx = _canvas.getContext('2d', { willReadFrequently: true });
  }
  return _ctx;
}

function promptTensor(prompts) {
  if (_peCache.key !== prompts.key) {
    _peCache = {
      key: prompts.key,
      tensor: new ort.Tensor('float32', prompts.data, [1, prompts.names.length, prompts.dim]),
    };
  }
  return _peCache.tensor;
}

/**
 * @param {HTMLVideoElement} video
 * @param {{ key: string, names: string[], dim: number, data: Float32Array, cocoLabel: string|null } | null} prompts
 *   Prompt set from detection/prompts.js; with no prompts there is nothing to look for.
 */
export async function runInference(video, prompts) {
  if (FLAGS.noDetect || !prompts?.names?.length) return [];
  const detector = chooseDetector(prompts);
  if (!detector) return [];

  const vw = video.videoWidth  || 640;
  const vh = video.videoHeight || 480;
  const scale = Math.min(INPUT_W / vw, INPUT_H / vh);
  const sw = Math.round(vw * scale);
  const sh = Math.round(vh * scale);
  const padX = Math.round((INPUT_W - sw) / 2);
  const padY = Math.round((INPUT_H - sh) / 2);

  const ctx = frameContext();
  ctx.fillStyle = '#808080';
  ctx.fillRect(0, 0, INPUT_W, INPUT_H);
  ctx.drawImage(video, padX, padY, sw, sh);

  const px = ctx.getImageData(0, 0, INPUT_W, INPUT_H).data;
  const n  = INPUT_W * INPUT_H;
  const buf = _input;
  for (let i = 0; i < n; i++) {
    buf[i]         = px[i * 4]     / 255;
    buf[n + i]     = px[i * 4 + 1] / 255;
    buf[2 * n + i] = px[i * 4 + 2] / 255;
  }
  if (FLAGS.captureOnly) return [];
  const images = new ort.Tensor('float32', buf, [1, 3, INPUT_H, INPUT_W]);

  let raw, numClasses, labelFor;
  if (detector === 'yoloe') {
    const out = await models.yoloe.session.run({ images, pe: promptTensor(prompts) });
    raw = out[Object.keys(out)[0]].data;
    numClasses = prompts.names.length;
    labelFor = cls => prompts.names[cls];
  } else {
    const out = await models.yolo.session.run({ images });
    raw = out[Object.keys(out)[0]].data;
    numClasses = COCO_LABELS.length;
    labelFor = cls => (COCO_LABELS[cls] === prompts.cocoLabel ? prompts.names[0] : COCO_LABELS[cls]);
  }

  const hits = [];
  for (let i = 0; i < NUM_ANCHORS; i++) {
    let best = 0, cls = 0;
    for (let c = 0; c < numClasses; c++) {
      const s = raw[(4 + c) * NUM_ANCHORS + i];
      if (s > best) { best = s; cls = c; }
    }
    if (best < CONF_THRESH) continue;

    const cx = raw[0 * NUM_ANCHORS + i];
    const cy = raw[1 * NUM_ANCHORS + i];
    const bw = raw[2 * NUM_ANCHORS + i];
    const bh = raw[3 * NUM_ANCHORS + i];

    hits.push({
      class: labelFor(cls),
      score: best,
      bbox: [((cx - bw / 2) - padX) / scale, ((cy - bh / 2) - padY) / scale, bw / scale, bh / scale],
    });
  }

  return nms(hits);
}

function nms(dets) {
  const sorted = [...dets].sort((a, b) => b.score - a.score);
  const kept = [];
  for (const d of sorted) {
    if (!kept.some(k => iou(d.bbox, k.bbox) > IOU_THRESH)) kept.push(d);
  }
  return kept;
}

function iou([ax, ay, aw, ah], [bx, by, bw, bh]) {
  const ix1 = Math.max(ax, bx), iy1 = Math.max(ay, by);
  const ix2 = Math.min(ax + aw, bx + bw), iy2 = Math.min(ay + ah, by + bh);
  const inter = Math.max(0, ix2 - ix1) * Math.max(0, iy2 - iy1);
  return inter / (aw * ah + bw * bh - inter || 1);
}
