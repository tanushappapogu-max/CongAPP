import { normalizeDetection } from './detection.js';

const DEFAULTS = Object.freeze({
  staleAfterMs: 600,
  snapIou: 0.3,
  smoothAlpha: 0.55,
  velocityEma: 0.55,
  maxSpeedPerMs: 0.01,
});

export class DetectionTracker {
  constructor(options = {}) {
    this.options = { ...DEFAULTS, ...options };
    this.reset();
  }

  reset() {
    this.detection = null;
    this.timestampMs = 0;
    this.velocity = { x: 0, y: 0 };
    this.fresh = false;
  }

  update(input, nowMs = Date.now()) {
    const detection = normalizeDetection(input, { nowMs });
    if (!detection) return null;

    const previous = this.detection;
    if (previous && nowMs > this.timestampMs) {
      const dt = nowMs - this.timestampMs;
      const overlap = iou(previous.bbox, detection.bbox);
      if (dt <= this.options.staleAfterMs && overlap >= this.options.snapIou) {
        const a = this.options.smoothAlpha;
        const blended = blendBox(previous.bbox, detection.bbox, a);
        const previousCenter = center(previous.bbox);
        const currentCenter = center(blended);
        const raw = {
          x: clamp((currentCenter.x - previousCenter.x) / dt, -this.options.maxSpeedPerMs, this.options.maxSpeedPerMs),
          y: clamp((currentCenter.y - previousCenter.y) / dt, -this.options.maxSpeedPerMs, this.options.maxSpeedPerMs),
        };
        const v = this.options.velocityEma;
        this.velocity = {
          x: v * raw.x + (1 - v) * this.velocity.x,
          y: v * raw.y + (1 - v) * this.velocity.y,
        };
        // Keep the canonical observation intact; smoothing is used for velocity
        // estimation so callers can distinguish measured from predicted boxes.
      } else {
        this.velocity = { x: 0, y: 0 };
      }
    } else {
      this.velocity = { x: 0, y: 0 };
    }

    this.detection = detection;
    this.timestampMs = nowMs;
    this.fresh = true;
    return this.predict(nowMs);
  }

  predict(nowMs = Date.now()) {
    if (!this.detection) return null;
    const ageMs = Math.max(0, nowMs - this.timestampMs);
    if (ageMs >= this.options.staleAfterMs) return null;
    const { bbox } = this.detection;
    const x = clamp(bbox.x + this.velocity.x * ageMs, 0, 1 - bbox.width);
    const y = clamp(bbox.y + this.velocity.y * ageMs, 0, 1 - bbox.height);
    return {
      ...this.detection,
      bbox: { ...bbox, x, y },
      confidence: this.detection.confidence * confidenceDecay(ageMs, this.options.staleAfterMs),
      ageMs,
      fresh: this.fresh && ageMs === 0,
    };
  }

  markStale() {
    this.fresh = false;
  }

  isAlive(nowMs = Date.now()) {
    return Boolean(this.predict(nowMs));
  }
}

export function confidenceDecay(ageMs, staleAfterMs = 600) {
  if (staleAfterMs <= 0) return 0;
  return Math.max(0, 1 - Math.max(0, ageMs) / staleAfterMs);
}

function center(box) {
  return { x: box.x + box.width / 2, y: box.y + box.height / 2 };
}

function blendBox(a, b, alpha) {
  return {
    x: alpha * b.x + (1 - alpha) * a.x,
    y: alpha * b.y + (1 - alpha) * a.y,
    width: alpha * b.width + (1 - alpha) * a.width,
    height: alpha * b.height + (1 - alpha) * a.height,
  };
}

export function iou(a, b) {
  const x1 = Math.max(a.x, b.x);
  const y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.width, b.x + b.width);
  const y2 = Math.min(a.y + a.height, b.y + b.height);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const union = a.width * a.height + b.width * b.height - inter;
  return union > 0 ? inter / union : 0;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}
