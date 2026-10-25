import { normalizeTargetText, resolveCanonicalLabel } from './target.js';

export const DETECTION_SOURCES = Object.freeze([
  'web-onnx',
  'mobile-native',
  'server',
  'offline',
  'simulated',
]);

export const DISTANCE_METHODS = Object.freeze([
  'calibrated-pinhole',
  'area-estimate',
  'depth',
  'unknown',
]);

const clamp = (value, min, max) => Math.max(min, Math.min(max, value));

export function normalizeConfidence(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? clamp(numeric, 0, 1) : 0;
}

export function normalizeBBox(value) {
  if (!value) return null;
  const array = Array.isArray(value);
  const x = Number(array ? value[0] : value.x);
  const y = Number(array ? value[1] : value.y);
  const rawWidth = array ? value[2] : value.width;
  const rawHeight = array ? value[3] : value.height;
  const width = Number(rawWidth ?? (value.x2 != null ? Number(value.x2) - x : NaN));
  const height = Number(rawHeight ?? (value.y2 != null ? Number(value.y2) - y : NaN));
  if (![x, y, width, height].every(Number.isFinite) || width <= 0 || height <= 0) return null;

  const left = clamp(x, 0, 1);
  const top = clamp(y, 0, 1);
  const normalized = {
    x: left,
    y: top,
    width: clamp(width, 0, 1 - left),
    height: clamp(height, 0, 1 - top),
  };
  return normalized.width > 0 && normalized.height > 0 ? normalized : null;
}

export function normalizeDistance(value = {}) {
  const meters = Number(value.meters ?? value.distanceMeters);
  const uncertainty = Number(value.uncertaintyMeters);
  const method = DISTANCE_METHODS.includes(value.method) ? value.method : 'unknown';
  return {
    meters: Number.isFinite(meters) && meters >= 0 ? meters : null,
    method,
    uncertaintyMeters: Number.isFinite(uncertainty) && uncertainty >= 0 ? uncertainty : null,
  };
}

export function isSimulationSource(source) {
  return source === 'simulated' || source === 'offline';
}

export function normalizeDetection(input, options = {}) {
  if (!input) return null;
  const bbox = normalizeBBox(input.bbox ?? input.boundingBox);
  if (!bbox) return null;
  const source = DETECTION_SOURCES.includes(input.source)
    ? input.source
    : (options.source && DETECTION_SOURCES.includes(options.source) ? options.source : 'server');
  const timestamp = Number(input.timestampMs ?? options.nowMs ?? Date.now());
  const rawLabel = String(input.label ?? input.name ?? 'target');
  const canonicalLabel = resolveCanonicalLabel(rawLabel);
  const assistiveReady = !isSimulationSource(source)
    && input.assistiveReady !== false
    && Boolean(input.assistiveReady ?? options.assistiveReady ?? false);

  return {
    label: canonicalLabel || normalizeTargetText(rawLabel) || 'target',
    displayLabel: String(input.displayLabel ?? input.name ?? input.label ?? 'target'),
    bbox,
    confidence: normalizeConfidence(input.confidence),
    timestampMs: Number.isFinite(timestamp) ? timestamp : Date.now(),
    source,
    distance: normalizeDistance(input.distance ?? {
      meters: input.distanceMeters,
      method: input.distanceMethod,
      uncertaintyMeters: input.distanceUncertaintyMeters,
    }),
    assistiveReady,
    model: input.model ?? null,
  };
}

export function normalizeDetections(inputs, options = {}) {
  return (Array.isArray(inputs) ? inputs : [])
    .map((input) => normalizeDetection(input, options))
    .filter(Boolean);
}

export function selectDetection(inputs, target, options = {}) {
  const detections = normalizeDetections(inputs, options);
  const wanted = resolveCanonicalLabel(target) || normalizeTargetText(target);
  return detections
    .filter((item) => !wanted || item.label === wanted || normalizeTargetText(item.displayLabel) === wanted)
    .sort((a, b) => b.confidence - a.confidence || b.bbox.width * b.bbox.height - a.bbox.width * a.bbox.height || a.timestampMs - b.timestampMs)[0] || null;
}

export function detectionIsUsable(detection, { minConfidence = 0.35, nowMs = Date.now(), staleAfterMs = 1200 } = {}) {
  return Boolean(
    detection
    && detection.confidence >= minConfidence
    && detection.bbox
    && Number.isFinite(detection.timestampMs)
    && nowMs - detection.timestampMs <= staleAfterMs,
  );
}
