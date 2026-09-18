export const TARGET_SOURCES = Object.freeze(['typed', 'voice', 'suggestion']);

export const CANONICAL_LABELS = Object.freeze([
  'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
  'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
  'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
  'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
  'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
  'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
  'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
  'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
  'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
  'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
  'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
  'toothbrush',
]);

export const TARGET_ALIASES = Object.freeze({
  phone: 'cell phone',
  iphone: 'cell phone',
  android: 'cell phone',
  mobile: 'cell phone',
  'my phone': 'cell phone',
  'computer mouse': 'mouse',
  trackpad: 'mouse',
  tv: 'tv',
  television: 'tv',
  monitor: 'tv',
  screen: 'tv',
  sofa: 'couch',
  computer: 'laptop',
  macbook: 'laptop',
  notebook: 'laptop',
  remote: 'remote',
  'tv remote': 'remote',
  'remote control': 'remote',
  ship: 'boat',
  mug: 'cup',
  glass: 'cup',
  'water bottle': 'bottle',
});

export function normalizeTargetText(value) {
  return String(value ?? '')
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

export function resolveCanonicalLabel(value, labels = CANONICAL_LABELS) {
  const normalized = normalizeTargetText(value);
  if (!normalized) return null;
  const available = new Set(labels.map(normalizeTargetText));
  if (available.has(normalized)) return normalized;
  const alias = TARGET_ALIASES[normalized];
  return alias && available.has(alias) ? alias : null;
}

export function canonicalizeTarget(value, labels = CANONICAL_LABELS) {
  return resolveCanonicalLabel(value, labels) || normalizeTargetText(value) || null;
}

export function createTargetRequest(rawText, source = 'typed', labels = CANONICAL_LABELS) {
  const raw = String(rawText ?? '').trim();
  const normalizedText = normalizeTargetText(raw);
  const safeSource = TARGET_SOURCES.includes(source) ? source : 'typed';
  return {
    rawText: raw,
    normalizedText,
    canonicalLabel: resolveCanonicalLabel(normalizedText, labels),
    source: safeSource,
  };
}
