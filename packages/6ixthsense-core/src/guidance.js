import { isSimulationSource } from './detection.js';

export const DEFAULT_GUIDANCE_POLICY = Object.freeze({
  horizontal: { left: 0.40, right: 0.60 },
  vertical: { top: 0.38, bottom: 0.62 },
  reachMeters: 0.55,
  reachArea: 0.20,
  allowAreaEstimateReach: true,
  nearGrowth: 1.05,
  minConfidence: 0.35,
  staleAfterMs: 600,
  lostAfterMs: 1400,
});

export const GUIDANCE_SIGNALS = Object.freeze([
  'looking', 'left', 'right', 'up', 'down', 'locked', 'closer', 'lost', 'reach',
]);

export function computeGuidance(detection, options = {}) {
  const policy = mergePolicy(options.policy);
  const previousArea = Number(options.previousArea) || 0;
  const nowMs = Number(options.nowMs ?? Date.now());
  const ageMs = Math.max(0, nowMs - Number(detection?.timestampMs ?? nowMs));

  if (!detection || ageMs > policy.staleAfterMs) {
    return createUncertainState(detection, 'lost', 'Lost sight of the target.', ageMs);
  }

  const confidence = Number(detection.confidence) || 0;
  if (confidence < policy.minConfidence) {
    return createUncertainState(
      detection,
      'looking',
      `Looking for ${displayName(detection)}.`,
      ageMs,
    );
  }

  const { x, y, width, height } = detection.bbox;
  const cx = x + width / 2;
  const cy = y + height / 2;
  const area = width * height;
  const inH = cx >= policy.horizontal.left && cx <= policy.horizontal.right;
  const inV = cy >= policy.vertical.top && cy <= policy.vertical.bottom;
  const distanceMeters = detection.distance?.meters ?? null;
  const distanceText = describeDistance(distanceMeters, area, detection.distance?.method);
  const proximity = proximityFor(distanceMeters, area, detection.distance?.method, policy);
  const gettingNear = previousArea > 0 && area > previousArea * policy.nearGrowth;

  let signal;
  let direction;
  if (inH && inV && proximity === 'reach') {
    signal = 'reach';
    direction = 'right in front of you';
  } else if (inH && inV) {
    signal = gettingNear ? 'closer' : 'locked';
    direction = 'centered';
  } else {
    const horizontalError = inH ? 0 : Math.abs(cx - 0.5);
    const verticalError = inV ? 0 : Math.abs(cy - 0.5);
    if (horizontalError >= verticalError) {
      signal = cx < 0.5 ? 'left' : 'right';
      direction = signal === 'left' ? 'turn left' : 'turn right';
    } else {
      signal = cy < 0.5 ? 'up' : 'down';
      direction = signal === 'up' ? 'tilt up' : 'tilt down';
    }
  }

  const assistiveReady = Boolean(detection.assistiveReady) && !isSimulationSource(detection.source);
  const state = {
    status: signal,
    signal,
    direction,
    proximity,
    confidence,
    distanceText,
    distanceMeters,
    sentence: sentenceFor(detection, signal, distanceText, distanceMeters),
    speechPhrase: speechFor(signal),
    assistiveReady,
    source: detection.source,
    detection,
    ageMs,
    area,
    cx,
    cy,
  };

  if (!assistiveReady) {
    state.sentence = `${state.sentence} Experimental ${detection.source} guidance.`;
    state.speechPhrase = `${state.speechPhrase}. Experimental mode.`;
  }
  return state;
}

export function lostGuidance(previousState = null, { nowMs = Date.now(), policy } = {}) {
  const ageMs = Math.max(0, nowMs - Number(previousState?.lastDetectionAt ?? nowMs));
  const lost = ageMs > mergePolicy(policy).lostAfterMs;
  return {
    status: lost ? 'lost' : 'reacquiring',
    signal: lost ? 'lost' : 'looking',
    direction: lost ? 'lost' : 'reacquiring',
    proximity: 'unknown',
    confidence: 0,
    distanceText: null,
    distanceMeters: null,
    sentence: lost ? 'Target lost. Point the camera around to reacquire it.' : 'Reacquiring target.',
    speechPhrase: lost ? 'Lost target' : 'Reacquiring',
    assistiveReady: false,
    source: previousState?.source ?? null,
    detection: null,
    ageMs,
    lastDetectionAt: previousState?.lastDetectionAt ?? null,
  };
}

export function proximityFor(meters, area, method = 'unknown', policy = DEFAULT_GUIDANCE_POLICY) {
  if (meters != null && method !== 'unknown') {
    if (meters <= policy.reachMeters) return 'reach';
    if (meters <= 0.9) return 'near';
    if (meters <= 1.6) return 'close';
    if (meters <= 3) return 'medium';
    return 'far';
  }
  if (policy.allowAreaEstimateReach && area >= policy.reachArea) return 'reach';
  if (area >= 0.10) return 'near';
  if (area >= 0.04) return 'close';
  if (area >= 0.01) return 'medium';
  return 'far';
}

function createUncertainState(detection, status, sentence, ageMs) {
  return {
    status,
    signal: status === 'lost' ? 'lost' : 'looking',
    direction: status === 'lost' ? 'lost' : 'looking',
    proximity: 'unknown',
    confidence: Number(detection?.confidence) || 0,
    distanceText: null,
    distanceMeters: null,
    sentence,
    speechPhrase: status === 'lost' ? 'Lost target' : 'Looking',
    assistiveReady: false,
    source: detection?.source ?? null,
    detection: null,
    ageMs,
    area: 0,
    cx: null,
    cy: null,
  };
}

function mergePolicy(policy = {}) {
  return {
    ...DEFAULT_GUIDANCE_POLICY,
    ...policy,
    horizontal: { ...DEFAULT_GUIDANCE_POLICY.horizontal, ...(policy.horizontal || {}) },
    vertical: { ...DEFAULT_GUIDANCE_POLICY.vertical, ...(policy.vertical || {}) },
  };
}

function displayName(detection) {
  return detection?.displayLabel || detection?.label || 'target';
}

function describeDistance(meters, area, method) {
  if (meters != null && method !== 'unknown') {
    if (meters < 1) return `${Math.round(meters * 100)} centimeters`;
    return `${meters.toFixed(1)} meters`;
  }
  if (area > 0.24) return 'very close';
  if (area > 0.14) return 'close';
  if (area > 0.07) return 'medium distance';
  return 'far';
}

function sentenceFor(detection, signal, distanceText, meters) {
  const name = capitalize(displayName(detection));
  const distance = meters != null ? `${meters < 1 ? Math.round(meters * 100) + ' centimeters' : meters.toFixed(1) + ' meters'}` : distanceText;
  switch (signal) {
    case 'reach': return `Reach now. ${name} is right in front of you.`;
    case 'closer': return `${name} is centered, getting closer, ${distance}.`;
    case 'locked': return `${name} is centered, ${distance}.`;
    case 'left': return `Turn left. ${name} ${distance}.`;
    case 'right': return `Turn right. ${name} ${distance}.`;
    case 'up': return `Tilt up. ${name} ${distance}.`;
    case 'down': return `Tilt down. ${name} ${distance}.`;
    default: return `${name} ${distance}.`;
  }
}

function speechFor(signal) {
  return {
    reach: 'Reach', closer: 'Closer', locked: 'Set', left: 'Left', right: 'Right',
    up: 'Up', down: 'Down', lost: 'Lost target', looking: 'Looking',
  }[signal] || 'Looking';
}

function capitalize(value) {
  return value ? value[0].toUpperCase() + value.slice(1) : 'Target';
}
