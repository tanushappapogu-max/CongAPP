import { normalizeDetection, selectDetection } from './detection.js';
import { canonicalizeTarget } from './target.js';
import { computeGuidance, lostGuidance } from './guidance.js';

export const GUIDANCE_STATUSES = Object.freeze([
  'idle', 'loading', 'looking', 'locked', 'closer', 'lost', 'reacquiring', 'reach', 'error',
]);

export const EVENTS = Object.freeze({
  START: 'START',
  STOP: 'STOP',
  TARGET_SET: 'TARGET_SET',
  DETECTION_RECEIVED: 'DETECTION_RECEIVED',
  DETECTION_MISSED: 'DETECTION_MISSED',
  MODEL_ERROR: 'MODEL_ERROR',
  CAMERA_ERROR: 'CAMERA_ERROR',
  LOADING: 'LOADING',
  RESET: 'RESET',
});

export function createInitialGuidanceState(target = null) {
  const canonicalTarget = canonicalizeTarget(target);
  return {
    status: 'idle',
    signal: 'looking',
    target: canonicalTarget,
    detection: null,
    source: null,
    proximity: 'unknown',
    confidence: 0,
    distanceText: null,
    distanceMeters: null,
    sentence: canonicalTarget ? `Ready to look for ${canonicalTarget}.` : 'Enter a target to begin.',
    speechPhrase: 'Ready',
    assistiveReady: false,
    lastDetectionAt: null,
    previousArea: 0,
    missedFrames: 0,
    error: null,
    sessionId: 0,
  };
}

export function guidanceReducer(state, event, options = {}) {
  const type = event?.type;
  switch (type) {
    case EVENTS.START:
      return {
        ...createInitialGuidanceState(event.target ?? state.target),
        status: 'loading',
        sentence: 'Loading detector.',
        speechPhrase: 'Loading',
        sessionId: state.sessionId + 1,
      };
    case EVENTS.LOADING:
      return { ...state, status: 'loading', error: null };
    case EVENTS.TARGET_SET: {
      const target = canonicalizeTarget(event.target);
      return {
        ...state,
        target,
        status: state.status === 'idle' ? 'idle' : 'looking',
        detection: null,
        source: null,
        proximity: 'unknown',
        confidence: 0,
        distanceText: null,
        distanceMeters: null,
        assistiveReady: false,
        previousArea: 0,
        missedFrames: 0,
        error: null,
        sentence: target ? `Looking for ${target}.` : 'Enter a target to begin.',
      };
    }
    case EVENTS.DETECTION_RECEIVED: {
      const candidates = event.detections ?? event.candidates;
      const detection = candidates
        ? selectDetection(candidates, state.target, { nowMs: event.nowMs, assistiveReady: event.assistiveReady })
        : normalizeDetection(event.detection, { nowMs: event.nowMs, assistiveReady: event.assistiveReady });
      if (!detection) return guidanceReducer(state, { type: EVENTS.DETECTION_MISSED, nowMs: event.nowMs }, options);
      const guidance = computeGuidance(detection, {
        previousArea: state.previousArea,
        nowMs: event.nowMs,
        policy: options.policy,
      });
      return {
        ...state,
        ...guidance,
        status: guidance.status,
        detection,
        source: detection.source,
        lastDetectionAt: event.nowMs ?? detection.timestampMs,
        previousArea: guidance.area ?? state.previousArea,
        missedFrames: 0,
        error: null,
      };
    }
    case EVENTS.DETECTION_MISSED: {
      const guidance = lostGuidance(state, { nowMs: event.nowMs, policy: options.policy });
      return {
        ...state,
        ...guidance,
        missedFrames: state.missedFrames + 1,
        detection: null,
        assistiveReady: false,
      };
    }
    case EVENTS.MODEL_ERROR:
    case EVENTS.CAMERA_ERROR:
      return {
        ...state,
        status: 'error',
        signal: 'lost',
        assistiveReady: false,
        error: event.error || (type === EVENTS.MODEL_ERROR ? 'Detector failed.' : 'Camera failed.'),
        sentence: event.error || (type === EVENTS.MODEL_ERROR ? 'Detector failed.' : 'Camera failed.'),
        speechPhrase: 'Error',
      };
    case EVENTS.STOP:
    case EVENTS.RESET:
      return createInitialGuidanceState(type === EVENTS.RESET ? null : state.target);
    default:
      return state;
  }
}
