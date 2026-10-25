import { EVENTS } from '../stateMachine.js';

const nativeSource = {
  source: 'mobile-native',
  assistiveReady: true,
};

const centeredBox = { x: 0.4, y: 0.4, width: 0.2, height: 0.2 };

export const GUIDANCE_REPLAY_FIXTURES = Object.freeze([
  {
    name: 'found-locked',
    target: 'bottle',
    events: [
      { type: EVENTS.START, target: 'bottle' },
      {
        type: EVENTS.DETECTION_RECEIVED,
        nowMs: 1000,
        detection: {
          label: 'bottle', displayLabel: 'Bottle', bbox: centeredBox,
          confidence: 0.92, timestampMs: 1000,
          distance: { meters: 1.2, method: 'depth', uncertaintyMeters: 0.2 },
          ...nativeSource,
        },
      },
    ],
    expected: {
      final: { status: 'locked', signal: 'locked', target: 'bottle', assistiveReady: true },
    },
  },
  {
    name: 'lost',
    target: 'bottle',
    events: [
      { type: EVENTS.START, target: 'bottle' },
      {
        type: EVENTS.DETECTION_RECEIVED,
        nowMs: 1000,
        detection: {
          label: 'bottle', bbox: centeredBox, confidence: 0.9,
          timestampMs: 1000, distance: { meters: 1.2, method: 'depth' }, ...nativeSource,
        },
      },
      { type: EVENTS.DETECTION_MISSED, nowMs: 1300 },
      { type: EVENTS.DETECTION_MISSED, nowMs: 2501 },
    ],
    expected: {
      final: { status: 'lost', signal: 'lost', detection: null, assistiveReady: false },
      checkpoints: [{ after: 3, state: { status: 'reacquiring', signal: 'looking' } }],
    },
  },
  {
    name: 'reacquired',
    target: 'bottle',
    events: [
      { type: EVENTS.START, target: 'bottle' },
      {
        type: EVENTS.DETECTION_RECEIVED,
        nowMs: 1000,
        detection: {
          label: 'bottle', bbox: centeredBox, confidence: 0.9,
          timestampMs: 1000, distance: { meters: 1.2, method: 'depth' }, ...nativeSource,
        },
      },
      { type: EVENTS.DETECTION_MISSED, nowMs: 2501 },
      {
        type: EVENTS.DETECTION_RECEIVED,
        nowMs: 2700,
        detection: {
          label: 'bottle', bbox: centeredBox, confidence: 0.88,
          timestampMs: 2700, distance: { meters: 1.1, method: 'depth' }, ...nativeSource,
        },
      },
    ],
    expected: {
      final: { status: 'locked', signal: 'locked', detection: { label: 'bottle' }, assistiveReady: true },
      checkpoints: [{ after: 3, state: { status: 'lost', signal: 'lost' } }],
    },
  },
  {
    name: 'moving-target',
    target: 'bottle',
    events: [
      { type: EVENTS.START, target: 'bottle' },
      {
        type: EVENTS.DETECTION_RECEIVED,
        nowMs: 1000,
        detection: {
          label: 'bottle', bbox: { x: 0.1, y: 0.42, width: 0.12, height: 0.12 },
          confidence: 0.9, timestampMs: 1000, distance: { meters: 2, method: 'depth' }, ...nativeSource,
        },
      },
      {
        type: EVENTS.DETECTION_RECEIVED,
        nowMs: 1100,
        detection: {
          label: 'bottle', bbox: { x: 0.42, y: 0.42, width: 0.12, height: 0.12 },
          confidence: 0.9, timestampMs: 1100, distance: { meters: 2, method: 'depth' }, ...nativeSource,
        },
      },
    ],
    expected: {
      final: { status: 'locked', signal: 'locked', detection: { bbox: { x: 0.42 } } },
      checkpoints: [{ after: 2, state: { status: 'left', signal: 'left' } }],
    },
  },
  {
    name: 'multiple-candidates',
    target: 'cell phone',
    events: [
      { type: EVENTS.START, target: 'cell phone' },
      {
        type: EVENTS.DETECTION_RECEIVED,
        nowMs: 1000,
        detections: [
          { label: 'laptop', bbox: { x: 0.4, y: 0.4, width: 0.3, height: 0.3 }, confidence: 0.99, timestampMs: 1000 },
          { label: 'phone', bbox: { x: 0.05, y: 0.4, width: 0.1, height: 0.1 }, confidence: 0.45, timestampMs: 1000, ...nativeSource },
          { label: 'cell phone', bbox: centeredBox, confidence: 0.9, timestampMs: 1000, ...nativeSource },
        ],
      },
    ],
    expected: {
      final: { status: 'locked', signal: 'locked', target: 'cell phone', detection: { label: 'cell phone', confidence: 0.9 } },
    },
  },
  {
    name: 'low-light-safe',
    target: 'bottle',
    events: [
      { type: EVENTS.START, target: 'bottle' },
      {
        type: EVENTS.DETECTION_RECEIVED,
        nowMs: 1000,
        detection: {
          label: 'bottle', displayLabel: 'Bottle in low light', bbox: centeredBox,
          confidence: 0.34, timestampMs: 1000, lighting: 'low-light', ...nativeSource,
        },
      },
    ],
    expected: {
      final: { status: 'looking', signal: 'looking', assistiveReady: false },
    },
  },
  {
    name: 'low-confidence-safe',
    target: 'bottle',
    events: [
      { type: EVENTS.START, target: 'bottle' },
      {
        type: EVENTS.DETECTION_RECEIVED,
        nowMs: 1000,
        detection: {
          label: 'bottle', bbox: centeredBox, confidence: 0.1,
          timestampMs: 1000, ...nativeSource,
        },
      },
    ],
    expected: {
      final: { status: 'looking', signal: 'looking', assistiveReady: false },
    },
  },
  {
    name: 'reach-policy',
    target: 'bottle',
    options: { policy: { allowAreaEstimateReach: false } },
    events: [
      { type: EVENTS.START, target: 'bottle' },
      {
        type: EVENTS.DETECTION_RECEIVED,
        nowMs: 1000,
        detection: {
          label: 'bottle', bbox: { x: 0.25, y: 0.25, width: 0.5, height: 0.5 }, confidence: 0.95,
          timestampMs: 1000, distance: { meters: 0.4, method: 'depth' }, ...nativeSource,
        },
      },
      {
        type: EVENTS.DETECTION_RECEIVED,
        nowMs: 1100,
        detection: {
          label: 'bottle', bbox: { x: 0.25, y: 0.25, width: 0.5, height: 0.5 },
          confidence: 0.95, timestampMs: 1100,
          distance: { meters: null, method: 'area-estimate' }, source: 'server', assistiveReady: false,
        },
      },
    ],
    expected: {
      final: { status: 'locked', signal: 'locked', proximity: 'near', assistiveReady: false },
      checkpoints: [{ after: 2, state: { status: 'reach', signal: 'reach', proximity: 'reach', assistiveReady: true } }],
    },
  },
]);
