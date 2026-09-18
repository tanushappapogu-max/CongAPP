import { describe, expect, test } from 'vitest';
import {
  EVENTS,
  DetectionTracker,
  computeGuidance,
  createTargetRequest,
  guidanceReducer,
  normalizeDetection,
  normalizeDetections,
  selectDetection,
  createInitialGuidanceState,
} from '../../../packages/pulsepoint-core/src/index.js';

const box = { x: 0.4, y: 0.4, width: 0.2, height: 0.2 };

function detection(overrides = {}) {
  return normalizeDetection({
    label: 'bottle',
    displayLabel: 'bottle',
    bbox: box,
    confidence: 0.9,
    timestampMs: 1000,
    source: 'mobile-native',
    assistiveReady: true,
    distance: { meters: 1.2, method: 'depth', uncertaintyMeters: 0.2 },
    ...overrides,
  });
}

describe('pulsepoint core target and detection contracts', () => {
  test('normalizes voice aliases into one target request', () => {
    expect(createTargetRequest(' Find my PHONE! ', 'voice')).toMatchObject({
      rawText: 'Find my PHONE!',
      normalizedText: 'find my phone',
      source: 'voice',
      canonicalLabel: null,
    });
    expect(createTargetRequest('phone', 'typed').canonicalLabel).toBe('cell phone');
  });

  test('clamps boxes and preserves distance provenance', () => {
    const value = normalizeDetection({
      label: 'cup',
      bbox: { x: -0.2, y: 0.2, width: 2, height: 0.4 },
      confidence: 1.4,
      source: 'server',
      distance: { meters: 2, method: 'area-estimate' },
    });
    expect(value.bbox).toEqual({ x: 0, y: 0.2, width: 1, height: 0.4 });
    expect(value.confidence).toBe(1);
    expect(value.distance.method).toBe('area-estimate');
    expect(value.assistiveReady).toBe(false);
  });

  test('normalizes and selects the strongest matching candidate', () => {
    const candidates = normalizeDetections([
      { label: 'cup', bbox: [0.1, 0.1, 0.2, 0.2], confidence: 0.4, timestampMs: 10 },
      { label: 'phone', bbox: { x: 0.2, y: 0.2, width: 0.2, height: 0.2 }, confidence: 0.8, timestampMs: 10 },
      { label: 'cell phone', bbox: { x: 0.3, y: 0.3, width: 0.2, height: 0.2 }, confidence: 0.7, timestampMs: 10 },
    ]);
    expect(selectDetection(candidates, 'PHONE').label).toBe('cell phone');
    expect(candidates).toHaveLength(3);
  });

  test('simulation can produce a demo state but never assistive-ready', () => {
    const result = computeGuidance(normalizeDetection({
      label: 'chair', bbox: { x: 0.25, y: 0.25, width: 0.5, height: 0.5 },
      confidence: 0.95, source: 'simulated', assistiveReady: true, timestampMs: 100,
    }), { nowMs: 100 });
    expect(result.status).toBe('reach');
    expect(result.assistiveReady).toBe(false);
    expect(result.sentence).toMatch(/experimental/i);
  });
});

describe('pulsepoint core tracking and guidance', () => {
  test('tracker predicts movement and decays confidence for stale frames', () => {
    const tracker = new DetectionTracker();
    tracker.update(detection({ bbox: { x: 0.2, y: 0.4, width: 0.2, height: 0.2 } }), 1000);
    tracker.update(detection({ bbox: { x: 0.25, y: 0.4, width: 0.2, height: 0.2 }, timestampMs: 1050 }), 1050);
    const predicted = tracker.predict(1100);
    expect(predicted.bbox.x).toBeGreaterThan(0.25);
    expect(predicted.confidence).toBeLessThan(0.9);
    expect(tracker.predict(1701)).toBeNull();
  });

  test('direction and reach are deterministic', () => {
    expect(computeGuidance(detection({ bbox: { x: 0.02, y: 0.4, width: 0.12, height: 0.12 } }), { nowMs: 1000 }).signal).toBe('left');
    expect(computeGuidance(detection({ bbox: { x: 0.4, y: 0.4, width: 0.2, height: 0.2 }, distance: { meters: 0.4, method: 'depth' } }), { nowMs: 1000 }).signal).toBe('reach');
    expect(computeGuidance(detection({ bbox: { x: 0.4, y: 0.4, width: 0.5, height: 0.5 }, distance: { meters: null, method: 'unknown' } }), { nowMs: 1000, policy: { allowAreaEstimateReach: false } }).signal).not.toBe('reach');
  });
});

describe('pulsepoint core state machine', () => {
  test('covers start, target, detection, miss, and reset', () => {
    let state = createInitialGuidanceState();
    state = guidanceReducer(state, { type: EVENTS.START, target: 'bottle' });
    expect(state.status).toBe('loading');
    state = guidanceReducer(state, { type: EVENTS.DETECTION_RECEIVED, detection: detection(), nowMs: 1000 }, { policy: { minConfidence: 0.35 } });
    expect(state.status).toBe('locked');
    state = guidanceReducer(state, { type: EVENTS.DETECTION_MISSED, nowMs: 1100 }, { policy: { lostAfterMs: 1400 } });
    expect(state.status).toBe('reacquiring');
    state = guidanceReducer(state, { type: EVENTS.DETECTION_MISSED, nowMs: 2600 }, { policy: { lostAfterMs: 1400 } });
    expect(state.status).toBe('lost');
    state = guidanceReducer(state, { type: EVENTS.RESET });
    expect(state.status).toBe('idle');
  });

  test('low confidence never locks', () => {
    const state = guidanceReducer(
      createInitialGuidanceState('bottle'),
      { type: EVENTS.DETECTION_RECEIVED, detection: detection({ confidence: 0.1 }), nowMs: 1000 },
    );
    expect(state.status).toBe('looking');
    expect(state.assistiveReady).toBe(false);
  });
});
