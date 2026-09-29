import { describe, test, expect } from 'vitest';
import { computeGuidance } from './compute.js';

const FRAME = { width: 640, height: 480 };

// Helper: build a match object with an unknown class so distance.js returns null
// (avoids meter-based veryClose logic; only area matters for proximity).
function match(bbox) {
  return { bbox, class: '__unknown__' };
}

describe('computeGuidance — signal selection', () => {
  test('reach: centered box with area > 0.20', () => {
    // cx=0.5, cy=0.5, area=(320*240)/307200=0.25 — large enough to trigger reach
    const result = computeGuidance(match([160, 120, 320, 240]), FRAME, 0);
    expect(result.signal).toBe('reach');
    expect(result.status).toBe('reach');
  });

  test('closer: centered box that grew > 5 % since last frame', () => {
    // area=(128*80)/307200≈0.0333; prevArea=0.031 → 0.0333 > 0.031*1.05=0.0326
    const result = computeGuidance(match([208, 154, 128, 80]), FRAME, 0.031);
    expect(result.signal).toBe('closer');
  });

  test('locked: centered box that did not grow', () => {
    // prevArea=0.034 → area(0.0333) < 0.034*1.05=0.0357 — not growing
    const result = computeGuidance(match([208, 154, 128, 80]), FRAME, 0.034);
    expect(result.signal).toBe('locked');
  });

  test('left: object on the left side', () => {
    // cx=(0+32)/640=0.05 < H_LEFT=0.40; cy=(200+40)/480=0.5
    const result = computeGuidance(match([0, 200, 64, 80]), FRAME, 0);
    expect(result.signal).toBe('left');
    expect(result.direction).toMatch(/left/i);
  });

  test('right: object on the right side', () => {
    // cx=(544+32)/640=0.9 > H_RIGHT=0.60; cy=(200+40)/480=0.5
    const result = computeGuidance(match([544, 200, 64, 80]), FRAME, 0);
    expect(result.signal).toBe('right');
    expect(result.direction).toMatch(/right/i);
  });

  test('up: object above, horizontally centered', () => {
    // cx=(256+64)/640=0.5; cy=(0+24)/480=0.05 < V_TOP=0.38
    // horizErr=0 (inH), vertErr=|0.05-0.5|=0.45 — vertical wins
    const result = computeGuidance(match([256, 0, 128, 48]), FRAME, 0);
    expect(result.signal).toBe('up');
  });

  test('down: object below, horizontally centered', () => {
    // cx=0.5; cy=(384+24)/480=0.85 > V_BOTTOM=0.62
    const result = computeGuidance(match([256, 384, 128, 48]), FRAME, 0);
    expect(result.signal).toBe('down');
  });

  test('dominant horizontal axis wins when horizErr > vertErr', () => {
    // cx=(32+32)/640=0.1 (left), cy=(290+24)/480=0.654 (slightly below)
    // horizErr=0.4, vertErr=0.154 → horizontal wins → left
    const result = computeGuidance(match([32, 290, 64, 48]), FRAME, 0);
    expect(result.signal).toBe('left');
  });
});

describe('computeGuidance — return shape', () => {
  test('always returns required fields', () => {
    const result = computeGuidance(match([208, 154, 128, 80]), FRAME, 0);
    expect(result).toHaveProperty('signal');
    expect(result).toHaveProperty('status');
    expect(result).toHaveProperty('direction');
    expect(result).toHaveProperty('area');
    expect(result).toHaveProperty('cx');
    expect(result).toHaveProperty('cy');
    expect(result).toHaveProperty('sentence');
  });

  test('area is a fraction in (0, 1)', () => {
    const result = computeGuidance(match([160, 120, 320, 240]), FRAME, 0);
    expect(result.area).toBeGreaterThan(0);
    expect(result.area).toBeLessThan(1);
  });

  test('sentence mentions the class name', () => {
    const m = { bbox: [208, 154, 128, 80], class: 'bottle' };
    const result = computeGuidance(m, FRAME, 0);
    expect(result.sentence.toLowerCase()).toContain('bottle');
  });

  test('real COCO class gets a distance estimate', () => {
    // 'laptop' has a reference width — should get a non-null distanceMeters
    const m = { bbox: [160, 120, 200, 150], class: 'laptop' };
    const result = computeGuidance(m, FRAME, 0);
    expect(result.distanceMeters).not.toBeNull();
    expect(result.distanceMeters).toBeGreaterThan(0);
  });
});

describe('computeGuidance — depth readings', () => {
  test('depth reading takes priority over the width estimate', () => {
    const box = [300, 220, 40, 40];
    const byWidth = computeGuidance({ bbox: box, class: 'laptop' }, FRAME, 0);
    const byDepth = computeGuidance({ bbox: box, class: 'laptop', depthMeters: 0.4 }, FRAME, 0);
    expect(byWidth.distanceMeters).toBeGreaterThan(2);
    expect(byDepth.distanceMeters).toBe(0.4);
    expect(byDepth.signal).toBe('reach');
  });

  test('without depth, a small centered box is not a reach', () => {
    const result = computeGuidance({ bbox: [300, 220, 40, 40], class: 'laptop', depthMeters: null }, FRAME, 0);
    expect(result.signal).not.toBe('reach');
  });
});

describe('computeGuidance — frame orientation', () => {
  test('portrait and landscape give the same distance for the same object size', () => {
    const landscape = computeGuidance({ bbox: [900, 500, 120, 80], class: 'laptop' }, { width: 1920, height: 1080 }, 0);
    const portrait = computeGuidance({ bbox: [480, 900, 120, 80], class: 'laptop' }, { width: 1080, height: 1920 }, 0);
    expect(portrait.distanceMeters).toBeCloseTo(landscape.distanceMeters, 5);
  });
});

describe('computeGuidance — lens field of view', () => {
  test('the same box on a wider lens means the object is closer', () => {
    const frame = { width: 1920, height: 1080 };
    const box = { bbox: [900, 500, 120, 80], class: 'laptop' };
    const normal = computeGuidance(box, frame, 0);
    const wide = computeGuidance(box, { ...frame, fovDeg: 102 }, 0);
    expect(wide.distanceMeters).toBeLessThan(normal.distanceMeters * 0.6);
  });
});
