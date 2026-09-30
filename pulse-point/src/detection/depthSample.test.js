import { describe, test, expect } from 'vitest';
import { sampleBoxDepth, currentDepthMeters, isRoughlyCentered, centerSquare, boxInCrop } from './depthSample.js';

function depthMap(dw, dh, fill, box = null, boxValue = 0) {
  const d = new Float32Array(dw * dh).fill(fill);
  if (box) {
    const [x0, y0, x1, y1] = box;
    for (let r = y0; r < y1; r++) for (let c = x0; c < x1; c++) d[r * dw + c] = boxValue;
  }
  return d;
}

describe('sampleBoxDepth', () => {
  test('reads the object, not the wall behind it', () => {
    // 100x100 map, wall at 4 m, object at 1.2 m filling the middle of the box
    const d = depthMap(100, 100, 4, [30, 30, 70, 70], 1.2);
    expect(sampleBoxDepth(d, 100, 100, [0.25, 0.25, 0.5, 0.5])).toBeCloseTo(1.2);
  });

  test('prefers nearer pixels when the box is part background', () => {
    const d = depthMap(100, 100, 5, [40, 0, 100, 100], 2);
    expect(sampleBoxDepth(d, 100, 100, [0, 0, 1, 1])).toBe(2);
  });

  test('returns null for an empty or off-frame box', () => {
    const d = depthMap(10, 10, 3);
    expect(sampleBoxDepth(d, 10, 10, [0.5, 0.5, 0, 0])).toBeNull();
    expect(sampleBoxDepth(d, 10, 10, [2, 2, 0.5, 0.5])).toBeNull();
  });

  test('ignores invalid depth values', () => {
    const d = depthMap(10, 10, NaN);
    expect(sampleBoxDepth(d, 10, 10, [0, 0, 1, 1])).toBeNull();
  });
});

describe('currentDepthMeters', () => {
  const reading = { meters: 2, boxWidth: 100, at: 1000, target: 'keys' };

  test('same box size keeps the measured distance', () => {
    expect(currentDepthMeters(reading, 'keys', 100, 2000)).toBe(2);
  });

  test('box twice as wide means half the distance', () => {
    expect(currentDepthMeters(reading, 'keys', 200, 2000)).toBe(1);
  });

  test('ignores readings for a different target or that are too old', () => {
    expect(currentDepthMeters(reading, 'wallet', 100, 2000)).toBeNull();
    expect(currentDepthMeters(reading, 'keys', 100, 1000 + 15001)).toBeNull();
    expect(currentDepthMeters(null, 'keys', 100, 2000)).toBeNull();
  });

  test('clamps to the model range', () => {
    expect(currentDepthMeters(reading, 'keys', 1, 2000)).toBe(20);
  });
});

describe('isRoughlyCentered', () => {
  const frame = { width: 1000, height: 1000 };

  test('a box in the middle counts', () => {
    expect(isRoughlyCentered([400, 400, 200, 200], frame)).toBe(true);
  });

  test('a box off to the side does not', () => {
    expect(isRoughlyCentered([0, 400, 150, 200], frame)).toBe(false);
    expect(isRoughlyCentered([400, 850, 100, 100], frame)).toBe(false);
  });
});

describe('centerSquare / boxInCrop', () => {
  test('landscape frame: square in the middle, full height', () => {
    expect(centerSquare(1920, 1080)).toEqual({ left: 420, top: 0, side: 1080 });
  });

  test('portrait frame: square in the middle, full width', () => {
    expect(centerSquare(1080, 1920)).toEqual({ left: 0, top: 420, side: 1080 });
  });

  test('a centered box maps into crop-relative coordinates', () => {
    const crop = centerSquare(1920, 1080);
    expect(boxInCrop([900, 440, 120, 200], crop)).toEqual([480 / 1080, 440 / 1080, 120 / 1080, 200 / 1080]);
  });

  test('a box whose center is outside the crop is skipped', () => {
    expect(boxInCrop([50, 400, 100, 100], centerSquare(1920, 1080))).toBeNull();
  });
});
