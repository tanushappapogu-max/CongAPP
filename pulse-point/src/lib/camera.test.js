import { describe, test, expect, vi } from 'vitest';
import { pickMainBackCamera, pickWideBackCamera, setWidestZoom, lensFovDeg } from './camera.js';

const cam = (label, deviceId = label) => ({ kind: 'videoinput', label, deviceId });

const IPHONE = [
  cam('Front Camera'),
  cam('Back Triple Camera'),
  cam('Back Dual Wide Camera'),
  cam('Back Ultra Wide Camera'),
  cam('Back Camera'),
  cam('Back Telephoto Camera'),
];

describe('pickWideBackCamera', () => {
  test('iPhone: picks the dedicated ultra-wide lens, not a multi-lens virtual camera', () => {
    expect(pickWideBackCamera(IPHONE).label).toBe('Back Ultra Wide Camera');
  });

  test('falls back to the main back camera without an ultra-wide', () => {
    expect(pickWideBackCamera([cam('Front Camera'), cam('Back Camera')]).label).toBe('Back Camera');
    const android = [cam('camera2 1, facing front'), cam('camera2 0, facing back')];
    expect(pickWideBackCamera(android).label).toBe('camera2 0, facing back');
  });

  test('returns null when labels are missing', () => {
    expect(pickWideBackCamera([{ kind: 'videoinput', label: '', deviceId: 'x' }])).toBeNull();
  });
});

describe('pickMainBackCamera', () => {
  test('iPhone: picks the 1× "Back Camera"', () => {
    expect(pickMainBackCamera(IPHONE).label).toBe('Back Camera');
  });
});

describe('setWidestZoom', () => {
  const streamWith = (zoom) => {
    const track = { getCapabilities: () => (zoom ? { zoom } : {}), applyConstraints: vi.fn(async () => {}) };
    return { track, stream: { getVideoTracks: () => [track] } };
  };

  test('zooms all the way out', async () => {
    const { track, stream } = streamWith({ min: 0.6, max: 10 });
    await setWidestZoom(stream);
    expect(track.applyConstraints).toHaveBeenCalledWith({ advanced: [{ zoom: 0.6 }] });
  });

  test('does nothing without zoom support', async () => {
    const { track, stream } = streamWith(null);
    await setWidestZoom(stream);
    expect(track.applyConstraints).not.toHaveBeenCalled();
  });
});

describe('lensFovDeg', () => {
  const stream = (label, zoom) => ({ getVideoTracks: () => [{ label, getSettings: () => (zoom ? { zoom } : {}) }] });

  test('main camera at 1× is the normal ~64°', () => {
    expect(lensFovDeg(stream('Back Camera'))).toBeCloseTo(64, 5);
  });

  test('ultra-wide lens is ~103°', () => {
    expect(lensFovDeg(stream('Back Ultra Wide Camera', 1))).toBeCloseTo(102.7, 1);
  });

  test('zooming out on a main camera widens the view', () => {
    const fov = lensFovDeg(stream('camera2 0, facing back', 0.6));
    expect(fov).toBeGreaterThan(90);
    expect(fov).toBeLessThan(100);
  });

  test('defaults to the main camera without a track', () => {
    expect(lensFovDeg(null)).toBe(64);
  });
});
