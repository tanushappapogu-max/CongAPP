import { describe, test, expect, vi } from 'vitest';
import { pickMainBackCamera, setNormalZoom } from './camera.js';

const cam = (label, deviceId = label) => ({ kind: 'videoinput', label, deviceId });

describe('pickMainBackCamera', () => {
  test('iPhone: picks the 1× "Back Camera", not ultra-wide or multi-lens virtual cameras', () => {
    const devices = [
      cam('Front Camera'),
      cam('Back Triple Camera'),
      cam('Back Dual Wide Camera'),
      cam('Back Ultra Wide Camera'),
      cam('Back Camera'),
      cam('Back Telephoto Camera'),
    ];
    expect(pickMainBackCamera(devices).label).toBe('Back Camera');
  });

  test('Android: picks the first plain back camera', () => {
    const devices = [cam('camera2 1, facing front'), cam('camera2 0, facing back'), cam('camera2 2, facing back')];
    expect(pickMainBackCamera(devices).label).toBe('camera2 0, facing back');
  });

  test('returns null when labels are missing or only non-main lenses exist', () => {
    expect(pickMainBackCamera([{ kind: 'videoinput', label: '', deviceId: 'x' }])).toBeNull();
    expect(pickMainBackCamera([cam('Back Ultra Wide Camera')])).toBeNull();
  });
});

describe('setNormalZoom', () => {
  const streamWith = (zoom) => {
    const track = { getCapabilities: () => (zoom ? { zoom } : {}), applyConstraints: vi.fn(async () => {}) };
    return { track, stream: { getVideoTracks: () => [track] } };
  };

  test('multi-lens camera that can go to 0.5× is set to 1×', async () => {
    const { track, stream } = streamWith({ min: 0.5, max: 15 });
    await setNormalZoom(stream);
    expect(track.applyConstraints).toHaveBeenCalledWith({ advanced: [{ zoom: 1 }] });
  });

  test('clamps into the supported range', async () => {
    const { track, stream } = streamWith({ min: 2, max: 8 });
    await setNormalZoom(stream);
    expect(track.applyConstraints).toHaveBeenCalledWith({ advanced: [{ zoom: 2 }] });
  });

  test('does nothing without zoom support', async () => {
    const { track, stream } = streamWith(null);
    await setNormalZoom(stream);
    expect(track.applyConstraints).not.toHaveBeenCalled();
  });
});
