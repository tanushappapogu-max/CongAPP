import { describe, test, expect } from 'vitest';
import { isWebKit } from './flags.js';

const nav = (userAgent, extra = {}) => ({ userAgent, platform: '', maxTouchPoints: 0, ...extra });

describe('isWebKit', () => {
  test('iPhone Safari and iPhone Chrome both count (Chrome on iOS runs WebKit)', () => {
    expect(isWebKit(nav('Mozilla/5.0 (iPhone; CPU iPhone OS 18_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.5 Mobile/15E148 Safari/604.1'))).toBe(true);
    expect(isWebKit(nav('Mozilla/5.0 (iPhone; CPU iPhone OS 18_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/140.0 Mobile/15E148 Safari/604.1'))).toBe(true);
  });

  test('iPad reporting a Mac user agent counts', () => {
    expect(isWebKit(nav('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.5 Safari/605.1.15', { platform: 'MacIntel', maxTouchPoints: 5 }))).toBe(true);
  });

  test('desktop Safari counts; desktop and Android Chrome do not', () => {
    expect(isWebKit(nav('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.5 Safari/605.1.15', { platform: 'MacIntel' }))).toBe(true);
    expect(isWebKit(nav('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0 Safari/537.36', { platform: 'MacIntel' }))).toBe(false);
    expect(isWebKit(nav('Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0 Mobile Safari/537.36'))).toBe(false);
  });
});
