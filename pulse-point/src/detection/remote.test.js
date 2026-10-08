import { describe, test, expect, beforeEach, afterEach, vi } from 'vitest';

// remote.js reads FLAGS and the server URL once at import, so every test imports a fresh copy.
const flags = vi.hoisted(() => ({ FLAGS: { server: null, serverOff: false } }));
vi.mock('../lib/flags.js', () => flags);

const DEFAULT_URL = 'https://vision.example.test';

async function loadRemote({ server = null, serverOff = false } = {}) {
  flags.FLAGS.server = server;
  flags.FLAGS.serverOff = serverOff;
  vi.resetModules();
  return import('./remote.js');
}

function response(status, body = {}) {
  return { ok: status >= 200 && status < 300, status, json: async () => body };
}

// Stands in for the letterboxed frame / depth crop; only toDataURL is used.
const canvas = { toDataURL: () => 'data:image/jpeg;base64,/9j/4AAQ' };

let fetchMock;

beforeEach(() => {
  vi.useFakeTimers();
  vi.stubEnv('VITE_VISION_URL', DEFAULT_URL);
  fetchMock = vi.fn();
  vi.stubGlobal('fetch', fetchMock);
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
});

describe('configuration', () => {
  test('?server=off disables the server without any network request', async () => {
    const remote = await loadRemote({ serverOff: true });
    expect(remote.isRemoteConfigured()).toBe(false);
    expect(await remote.checkRemote()).toBe(false);
    expect(remote.isRemoteUp()).toBe(false);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  test('uses VITE_VISION_URL by default', async () => {
    fetchMock.mockResolvedValue(response(200));
    const remote = await loadRemote();
    expect(remote.isRemoteConfigured()).toBe(true);
    await remote.checkRemote();
    expect(fetchMock.mock.calls[0][0]).toBe(`${DEFAULT_URL}/health`);
  });

  test('?server=URL overrides the default and drops a trailing slash', async () => {
    fetchMock.mockResolvedValue(response(200));
    const remote = await loadRemote({ server: 'http://localhost:8788/' });
    await remote.checkRemote();
    expect(fetchMock.mock.calls[0][0]).toBe('http://localhost:8788/health');
  });
});

describe('checkRemote', () => {
  test('a healthy server is reported up', async () => {
    fetchMock.mockResolvedValue(response(200));
    const remote = await loadRemote();
    expect(remote.isRemoteUp()).toBe(false);
    expect(await remote.checkRemote()).toBe(true);
    expect(remote.isRemoteUp()).toBe(true);
  });

  test('keeps waiting while a sleeping server answers 503', async () => {
    fetchMock
      .mockResolvedValueOnce(response(503))
      .mockResolvedValueOnce(response(503))
      .mockResolvedValueOnce(response(200));
    const remote = await loadRemote();
    const check = remote.checkRemote();
    await vi.advanceTimersByTimeAsync(4000);
    expect(await check).toBe(true);
    expect(fetchMock).toHaveBeenCalledTimes(3);
  });

  test('keeps waiting through network errors (503s without CORS headers)', async () => {
    fetchMock
      .mockRejectedValueOnce(new TypeError('Failed to fetch'))
      .mockResolvedValueOnce(response(200));
    const remote = await loadRemote();
    const check = remote.checkRemote();
    await vi.advanceTimersByTimeAsync(2000);
    expect(await check).toBe(true);
  });

  test('gives up straight away on any other error status', async () => {
    fetchMock.mockResolvedValue(response(500));
    const remote = await loadRemote();
    expect(await remote.checkRemote()).toBe(false);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(remote.isRemoteUp()).toBe(false);
  });

  test('gives up when the server never wakes before the deadline', async () => {
    fetchMock.mockResolvedValue(response(503));
    const remote = await loadRemote();
    const check = remote.checkRemote(5000);
    await vi.advanceTimersByTimeAsync(6000);
    expect(await check).toBe(false);
    expect(remote.isRemoteUp()).toBe(false);
  });

  test('concurrent checks share one health request', async () => {
    fetchMock.mockResolvedValue(response(200));
    const remote = await loadRemote();
    const [a, b] = await Promise.all([remote.checkRemote(), remote.checkRemote()]);
    expect(a).toBe(true);
    expect(b).toBe(true);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  test('a down server is checked again after 15 s', async () => {
    fetchMock.mockResolvedValueOnce(response(500));
    const remote = await loadRemote();
    await remote.checkRemote();

    expect(remote.isRemoteUp()).toBe(false);
    expect(fetchMock).toHaveBeenCalledTimes(1); // too soon to retry

    fetchMock.mockResolvedValueOnce(response(200));
    await vi.advanceTimersByTimeAsync(15001);
    expect(remote.isRemoteUp()).toBe(false); // starts the retry
    await vi.advanceTimersByTimeAsync(0);
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(remote.isRemoteUp()).toBe(true);
  });
});

describe('remoteDetect', () => {
  async function upRemote() {
    fetchMock.mockResolvedValueOnce(response(200));
    const remote = await loadRemote();
    await remote.checkRemote();
    return remote;
  }

  test('posts the frame and target and returns the boxes', async () => {
    const remote = await upRemote();
    const detections = [{ class: 'keys', score: 0.71, bbox: [10, 20, 30, 40] }];
    fetchMock.mockResolvedValueOnce(response(200, { detections, ms: 42 }));

    expect(await remote.remoteDetect(canvas, 'keys')).toEqual(detections);

    const [url, init] = fetchMock.mock.calls[1];
    expect(url).toBe(`${DEFAULT_URL}/v1/detect`);
    expect(init.method).toBe('POST');
    expect(init.body.get('target')).toBe('keys');
    const file = init.body.get('file');
    expect(file.type).toBe('image/jpeg');
    expect(file.name).toBe('frame.jpg');
  });

  test.each([429, 503])('skips the frame on %i but stays up', async status => {
    const remote = await upRemote();
    fetchMock.mockResolvedValueOnce(response(status));
    expect(await remote.remoteDetect(canvas, 'keys')).toBeNull();
    expect(remote.isRemoteUp()).toBe(true);
  });

  test('an error status throws and marks the server down', async () => {
    const remote = await upRemote();
    fetchMock.mockResolvedValueOnce(response(500));
    await expect(remote.remoteDetect(canvas, 'keys')).rejects.toThrow('/v1/detect 500');
    expect(remote.isRemoteUp()).toBe(false);
  });

  test('a network failure throws and marks the server down', async () => {
    const remote = await upRemote();
    fetchMock.mockRejectedValueOnce(new TypeError('Failed to fetch'));
    await expect(remote.remoteDetect(canvas, 'keys')).rejects.toThrow('Failed to fetch');
    expect(remote.isRemoteUp()).toBe(false);
  });
});

describe('remoteDepth', () => {
  test('sends the box as crop fractions and returns meters', async () => {
    fetchMock.mockResolvedValueOnce(response(200));
    const remote = await loadRemote();
    await remote.checkRemote();
    fetchMock.mockResolvedValueOnce(response(200, { meters: 1.37, ms: 120 }));

    expect(await remote.remoteDepth(canvas, [0.1, 0.25, 1 / 3, 0.5])).toBe(1.37);

    const [url, init] = fetchMock.mock.calls[1];
    expect(url).toBe(`${DEFAULT_URL}/v1/depth`);
    expect(init.body.get('box')).toBe('0.10000,0.25000,0.33333,0.50000');
  });

  test('returns null when the server skips the request', async () => {
    fetchMock.mockResolvedValueOnce(response(200));
    const remote = await loadRemote();
    await remote.checkRemote();
    fetchMock.mockResolvedValueOnce(response(429));
    expect(await remote.remoteDepth(canvas, [0, 0, 1, 1])).toBeNull();
  });
});
