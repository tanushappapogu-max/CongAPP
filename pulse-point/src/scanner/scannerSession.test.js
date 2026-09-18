import { describe, expect, test, vi } from 'vitest';
import { createScannerSession, SCANNER_EVENTS } from './scannerSession.js';

function scheduler() {
  let now = 0; let id = 0; const timers = new Map();
  return {
    now: () => now,
    setTimeout(fn) { const handle = ++id; timers.set(handle, fn); return handle; },
    clearTimeout(handle) { timers.delete(handle); },
    async tick(ms = 0) {
      now += ms;
      const pending = [...timers.values()]; timers.clear(); pending.forEach((fn) => fn());
      await Promise.resolve(); await Promise.resolve();
    },
  };
}

function deferred() { let resolve; const promise = new Promise((res) => { resolve = res; }); return { promise, resolve }; }
const detection = () => ({ bbox: [0, 0, 1, 1] });

function makeSession(options = {}) {
  const clock = options.scheduler || scheduler(); const events = [];
  const session = createScannerSession({
    initializeCamera: vi.fn(async () => ({ id: 'camera' })),
    initializeModel: vi.fn(async () => ({ id: 'model' })),
    detect: vi.fn(async () => detection()), scheduler: clock, intervalMs: 10,
    onEvent: (event) => events.push(event), ...options,
  });
  return { clock, events, session };
}

describe('scanner session', () => {
  test('emits start, loads camera/model, then detects', async () => {
    const { clock, events, session } = makeSession();
    await session.start('cup');
    expect(events.map((event) => event.type)).toEqual([SCANNER_EVENTS.START]);
    await clock.tick(10);
    expect(events.at(-1)).toMatchObject({ type: SCANNER_EVENTS.DETECTION_RECEIVED, target: 'cup' });
  });

  test('target change invalidates the old generation and detects the new target', async () => {
    const oldResult = deferred(); const detect = vi.fn(() => oldResult.promise);
    const { clock, events, session } = makeSession({ detect });
    await session.start('cup'); await clock.tick(10); session.setTarget('bottle');
    oldResult.resolve(detection()); await Promise.resolve(); await clock.tick(10);
    expect(detect).toHaveBeenCalledWith(expect.objectContaining({ target: 'bottle' }));
    expect(events.filter((event) => event.type === SCANNER_EVENTS.DETECTION_RECEIVED)
      .map((event) => event.target)).toEqual(['bottle']);
  });

  test('stop during loading aborts and suppresses late initialization', async () => {
    const cameraResult = deferred(); const stopCamera = vi.fn();
    const { events, session } = makeSession({ initializeCamera: vi.fn(() => cameraResult.promise), stopCamera });
    const pending = session.start('cup'); session.stop(); cameraResult.resolve({ id: 'late-camera' }); await pending;
    expect(stopCamera).toHaveBeenCalledWith({ id: 'late-camera' });
    expect(events.map((event) => event.type)).toEqual([SCANNER_EVENTS.START, SCANNER_EVENTS.STOP]);
  });

  test('retarget during model loading stops the stale camera resource', async () => {
    const modelResult = deferred(); const stopCamera = vi.fn(); let cameraId = 0;
    const initializeCamera = vi.fn(async () => ({ id: `camera-${++cameraId}` }));
    const initializeModel = vi.fn(() => initializeModel.mock.calls.length === 1
      ? modelResult.promise
      : Promise.resolve({ id: 'new-model' }));
    const { session } = makeSession({ initializeCamera, initializeModel, stopCamera });

    const pending = session.start('cup');
    await Promise.resolve(); await Promise.resolve();
    session.setTarget('bottle');
    modelResult.resolve({ id: 'old-model' });
    await pending;
    await Promise.resolve(); await Promise.resolve();

    expect(stopCamera).toHaveBeenCalledWith({ id: 'camera-1' });
    expect(initializeModel).toHaveBeenCalledWith(expect.objectContaining({ target: 'bottle' }));
  });

  test('reports missed, lost, and reacquired detection flow', async () => {
    let call = 0;
    const { clock, events, session } = makeSession({
      lostAfterMisses: 2, detect: vi.fn(async () => (++call < 3 ? null : detection())),
    });
    await session.start('cup'); await clock.tick(10); await clock.tick(10); await clock.tick(10);
    expect(events.filter((event) => event.type === SCANNER_EVENTS.DETECTION_MISSED).map((event) => event.status))
      .toEqual(['reacquiring', 'lost']);
    expect(events.at(-1)).toMatchObject({ type: SCANNER_EVENTS.DETECTION_RECEIVED, reacquired: true });
  });

  test('passes reach guidance through the received event', async () => {
    const { clock, events, session } = makeSession({ computeGuidance: () => ({ signal: 'reach', status: 'reach' }) });
    await session.start('cup'); await clock.tick(10);
    expect(events.at(-1)).toMatchObject({ type: SCANNER_EVENTS.DETECTION_RECEIVED, guidance: { signal: 'reach' } });
  });

  test('never overlaps detector calls and ignores stale results after stop', async () => {
    const result = deferred(); const detect = vi.fn(() => result.promise);
    const { clock, events, session } = makeSession({ detect });
    await session.start('cup'); await clock.tick(10); await clock.tick(10);
    expect(detect).toHaveBeenCalledTimes(1); session.stop(); result.resolve(detection()); await Promise.resolve();
    expect(events.filter((event) => event.type === SCANNER_EVENTS.DETECTION_RECEIVED)).toHaveLength(0);
  });
});
