import assert from 'node:assert/strict';
import { CaptureQueue } from './captureQueue.js';
import {
  createInitialGuidanceState,
  EVENTS,
  mobileGuidanceReducer,
} from './guidanceState.js';

const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function noOverlapAndBackoff() {
  let active = 0;
  let maxActive = 0;
  let requests = 0;
  const queue = new CaptureQueue({
    intervalMs: 2,
    backoffBaseMs: 2,
    requestTimeoutMs: 30,
    maxRetries: 2,
    capture: async () => {
      active += 1;
      maxActive = Math.max(maxActive, active);
      await wait(3);
      active -= 1;
      return { uri: 'frame.jpg' };
    },
    request: async () => {
      requests += 1;
      if (requests < 3) throw new Error('transient');
      return { detected: false };
    },
  });
  queue.start('cup');
  for (let attempt = 0; attempt < 20 && requests < 3; attempt += 1) await wait(5);
  queue.stop();
  assert.equal(maxActive, 1, 'capture cycles must not overlap');
  assert.equal(requests, 3, 'transient failures should use bounded retry/backoff');
}

async function timeoutAndCancel() {
  let signal;
  let errors = 0;
  const queue = new CaptureQueue({
    intervalMs: 2,
    requestTimeoutMs: 8,
    maxRetries: 0,
    capture: async () => ({ uri: 'frame.jpg' }),
    request: async (_frame, options) => {
      signal = options.signal;
      await new Promise((resolve, reject) => {
        const onAbort = () => reject(Object.assign(new Error('timed out'), { name: 'AbortError' }));
        options.signal.addEventListener('abort', onAbort, { once: true });
      });
    },
    onError: () => { errors += 1; },
  });
  queue.start('book');
  await wait(3);
  queue.stop();
  await wait(3);
  assert.equal(signal?.aborted, true, 'stop must abort the active request');
  assert.equal(errors, 0, 'cancellation is not a detector error');
}

function reducerTransitions() {
  const initial = createInitialGuidanceState();
  const loading = mobileGuidanceReducer(initial, { type: EVENTS.START, target: 'mug' });
  assert.equal(loading.status, 'loading');
  const locked = mobileGuidanceReducer(loading, {
    type: EVENTS.DETECTION_RECEIVED,
    nowMs: 1000,
    detection: {
      label: 'mug', displayLabel: 'Mug', confidence: 0.9,
      bbox: { x: 0.43, y: 0.43, width: 0.08, height: 0.08 },
      source: 'server', assistiveReady: false,
      distance: { meters: 1.5, method: 'area-estimate', uncertaintyMeters: 0.8 },
      timestampMs: 1000,
    },
  });
  assert.equal(locked.assistiveReady, false);
  assert.notEqual(locked.status, 'reach');
  const lost = mobileGuidanceReducer(locked, { type: EVENTS.DETECTION_MISSED, nowMs: 3000 });
  assert.equal(lost.status, 'lost');
  const reset = mobileGuidanceReducer(lost, { type: EVENTS.RESET });
  assert.equal(reset.status, 'idle');
}

await noOverlapAndBackoff();
await timeoutAndCancel();
reducerTransitions();
console.log('Milestone 4 smoke checks passed: no overlap, timeout/cancel, backoff, reducer safety.');
