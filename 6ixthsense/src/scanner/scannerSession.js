export const SCANNER_EVENTS = Object.freeze({
  START: 'START',
  STOP: 'STOP',
  TARGET_SET: 'TARGET_SET',
  DETECTION_RECEIVED: 'DETECTION_RECEIVED',
  DETECTION_MISSED: 'DETECTION_MISSED',
  MODEL_ERROR: 'MODEL_ERROR',
  CAMERA_ERROR: 'CAMERA_ERROR',
});

const DEFAULT_INTERVAL_MS = 100;
const DEFAULT_LOST_AFTER_MISSES = 2;

const defaultScheduler = {
  now: () => Date.now(),
  setTimeout: (callback, delay) => setTimeout(callback, delay),
  clearTimeout: (handle) => clearTimeout(handle),
};

function makeAbortController() {
  if (typeof AbortController === 'function') return new AbortController();
  return { signal: { aborted: false }, abort() { this.signal.aborted = true; } };
}

function callStopHook(hook, resource) {
  if (!resource) return Promise.resolve();
  if (hook) return Promise.resolve(hook(resource));
  if (typeof resource.stop === 'function') return Promise.resolve(resource.stop());
  if (typeof resource.dispose === 'function') return Promise.resolve(resource.dispose());
  return Promise.resolve();
}

/** Creates a reusable, browser-independent scanner lifecycle. */
export function createScannerSession({
  initializeCamera,
  initializeModel,
  detect,
  computeGuidance,
  scheduler = defaultScheduler,
  intervalMs = DEFAULT_INTERVAL_MS,
  lostAfterMisses = DEFAULT_LOST_AFTER_MISSES,
  onEvent = () => {},
  onDetection = () => {},
  onGuidance = () => {},
  onOutput = () => {},
  stopCamera,
  disposeModel,
} = {}) {
  if (typeof initializeCamera !== 'function') throw new TypeError('initializeCamera is required');
  if (typeof initializeModel !== 'function') throw new TypeError('initializeModel is required');
  if (typeof detect !== 'function') throw new TypeError('detect is required');

  let generation = 0;
  let target = null;
  let phase = 'idle';
  let camera = null;
  let model = null;
  let timer = null;
  let detectorInFlight = false;
  let controller = null;
  let missedCount = 0;
  let disposed = false;

  const emit = (type, payload = {}) => {
    const event = Object.freeze({ type, target, generation, timestampMs: scheduler.now(), ...payload });
    onEvent(event);
    return event;
  };

  const isCurrent = (token, expectedTarget = target) => (
    !disposed && token === generation && expectedTarget === target && controller
      && !controller.signal.aborted && (phase === 'loading' || phase === 'running')
  );

  const clearTimer = () => {
    if (timer !== null) scheduler.clearTimeout(timer);
    timer = null;
  };

  const cancelGeneration = () => {
    generation += 1;
    clearTimer();
    controller?.abort();
    controller = null;
  };

  const closeCamera = (resource = camera) => {
    camera = null;
    return callStopHook(stopCamera, resource);
  };

  const scheduleDetection = (token, expectedTarget) => {
    if (!isCurrent(token, expectedTarget) || timer !== null || detectorInFlight) return;
    timer = scheduler.setTimeout(() => {
      timer = null;
      runDetection(token, expectedTarget);
    }, Math.max(0, intervalMs));
  };

  async function runDetection(token, expectedTarget) {
    if (!isCurrent(token, expectedTarget) || detectorInFlight) return;
    detectorInFlight = true;
    try {
      const result = await detect({
        camera,
        model,
        target: expectedTarget,
        signal: controller.signal,
        sessionId: token,
        nowMs: scheduler.now(),
      });
      if (!isCurrent(token, expectedTarget)) return;

      const detections = Array.isArray(result) ? result : result == null ? [] : [result];
      if (detections.length === 0) {
        missedCount += 1;
        const status = missedCount >= lostAfterMisses ? 'lost' : 'reacquiring';
        const event = emit(SCANNER_EVENTS.DETECTION_MISSED, {
          missedCount, status, lost: status === 'lost',
        });
        onOutput(event);
      } else {
        const detection = detections[0];
        const guidance = computeGuidance ? computeGuidance(detection, {
          target: expectedTarget, missedCount, nowMs: scheduler.now(),
        }) : undefined;
        const event = emit(SCANNER_EVENTS.DETECTION_RECEIVED, {
          detections, detection, guidance, reacquired: missedCount > 0,
        });
        missedCount = 0;
        onDetection(detection, event);
        if (guidance !== undefined) onGuidance(guidance, event);
        onOutput(event);
      }
    } finally {
      detectorInFlight = false;
      if (isCurrent(token, expectedTarget)) {
        scheduleDetection(token, expectedTarget);
      } else if (!disposed && phase === 'running') {
        // A retarget can cancel a detector promise. Let that promise finish,
        // then start exactly one detector for the newest generation.
        scheduleDetection(generation, target);
      }
    }
  }

  async function begin() {
    const token = generation;
    const expectedTarget = target;
    phase = 'loading';
    controller = makeAbortController();
    const signal = controller.signal;
    try {
      const initializedCamera = await initializeCamera({ target: expectedTarget, signal, sessionId: token });
      if (!isCurrent(token, expectedTarget)) { await callStopHook(stopCamera, initializedCamera); return; }
      camera = initializedCamera;
    } catch (error) {
      if (isCurrent(token, expectedTarget)) {
        phase = 'error';
        emit(SCANNER_EVENTS.CAMERA_ERROR, { error });
        await closeCamera();
      }
      return;
    }
    try {
      const cameraForModel = camera;
      const initializedModel = await initializeModel({ camera: cameraForModel, target: expectedTarget, signal, sessionId: token });
      if (!isCurrent(token, expectedTarget)) {
        if (disposeModel) await disposeModel(initializedModel);
        if (camera === cameraForModel) {
          camera = null;
          await callStopHook(stopCamera, cameraForModel);
        } else if (camera) {
          // A newer generation may already own `camera`; leave it alone while
          // releasing the resource captured by this stale initialization.
          await callStopHook(stopCamera, cameraForModel);
        }
        return;
      }
      model = initializedModel;
    } catch (error) {
      if (isCurrent(token, expectedTarget)) {
        phase = 'error';
        emit(SCANNER_EVENTS.MODEL_ERROR, { error });
        await closeCamera();
      }
      return;
    }
    phase = 'running';
    scheduleDetection(token, expectedTarget);
  }

  function start(nextTarget = target) {
    if (disposed) return Promise.resolve();
    cancelGeneration();
    target = nextTarget == null ? null : String(nextTarget);
    missedCount = 0;
    emit(SCANNER_EVENTS.START);
    return begin();
  }

  function stop() {
    if (disposed) return;
    cancelGeneration();
    phase = 'idle';
    missedCount = 0;
    const resource = camera;
    camera = null;
    void callStopHook(stopCamera, resource);
    emit(SCANNER_EVENTS.STOP);
  }

  function setTarget(nextTarget) {
    if (disposed) return;
    target = nextTarget == null ? null : String(nextTarget);
    missedCount = 0;
    emit(SCANNER_EVENTS.TARGET_SET, { target });
    if (phase === 'loading') {
      cancelGeneration();
      void begin();
    } else if (phase === 'running') {
      cancelGeneration();
      phase = 'running';
      controller = makeAbortController();
      scheduleDetection(generation, target);
    }
  }

  async function dispose() {
    if (disposed) return;
    stop();
    disposed = true;
    await closeCamera();
    if (model && disposeModel) await disposeModel(model);
    model = null;
  }

  return Object.freeze({ start, stop, setTarget, dispose });
}

export default createScannerSession;
