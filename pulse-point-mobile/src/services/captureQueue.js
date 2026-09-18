const DEFAULTS = Object.freeze({
  intervalMs: 800,
  requestTimeoutMs: 5500,
  maxRetries: 2,
  backoffBaseMs: 250,
  backoffMaxMs: 2000,
});

function abortError(message = 'Capture cancelled.') {
  const error = new Error(message);
  error.name = 'AbortError';
  error.code = 'CAPTURE_CANCELLED';
  return error;
}

function isAbort(error) {
  return error?.name === 'AbortError' || error?.code === 'ABORT_ERR' || error?.code === 'CAPTURE_CANCELLED';
}

function withAbort(promise, signal) {
  if (!signal) return promise;
  if (signal.aborted) return Promise.reject(abortError());
  return new Promise((resolve, reject) => {
    const onAbort = () => reject(abortError());
    signal.addEventListener('abort', onAbort, { once: true });
    Promise.resolve(promise).then(
      (value) => { signal.removeEventListener('abort', onAbort); resolve(value); },
      (error) => { signal.removeEventListener('abort', onAbort); reject(error); },
    );
  });
}

function abortableDelay(delayMs, signal) {
  if (!signal) return new Promise((resolve) => setTimeout(resolve, delayMs));
  if (signal.aborted) return Promise.reject(abortError());
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      signal.removeEventListener('abort', onAbort);
      resolve();
    }, delayMs);
    const onAbort = () => {
      clearTimeout(timer);
      signal.removeEventListener('abort', onAbort);
      reject(abortError());
    };
    signal.addEventListener('abort', onAbort, { once: true });
  });
}

/**
 * Runs exactly one capture -> request -> result cycle at a time.
 *
 * The queue deliberately uses a one-shot timer instead of setInterval. A
 * new timer is installed only after the previous request has settled, so a
 * slow camera or network cannot create overlapping work.
 */
export class CaptureQueue {
  constructor({ capture, request, onResult, onError, onMiss, ...options }) {
    if (typeof capture !== 'function' || typeof request !== 'function') {
      throw new TypeError('CaptureQueue requires capture and request functions.');
    }
    this.capture = capture;
    this.request = request;
    this.onResult = onResult;
    this.onError = onError;
    this.onMiss = onMiss;
    this.options = { ...DEFAULTS, ...options };
    this.running = false;
    this.inFlight = false;
    this.timer = null;
    this.controller = null;
    this.generation = 0;
    this.target = null;
    this.sequence = 0;
  }

  start(target) {
    this.stop();
    this.running = true;
    this.target = target;
    const generation = ++this.generation;
    this._schedule(0, generation);
  }

  stop() {
    this.running = false;
    this.generation += 1;
    if (this.timer !== null) {
      clearTimeout(this.timer);
      this.timer = null;
    }
    if (this.controller) {
      this.controller.abort();
      this.controller = null;
    }
  }

  isRunning() {
    return this.running;
  }

  _schedule(delay, generation) {
    if (!this.running || generation !== this.generation) return;
    this.timer = setTimeout(() => {
      this.timer = null;
      this._run(generation).catch((error) => this._fail(error, generation));
    }, Math.max(0, delay));
  }

  async _run(generation) {
    if (!this.running || generation !== this.generation || this.inFlight) return;
    this.inFlight = true;
    const target = this.target;
    const sequence = ++this.sequence;

    try {
      const result = await this._attempt(target, generation);
      if (!this.running || generation !== this.generation) return;
      if (result == null) {
        this.onMiss?.({ generation, sequence, target });
      } else {
        this.onResult?.({ generation, sequence, target, result });
      }
      this._schedule(this.options.intervalMs, generation);
    } catch (error) {
      this._fail(error, generation, sequence, target);
    } finally {
      this.inFlight = false;
    }
  }

  async _attempt(target, generation) {
    let lastError = null;
    for (let retry = 0; retry <= this.options.maxRetries; retry += 1) {
      if (!this.running || generation !== this.generation) throw abortError();
      const controller = new AbortController();
      this.controller = controller;
      let timeoutId = null;
      let timedOut = false;
      try {
        timeoutId = setTimeout(() => {
          timedOut = true;
          controller.abort();
        }, this.options.requestTimeoutMs);
        const captureResult = await withAbort(this.capture({ target, signal: controller.signal }), controller.signal);
        if (!captureResult) return null;
        return await withAbort(this.request(captureResult, {
          target,
          signal: controller.signal,
          timeoutMs: this.options.requestTimeoutMs,
          retry,
        }), controller.signal);
      } catch (error) {
        if (timedOut) {
          lastError = Object.assign(
            new Error(`Vision request timed out after ${this.options.requestTimeoutMs} ms.`),
            { name: 'TimeoutError', code: 'CAPTURE_TIMEOUT' },
          );
        } else {
          lastError = error;
        }
        if (isAbort(error) || !this.running || generation !== this.generation) throw error;
        if (retry >= this.options.maxRetries) break;
        const delay = Math.min(
          this.options.backoffBaseMs * (2 ** retry),
          this.options.backoffMaxMs,
        );
        await abortableDelay(delay, controller.signal);
      } finally {
        if (timeoutId !== null) clearTimeout(timeoutId);
        if (this.controller === controller) this.controller = null;
      }
    }
    throw lastError || new Error('Capture failed.');
  }

  _fail(error, generation, sequence, target) {
    if (isAbort(error) || !this.running || generation !== this.generation) return;
    this.onError?.({ error, generation, sequence, target });
  }
}

export { DEFAULTS as CAPTURE_QUEUE_DEFAULTS };
