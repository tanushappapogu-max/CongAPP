// Service worker for Pulse Point.
// The web detector loads the YOLOE model, its prompt pack, and the depth model from
// the public root. Keep this list exact so stale model formats are not mixed into the
// runtime cache. The prompt pack must match the model, so bump CACHE_VERSION whenever
// either is rebuilt. The ONNX Runtime binary is emitted by Vite under a content-hashed
// /assets/ name, so it is cached by pattern and never goes stale.
const CACHE_VERSION = 'v5';
const MODEL_CACHE = `pulse-point-model-${CACHE_VERSION}`;
const APP_CACHE   = `pulse-point-app-${CACHE_VERSION}`;
const MODEL_ASSETS = new Set([
  '/yoloe-11s.onnx',
  '/prompts/pack.json',
  '/prompts/pack.bin',
  '/depth-indoor-small.fp16.onnx',
  '/depth-indoor-small.uint8.onnx',
]);
const HASHED_RUNTIME = /^\/assets\/ort-wasm-[\w.-]+\.wasm$/;

self.addEventListener('install', event => {
  // Activate immediately without waiting for old clients to close.
  event.waitUntil(self.skipWaiting());
});

self.addEventListener('activate', event => {
  // Take control of existing clients so the cache is available right away.
  event.waitUntil(
    Promise.all([
      self.clients.claim(),
      // Bumping CACHE_VERSION invalidates the model/runtime and app caches.
      caches.keys().then(keys =>
        Promise.all(
          keys
            .filter(k =>
              (k.startsWith('pulse-point-model-') || k.startsWith('pulse-point-app-')) &&
              k !== MODEL_CACHE &&
              k !== APP_CACHE,
            )
            .map(k => caches.delete(k)),
        ),
      ),
    ]),
  );
});

self.addEventListener('fetch', event => {
  const { request } = event;
  const url = new URL(request.url);

  // Only intercept same-origin GET requests.
  if (request.method !== 'GET' || url.origin !== self.location.origin) return;

  if (MODEL_ASSETS.has(url.pathname) || HASHED_RUNTIME.test(url.pathname)) {
    // Cache-first for immutable model/runtime assets. Never cache errors or
    // opaque responses, and let a failed first download surface normally.
    event.respondWith(
      caches.open(MODEL_CACHE).then(async cache => {
        const cached = await cache.match(request);
        if (cached) return cached;
        const response = await fetch(request);
        if (response.ok && response.type !== 'opaque') {
          await cache.put(request, response.clone());
        }
        return response;
      }),
    );
    return;
  }

  // Network-first for HTML / JS / CSS — fall back to cache when offline.
  event.respondWith(
    fetch(request)
      .then(response => {
        if (response.ok) {
          caches.open(APP_CACHE).then(cache => cache.put(request, response.clone()));
        }
        return response;
      })
      .catch(() => caches.match(request)),
  );
});
