# Pulse Point Architecture

This document summarizes the core end-to-end flow for both the web and mobile prototypes.

## High-Level Data Flow

```text
User voice/text target
        |
        v
Target extraction (voice.js)
        |
        v
COCO label resolution (coco.js)
        |
        v
Web: YOLO11n ONNX/WASM inference loop (detection/engine.js)
Mobile: configured experimental server detector
        (or explicit demo-only simulation)
        |
        v
Tracker + smoothing (tracker.js)
        |
        v
Distance + direction computation
(distance.js + guidance/compute.js)
        |
        v
Guidance outputs
  - Haptics pattern (guidance/haptics.js)
  - Speech prompts (guidance/speech.js, lib/voice.js)
```

## Web App (`pulse-point/`)

- React + Vite frontend handles camera access and render loop.
- `detection/engine.js` loads `/net.onnx` with `onnxruntime-web` and the WASM execution provider. With `simd: true` and `numThreads: 1`, the deployed runtime asset is `/ort-wasm-simd.wasm`.
- `tracker.js` stabilizes noisy frame-to-frame detections.
- `compute.js` determines directional guidance (`left`, `right`, `up`, `down`, `locked`, `closer`, `reach`).
- `public/sw.js` uses a versioned cache-first strategy for the exact immutable assets `/net.onnx` and `/ort-wasm-simd.wasm`; the old `yolo11n_web_model` shard path is no longer part of the web detector.
- Cached assets can speed up repeat loads after a successful download, but the service worker does not guarantee offline camera access, navigation, or inference on every browser.

## Mobile App (`pulse-point-mobile/`)

- Expo app provides the user flow, haptics, and orientation guidance UX.
- The app requires a configured server detector for normal scanning. Missing configuration, failed health checks, and request errors surface as `UNAVAILABLE`/error; there is no automatic simulation fallback.
- Simulation is available only through explicit opt-in (`EXPO_PUBLIC_PULSEPOINT_ALLOW_SIMULATION=true` or `extra.pulsepointAllowSimulation: true`), is labeled demo-only, and cannot produce a reach signal.
- The server `/detect` path is experimental and unvalidated; its response is explicitly never assistive-ready. The mobile prototype makes no production assistive or safety claim.
- App structure keeps a clear seam for future native detection integration (ML Kit/CoreML/ARKit/ARCore).

## Experimental Python detector (`server/`)

- `/detect` is retained for mobile compatibility; `/v1/detect` is its versioned alias and both use the same bounded handler.
- The detector maps ImageNet classification probabilities to an indoor-object ontology and uses approximate Grad-CAM for a debug localization overlay. It is not trained or evaluated for bounding-box localization.
- Responses expose detector version, experimental status, uncalibrated confidence, localization method, supported target metadata, and hard `assistiveReady: false` / `proof: false` signals. Raw probabilities are not multiplied, capped, or subset-renormalized.
- The boundary validates JPEG/PNG/WebP MIME and decoded content, bounds image and target sizes, limits detector requests per IP, and uses explicit configurable CORS origins. `server/README.md` is the source of the endpoint details.
- The tea-text classifier routes are unrelated legacy capability and remain functional while the production Pulse Point detector path is undecided.

## AI Proxy (`api/`)

- `/api/ai` is a server-side proxy to OpenRouter/Gemini.
- API key remains server-only (`OPENROUTER_API_KEY`), never exposed in browser bundle.
- Existing boundary controls include a model allow-list, max-token cap, strict origin enforcement, per-IP in-memory rate limiting, and a 30-second Vercel function duration.
- It remains optional cloud-assist infrastructure; the Python detector's experimental status does not make cloud responses validated assistive sensing.

## Error Reporting

- `ErrorBoundary` catches frontend render crashes and posts a short payload to `/api/error`.
- `/api/error` writes structured crash data to Vercel function logs for live-demo debugging.
