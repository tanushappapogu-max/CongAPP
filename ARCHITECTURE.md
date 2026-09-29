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
Prompt-pack resolution (prompts.js): target + look-alike negatives
        |
        v
Web: promptable YOLOE-11s ONNX/WASM inference loop (detection/engine.js)
Mobile: configured experimental server detector
        (or explicit demo-only simulation)
        |
        v
Tracker + smoothing (tracker.js)
        |
        v
Distance + direction computation
(GPU depth meters, else distance.js widths + guidance/compute.js)
        |
        v
Guidance outputs
  - Haptics pattern (guidance/haptics.js)
  - Speech prompts (guidance/speech.js, lib/voice.js)
```

## Web App (`pulse-point/`)

- React + Vite frontend handles camera access and render loop.
- `detection/engine.js` loads YOLOE with `onnxruntime-web/webgpu`: `/yoloe-11s.fp16.onnx` on WebGPU (half the memory, faster on phone GPUs) and `/yoloe-11s.onnx` on WASM, preferring the WebGPU execution provider and falling back to single-threaded WASM when the browser has no usable GPU. `/net.onnx` (YOLO11n, 80 COCO classes) is the backup: devices without WebGPU use it for COCO targets because it is about 4× lighter than YOLOE on the CPU, loading YOLOE on demand only for targets outside COCO, and it takes over entirely if YOLOE fails to load. Vite emits the ONNX Runtime binary as a content-hashed `/assets/ort-wasm-simd-threaded.asyncify-*.wasm`, shared by the detector and the depth model. The model takes two inputs: `images` (1×3×640×640) and `pe` (1×K×512 prompt embeddings, K dynamic), and returns `output0` (1×(4+K)×8400).
- `detection/prompts.js` loads `/prompts/pack.json` + `/prompts/pack.bin` (precomputed YOLOE text embeddings built by `scripts/yoloe/build_prompt_pack.py`). A target resolves through pack names/aliases, then the existing COCO aliases in `coco.js`, then fuzzy matching. Each search feeds the target's vector plus its look-alike negatives (e.g. eyeglasses vs. sunglasses), so a box only counts when it scores highest for the target. Pack items carry a real-world width used by `distance.js`. Targets outside the pack fall through to the server and Gemini paths.
- `detection/depth.js` runs Depth Anything V2 Metric-Indoor-Small (`/depth-indoor-small.fp16.onnx`, exported by `scripts/depth/export_depth.py`) on WebGPU only, sharing the detector's ONNX Runtime instance and GPU device; `engine.js`'s `runExclusive` queue keeps the two models from running at once. The model downloads only after the first detection. While the target is visible it measures every 0.5 s on desktop or 2 s on phones: the current frame is measured at a 518 px short side and `depthSample.js` takes a foreground-biased percentile of the depth inside the target box. Between measurements the reading is carried forward by box-width ratio, and it expires after 8 s. Devices without WebGPU skip depth (it takes seconds per frame on the CPU) and use width-based distance.
- `tracker.js` stabilizes noisy frame-to-frame detections.
- `compute.js` determines directional guidance (`left`, `right`, `up`, `down`, `locked`, `closer`, `reach`).
- `public/sw.js` uses a versioned cache-first strategy for the exact immutable assets `/yoloe-11s.onnx`, `/prompts/pack.json`, `/prompts/pack.bin`, `/yoloe-11s.fp16.onnx`, `/net.onnx`, and `/depth-indoor-small.fp16.onnx`, plus the hashed ONNX Runtime binary under `/assets/`. The pack must come from the same checkpoint as the model, so bump `CACHE_VERSION` whenever either is rebuilt.
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
- The web app no longer calls it; the Gemini cloud fallback was removed from the detection loop.

## Error Reporting

- `ErrorBoundary` catches frontend render crashes and posts a short payload to `/api/error`.
- `/api/error` writes structured crash data to Vercel function logs for live-demo debugging.
