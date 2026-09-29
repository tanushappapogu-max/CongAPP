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
(depth worker meters, else distance.js widths + guidance/compute.js)
        |
        v
Guidance outputs
  - Haptics pattern (guidance/haptics.js)
  - Speech prompts (guidance/speech.js, lib/voice.js)
```

## Web App (`pulse-point/`)

- React + Vite frontend handles camera access and render loop.
- `detection/engine.js` loads `/yoloe-11s.onnx` with `onnxruntime-web/webgpu`, preferring the WebGPU execution provider and falling back to single-threaded WASM when the browser has no usable GPU. `/net.onnx` (YOLO11n, 80 COCO classes) is the backup: devices without WebGPU use it for COCO targets because it is about 4× lighter than YOLOE on the CPU, loading YOLOE on demand only for targets outside COCO, and it takes over entirely if YOLOE fails to load. Vite emits the ONNX Runtime binary as a content-hashed `/assets/ort-wasm-simd-threaded.asyncify-*.wasm`, shared by the engine and the depth worker. The model takes two inputs: `images` (1×3×640×640) and `pe` (1×K×512 prompt embeddings, K dynamic), and returns `output0` (1×(4+K)×8400).
- `detection/prompts.js` loads `/prompts/pack.json` + `/prompts/pack.bin` (precomputed YOLOE text embeddings built by `scripts/yoloe/build_prompt_pack.py`). A target resolves through pack names/aliases, then the existing COCO aliases in `coco.js`, then fuzzy matching. Each search feeds the target's vector plus its look-alike negatives (e.g. eyeglasses vs. sunglasses), so a box only counts when it scores highest for the target. Pack items carry a real-world width used by `distance.js`. Targets outside the pack fall through to the server and Gemini paths.
- `detection/depth.js` + `depth.worker.js` run Depth Anything V2 Metric-Indoor-Small in a Web Worker with its own ONNX Runtime instance, so it never blocks detection. The worker uses `/depth-indoor-small.fp16.onnx` on WebGPU and falls back to `/depth-indoor-small.uint8.onnx` on WASM; both are exported from the same graph by `scripts/depth/export_depth.py`. While the target is visible it measures every 0.5 s on WebGPU or 1.5 s on WASM: the current frame is measured at a 518 px short side and `depthSample.js` takes a foreground-biased percentile of the depth inside the target box. Between measurements the reading is carried forward by box-width ratio, and it expires after 8 s. `compute.js` prefers these meters and falls back to width-based distance.
- Safari's engine (every iOS browser and desktop Safari) runs detection on the CPU by default (`lib/flags.js`): on an iPhone the WebGPU path leaked about 150 MB/s while scanning and the tab was killed within seconds, while the CPU path stayed flat. `?gpu=1` re-enables WebGPU for testing; `?cpu`, `?nodepth`, `?nodetect` and `?captureonly` isolate memory problems.
- The site is cross-origin isolated (COOP `same-origin` + COEP `require-corp` in `vercel.json` and the Vite dev/preview servers) so the CPU backend can use up to 4 threads, about 3.5× faster than one. ONNX Runtime is loaded as its own chunk (`engine.js` imports it lazily) because its thread workers start from the file it lives in.
- The camera uses the widest back lens at its widest zoom (`lib/camera.js`), and `lensFovDeg` reports that lens's field of view (about 103° for the 0.5× ultra-wide) so width-based distance stays correct. On the CPU, depth only runs every 4 s and only while the target is roughly centered.
- `tracker.js` stabilizes noisy frame-to-frame detections.
- `compute.js` determines directional guidance (`left`, `right`, `up`, `down`, `locked`, `closer`, `reach`).
- `public/sw.js` uses a versioned cache-first strategy for the exact immutable assets `/yoloe-11s.onnx`, `/prompts/pack.json`, `/prompts/pack.bin`, `/depth-indoor-small.fp16.onnx`, and `/depth-indoor-small.uint8.onnx`, plus the hashed ONNX Runtime binary under `/assets/`. The pack must come from the same checkpoint as the model, so bump `CACHE_VERSION` whenever either is rebuilt.
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
