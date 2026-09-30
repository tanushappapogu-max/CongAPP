# Pulse Point Architecture

This document describes the supported browser app and the adjacent experimental services. `pulse-point/` is the active web client. `pulse-point-mobile/` is a separate exploratory/degraded prototype, not an equivalent production client.

## High-Level Data Flow

```text
Web voice/text target
        |
        v
Target extraction and prompt-pack resolution
        |
        +--> Local detector selected by device and target
        |      WebGPU: YOLOE prompt detector
        |      CPU: YOLO11n for COCO targets; YOLOE on demand otherwise
        |
        +--> Optional server request every ~2.5 seconds when configured
        |      Local match wins; server result is used if local has no match
        |
        v
Tracker + temporal smoothing
        |
        v
Depth sample / width estimate + direction computation
        |
        v
Prototype haptics and speech output
```

## Web App (`pulse-point/`)

- React + Vite frontend handles camera access and render loop.
- `detection/engine.js` runs YOLOE locally with WebGPU when available. On CPU, YOLO11n (`/net.onnx`, 80 COCO classes) is the normal local detector for targets with a COCO label because it is about 4× lighter than YOLOE; YOLOE loads on demand for other targets represented by the prompt pack. If YOLOE fails to load on the WebGPU path, the engine tries YOLO11n, which still has only COCO coverage. Vite emits a content-hashed ONNX Runtime WASM asset for the app.
- `detection/prompts.js` loads `/prompts/pack.json` and `/prompts/pack.bin` (precomputed YOLOE text embeddings built by `scripts/yoloe/build_prompt_pack.py`). Targets resolve through the pack's names/aliases and COCO aliases/fuzzy matching. Prompt sets include look-alike negatives where available. Local inference requires a resolved prompt set; this is a finite prompt pack, not arbitrary natural-language understanding. Pack items may include a reference width for the distance fallback.
- If `VITE_SERVER_URL` is configured and its health check passes, `App.jsx` starts a remote request roughly every 2.5 seconds while a target is set, regardless of whether local detection currently has a match. `server.js` captures the current camera frame as JPEG and uploads it with the target. The app gives a local match precedence and uses a returned server result only when local matching has no result. Therefore, remote results are fallback results, but remote requests are not currently gated on local failure. The server is optional, experimental, and unvalidated.
- `detection/depth.js` and `depth.worker.js` run Depth Anything V2 Metric-Indoor-Small in a Web Worker. The worker uses the FP16 asset with WebGPU and the uint8 asset with WASM. `depthSample.js` samples the detected box, and `compute.js` prefers a recent depth reading before using its width-based estimate. Depth runs every 0.5 seconds on WebGPU; on WASM it runs every 4 seconds and only while the target is roughly centered. These estimates have not been validated for safety or across representative environments.
- Depth runs on the largest centered square of the frame (518 px, corrected by 1/1.03). On indoor photos it read within about 2% of the full frame at about half the peak memory, because the model compares every image patch with every other. On the CPU path (iPhone) the depth worker is closed after each reading so its memory is handed back. `?depthsize=392` uses a smaller crop (about 200 MB peak, corrected by 1/1.15).
- Safari's engine (all iOS browsers and desktop Safari) uses CPU inference by default to avoid observed WebGPU memory growth. `?gpu=1` re-enables WebGPU for testing; `?cpu`, `?nodepth`, `?nodetect`, and `?captureonly` isolate specific paths during diagnosis.
- COOP/COEP headers in `vercel.json` and the Vite dev/preview servers make the site cross-origin isolated, allowing the CPU detector to use up to four WASM threads. ONNX Runtime is loaded as a separate chunk because its thread workers start from the file containing the runtime.
- `lib/camera.js` selects the widest available back camera and applies its minimum supported zoom. `lensFovDeg` accounts for the selected lens and zoom in the width-based distance estimate.
- `tracker.js` stabilizes noisy frame-to-frame detections.
- `compute.js` determines directional guidance (`left`, `right`, `up`, `down`, `locked`, `closer`, `reach`). The “reach” signal is a heuristic based on a depth threshold or box area, not confirmation that reaching or moving is safe.
- `public/sw.js` uses a versioned cache for selected model, prompt, depth, and ONNX Runtime assets. Caching can improve repeat loads after successful downloads; it does not guarantee offline camera access, app navigation, or inference on every browser.

## Mobile App (`pulse-point-mobile/`)

- Expo app provides the user flow, haptics, and orientation guidance UX.
- This is an exploratory/degraded prototype and should not be treated as the active web client. Its camera, haptics, server, and simulation flows are separate from the supported `pulse-point/` implementation.

## Experimental Python detector (`server/`)

- `/detect` and `/v1/detect` use the same handler. For a nonempty target, the service tries LocateAnything-3B first; if it is unavailable or returns no result, it uses an ImageNet classifier mapped to an indoor-object ontology with approximate Grad-CAM localization. The fallback is not trained or evaluated as a bounding-box detector. The returned LocateAnything score is currently a fixed value, not a calibrated probability.
- Responses carry experimental/unvalidated metadata and hard `assistiveReady: false` / `proof: false` signals. The shared metadata contract still reports the legacy `approximate/Grad-CAM` localization method, so do not treat that field as a precise description of the LocateAnything path.
- The boundary validates JPEG/PNG/WebP MIME and decoded content, bounds image and target sizes, limits detector requests per IP, and uses explicit configurable CORS origins. `server/README.md` is the source of the endpoint details.
- The tea-text classifier routes are unrelated legacy capabilities.

## AI Proxy (`api/`)

- `/api/ai` is a server-side proxy to OpenRouter/Gemini.
- API key remains server-only (`OPENROUTER_API_KEY`), never exposed in browser bundle.
- Existing boundary controls include a model allow-list, max-token cap, strict origin enforcement, per-IP in-memory rate limiting, and a 30-second Vercel function duration.
- The web scanner does not call it. It is a legacy endpoint and is not a vision fallback.

## Error Reporting

- `ErrorBoundary` catches frontend render crashes and posts a short payload to `/api/error`.
- `/api/error` writes structured crash data to Vercel function logs for live-demo debugging.
