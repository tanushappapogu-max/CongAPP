# 6ixth Sense Architecture

This document describes the supported browser app, the vision server it uses by default, and the adjacent legacy services. `pulse-point/` is the active web client. `pulse-point-mobile/` is a separate exploratory/degraded prototype, not an equivalent production client.

## High-Level Data Flow

```text
Web voice/text target
        |
        v
Target extraction and prompt-pack resolution
        |
        v
Detector selection (detection/engine.js)
        |
        +--> Vision server, when configured and healthy (server/vision/)
        |      Phone uploads a 640x640 letterboxed JPEG; server runs YOLOE and returns boxes
        |      Any request failure marks it down; health is rechecked after ~15 s
        |
        +--> On-device fallback (ONNX Runtime Web)
        |      WebGPU: YOLOE prompt detector
        |      CPU: YOLO11n for COCO targets; YOLOE on demand otherwise
        |
        v
Tracker + temporal smoothing
        |
        v
Distance: Depth Anything V2 (server or worker) / width estimate
        |
        v
Direction computation -> prototype haptics and speech output
```

The legacy LocateAnything server (`VITE_SERVER_URL`) can still run alongside this as a separate, optional probe. It is described under the web app below.

## Web App (`pulse-point/`)

### Detection

- React + Vite frontend handles camera access and the render loop. Inference runs in the background so the overlay and guidance update every frame.
- `detection/remote.js` is the vision server client. The URL is `VITE_VISION_URL` (default: the Modal deployment), overridden by `?server=URL` and disabled by `?server=off`. On page open, `preloadModel()` calls the health check, which keeps polling through 503s and network errors for up to 90 seconds while a scaled-to-zero server starts. While the server is up, the phone downloads no detector or depth models.
- In `detection/engine.js`, `runInference` uses the server whenever `isRemoteUp()` is true. With the server, the frame loop allows two requests in flight at roughly 66 ms spacing, and round-trip time sets the actual rate. A 429 or 503 skips that frame. Any other error or a network failure marks the server down, logs a warning, starts loading the on-device detector, and returns no boxes for that frame. While the server is down, `isRemoteUp()` retries the health check at most every 15 seconds; once it passes, detection moves back to the server.
- Without a healthy server, `engine.js` runs YOLOE locally with WebGPU when available. On CPU, YOLO11n (`/net.onnx`, 80 COCO classes) is the normal local detector for targets with a COCO label because it is about 4× lighter than YOLOE; YOLOE loads on demand for other targets represented by the prompt pack. If YOLOE fails to load on the WebGPU path, the engine tries YOLO11n, which still has only COCO coverage. Vite emits a content-hashed ONNX Runtime WASM asset for the app.
- `detection/prompts.js` loads `/prompts/pack.json` and `/prompts/pack.bin` (precomputed YOLOE text embeddings built by `scripts/yoloe/build_prompt_pack.py`). Targets resolve through the pack's names/aliases and COCO aliases/fuzzy matching. Prompt sets include look-alike negatives where available. Inference requires a resolved prompt set; this is a finite prompt pack, not arbitrary natural-language understanding. Pack items may include a reference width for the distance fallback.
- Legacy probe: if `VITE_SERVER_URL` is configured and its health check passes, `App.jsx` also starts a request to the LocateAnything server (`detection/server.js`) roughly every 2.5 seconds while a target is set, regardless of whether local detection currently has a match. A local match takes precedence; the legacy result is used only when there is none. This server is optional, experimental, and unvalidated, and it is separate from the vision server.

### Distance

- `detection/depth.js` runs Depth Anything V2 Metric-Indoor-Small. Both paths use the largest centered square of the frame and send a box in crop-relative fractions. With the vision server up, the crop goes to `/v1/depth` and the server applies the crop correction. Otherwise a Web Worker (`depth.worker.js`) runs the FP16 asset with WebGPU or the uint8 asset with WASM. When a server is configured, the worker model is not preloaded and only loads if the server is unavailable.
- Depth readings are requested about every 1 second with the server, every 0.5 seconds on WebGPU, and every 4 seconds on WASM (WASM only while the target is roughly centered). `depthSample.js` samples the detected box, and `compute.js` prefers a recent depth reading before using its width-based estimate. These estimates have not been validated for safety or across representative environments.
- The 518 px square crop is corrected by 1/1.03. On indoor photos it read within about 2% of the full frame at about half the peak memory, because the model compares every image patch with every other. One depth worker stays alive for the session; closing and reopening it after every reading crashed iPhone Safari. `?depthsize=392` uses a smaller on-device crop (about 200 MB peak, corrected by 1/1.15).
- `lib/camera.js` selects the widest available back camera and applies its minimum supported zoom. `lensFovDeg` accounts for the selected lens and zoom in the width-based distance estimate.

### Device constraints

- Safari's engine (all iOS browsers and desktop Safari) uses CPU inference by default to avoid observed WebGPU memory growth. `?gpu=1` re-enables WebGPU for testing; `?cpu`, `?nodepth`, `?nodetect`, and `?captureonly` isolate specific paths during diagnosis.
- The site is not cross-origin isolated, so the CPU detector runs single-threaded. COOP/COEP isolation let it use up to four threads (about 3.5× faster), but with isolation on iPhone Safari crashed whenever depth ran, even single-threaded. ONNX Runtime is still loaded lazily as its own chunk (`engine.js`), which threads would need if isolation is revisited.

### Guidance and caching

- `tracker.js` stabilizes noisy frame-to-frame detections.
- `compute.js` determines directional guidance (`left`, `right`, `up`, `down`, `locked`, `closer`, `reach`). The “reach” signal is a heuristic based on a depth threshold or box area, not confirmation that reaching or moving is safe.
- `public/sw.js` uses a versioned cache for selected model, prompt, depth, and ONNX Runtime assets. Caching can improve repeat loads after successful downloads; it does not guarantee offline camera access, app navigation, or inference on every browser.

## Vision Server (`server/vision/`)

- FastAPI app (`vision_app.py`) that loads the same `yoloe-11s.onnx`, `depth-indoor-small.fp16.onnx`, and prompt pack the web app ships, from `pulse-point/public/` (override with `PULSEPOINT_ASSETS`). It uses the CUDA execution provider when available, otherwise CPU. It does not run YOLO11n.
- `GET /health` returns `{"ok": true, "device": ...}`. `POST /v1/detect` takes a 640×640 letterboxed JPEG and a prompt-pack item name, builds the same target-plus-negatives prompt set as `prompts.js`, and returns boxes in 640 px input space. `POST /v1/depth` takes the square crop and a crop-relative box and returns meters with the 1.03 crop correction applied.
- Inference runs in a worker thread so a slow depth request does not block detection. An unknown target or invalid box returns 400.
- Boundary controls: CORS restricted to `cors_origins()` from `server/contract.py`, JPEG/PNG/WebP only, images capped at 1.5 MB, and an in-memory per-IP limit of 900 requests per minute (429 when exceeded).
- `modal_vision.py` deploys it on Modal with a T4 GPU. The container scales to zero after a few idle minutes; the client's health polling covers the cold start.
- Privacy: while the server is in use, camera frames and the target name are uploaded on every detection and depth request. The server does not store them.

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
