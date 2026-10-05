# 6ixth Sense Technical Overview

This document describes the implementation in the supported web app and the adjacent experimental server as they exist in this repository. It is an implementation reference, not a performance, safety, clinical, or accessibility validation report.

## Product Scope and Status

6ixth Sense is a browser prototype for finding a named object and presenting direction/proximity cues through a visual overlay, speech, and device vibration where supported. The active web client is pulse-point/. The Expo app in pulse-point-mobile/ is a separate exploratory/degraded prototype.

The project has not been validated for independent mobility, obstacle avoidance, route planning, or safety-critical use. A detection, distance estimate, or "reach" announcement must not be interpreted as proof that an object is safe to approach or touch. The repository includes unit tests for logic modules; it does not currently contain a representative device/scene accuracy benchmark or human-factors validation.

## Runtime Flow

1. The web app obtains a camera stream through the browser camera API after permission.
2. Voice or text provides a target. Voice recognition and target extraction are browser-dependent.
3. The target resolver looks for a match in the checked-in prompt pack, aliases, and COCO mappings. A resolved item provides the detector prompt set, including look-alike negatives where available.
4. The detector runs on the current video frame locally. Detector choice depends on WebGPU availability, model loading state, and whether the target maps to a COCO label.
5. A box tracker stabilizes detections. A worker estimates scene depth for the current target box; guidance combines that reading or a width-based estimate with the box position.
6. The app displays the target box and provides directional/proximity feedback through speech and vibration when browser/device APIs allow it.

A remote detector is optional. Its request and selection behavior is described separately below because a returned server result is only used as a fallback, but configured server requests are not currently conditional on a local miss.

## Local Target Resolution and Detection

### Prompt resolution

The prompt pack consists of metadata and precomputed YOLOE prompt embeddings under pulse-point/public/prompts/. Build scripts are under scripts/yoloe/. Targets can resolve through pack names and aliases, COCO aliases, and fuzzy matching. Prompt packs are finite: this is not unrestricted natural-language understanding. If a typed phrase does not resolve to a prompt-pack item, the local inference function receives no prompt set and returns no detections for that target.

The target item may also include a reference physical width used by the distance fallback. The width is an assumed class/item size, not a measurement of the specific object in the camera scene.

### Detector selection

The models are loaded by pulse-point/src/detection/engine.js using ONNX Runtime Web:

- With a usable WebGPU adapter, the app loads YOLOE-11s and runs it with the prompt embeddings. If that model/session fails to load, the startup path tries YOLO11n. YOLO11n still recognizes only its 80 COCO classes.
- Without usable WebGPU, the initial local model is YOLO11n on WASM/CPU. For a target that maps to a COCO class, YOLO11n is the normal local detector, not a fallback after YOLOE fails.
- For a non-COCO target represented in the prompt pack on a CPU device, YOLOE is loaded on demand using WASM. While it is loading, local inference can return no boxes.
- YOLOE can fall back from WebGPU execution to WASM within the ONNX Runtime session setup.

Therefore, the statement "YOLO11n only runs when the target is unidentifiable" would not describe the current code. It handles COCO-mappable targets on CPU as a standard local path. It is also attempted if the WebGPU YOLOE load fails, but its class coverage remains limited to COCO.

Input frames are letterboxed into 640 by 640 pixels and converted to the model tensor format. Postprocessing uses a 0.25 confidence threshold and non-maximum suppression at 0.45 IoU. The scanning loop adapts its requested inference cadence within configured limits; those values are scheduling bounds, not measured FPS guarantees or benchmark results.

## Optional Remote Vision Service and Data Flow

The web client only enables this path when VITE_SERVER_URL is configured. It checks the service health endpoint and, while a target is set and the scanner is running, attempts a remote request on a roughly 2.5-second cooldown when the last health state says the service is available.

The request captures the current camera frame as a JPEG and sends that image together with the target text to the configured server. The request is not gated on a local detection miss. The merge logic prefers a current local target match; if no local match exists, it can use a returned server result. Thus, "the server result is a fallback" is accurate, while "the server is contacted only when local detection cannot identify the object" is not accurate for the current implementation.

With VITE_SERVER_URL unset, this optional vision-server path is disabled and object inference runs in the browser. Anyone enabling a server URL should understand that camera frames and target text leave the device during scanning, including periods when local detection is succeeding. Do not put secrets in VITE_* variables because Vite embeds them in browser-delivered code.

The Python service exposes POST /detect and POST /v1/detect. For a nonempty target it first attempts LocateAnything-3B. If that path is unavailable or produces no result, the service falls back to an ImageNet classification model mapped to an indoor-object ontology, with approximate Grad-CAM localization. The fallback is not a trained/evaluated bounding-box detector. The LocateAnything adapter currently assigns a fixed 0.90 score; that value is not calibrated confidence. All service responses are marked experimental/unvalidated and force assistiveReady and proof fields to false. The shared response metadata retains a legacy localizationMethod value of approximate/Grad-CAM, which should not be treated as an accurate description of every primary-model result.

The scanner does not call the separate Vercel /api/ai OpenRouter proxy. That endpoint is legacy infrastructure and is not a detector fallback.

## Tracking, Depth, and Guidance

### Tracking

The BoxTracker associates the requested object across frames and smooths its box/confidence state to reduce visual and guidance jitter. It can predict a track briefly between detections, but this does not establish identity, guarantee continuity, or make a stale box safe to act on.

### Depth and distance

Depth Anything V2 Metric-Indoor-Small runs in a worker with its own ONNX Runtime session. The worker uses the checked-in FP16 model with WebGPU where available and the uint8 model with WASM otherwise. Depth sampling is performed within the target box. The guidance code prefers a recent depth reading when available and otherwise uses an object-width estimate.

Depth output and width-based estimates are not calibrated for every camera, device, object instance, lighting condition, or environment. A numeric-looking distance should be treated as an experimental estimate, not a verified measurement. The testing-only protocol in testing/pulse-point-benchmark/ describes manual measurement collection without adding production telemetry.

### Direction and proximity

The guidance module maps a detected box's position relative to the frame into directional labels and computes proximity cues. Its current reach condition can be triggered by a depth estimate below 0.55 m or by a target box occupying more than 20 percent of the frame area. These are software thresholds, not user-tested safety distances or a confirmation of arm's reach.

Haptics and speech are browser/device-dependent. In particular, web vibration support is not reliable across all iPhone browsers. The app does not provide obstacle detection, a traversable route, or a way to verify that the path between the camera and target is clear.

## On-Screen Visuals

The canvas renders detection boxes, a target label, coordinates, a decorative 7 by 7 grid, anchor-style lines, and a radial gradient behind a detected target. The grid and gradient are interface decoration; they are not a model feature map, attention map, Grad-CAM output, or explanation of why the detector selected a box. Do not present them as model interpretability.

## Assets, Caching, and Deployment

The web app is a React/Vite static build configured at the repository root for Vercel. ONNX models, prompt assets, and the service worker are under pulse-point/public/. The service worker caches selected assets after they have been fetched successfully. Initial model/runtime downloads, camera access, app navigation, browser APIs, and device support can still require a network or fail; caching is not an offline guarantee.

The app's package scripts are npm run dev, npm test, and npm run build from pulse-point/. Model export and prompt-pack construction scripts are in scripts/yoloe/; depth export tooling is in scripts/depth/.

## Tests and Evaluation

Unit tests cover target resolution, detector prompt preparation, tracking, scanner state, guidance, camera/settings/voice helpers, and core contract behavior. Run the web checks from pulse-point/:

- npm test
- npm run build

The testing/pulse-point-benchmark/ folder is explicitly separate from the production app. It contains a repeatable manual protocol, blank CSV template, and standard-library summary script for exploratory outcomes such as correct locks, false locks, time to lock, FPS when available, and distance error when numeric estimates are exposed. Its sample sizes are a screening baseline, not a statistically powered study. Human-participant or fair research may require advance approvals; follow the relevant rules before recruiting or collecting data.

## Known Evidence Gaps

- No representative accuracy, false-positive, distance-error, latency, or battery benchmark across a declared set of devices and scenes.
- No evidence that model confidence is calibrated.
- No validation of the haptic vocabulary or speech phrasing with target users.
- No validation for safe approach, obstacle avoidance, navigation, or reach behavior.
- The optional remote model's availability, output quality, fixed score, and metadata require separate evaluation; it is not a substitute for local evidence.

Future competition materials should distinguish implementation facts, measured results, assumptions, and future work. Do not reuse historical performance or architecture claims unless independently remeasured and verified against this source tree.
