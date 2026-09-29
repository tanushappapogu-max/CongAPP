# Pulse Point

**Live demo:** https://pulse-point-steel.vercel.app

[![Tests](https://github.com/ketchup235/Pulse-Point/actions/workflows/test.yml/badge.svg)](https://github.com/ketchup235/Pulse-Point/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Pulse Point is a browser-based prototype exploring a haptic-first way to find a named object. The supported app is `pulse-point/`, built with React, Vite, ONNX Runtime Web, and camera input. It is not a production assistive device and has not been validated for independent mobility, obstacle avoidance, or safety-critical use.

---

## How It Works

Voice or text names a target, which is resolved against the checked-in prompt pack and aliases. YOLOE runs locally with prompt embeddings for items represented by that pack. When usable WebGPU is unavailable, YOLO11n normally handles targets that map to a COCO class because it is lighter on CPU; YOLOE is loaded on demand for other prompt-pack targets. If YOLOE fails to load on the WebGPU path, YOLO11n is attempted as a backup but can only detect its COCO classes. A tracker stabilizes boxes, Depth Anything V2 estimates scene depth in a worker, and the guidance layer produces direction, speech, and vibration feedback. These outputs are a prototype, not safety assurances.

With no `VITE_SERVER_URL` configured, the scanner's object detection is local. If that optional server URL is configured and healthy, the app sends a JPEG camera frame and target to it about every 2.5 seconds while scanning, even if local detection is working. A local match takes precedence; a server result is used only when there is no local match. So remote *use* is a fallback, but remote *requests* are not currently gated on local failure. Configure it only when you accept that camera frames and target text leave the browser. See [the server notes](server/README.md).

The legacy Vercel proxy at `/api/ai` (OpenRouter) remains in the repository but is not called by the web scanner.

```
Voice/text target -> prompt-pack resolver -> local model selection -> box tracker
                                                     │
                                              Distance model
                                                     │
                                           Guidance compute
                                            ┌──────────────┐
                                         Haptics        Speech
```

---

## Built With

![React](https://img.shields.io/badge/React-19-61dafb?logo=react&logoColor=white)
![Vite](https://img.shields.io/badge/Vite-7-646cff?logo=vite&logoColor=white)
![ONNX Runtime Web](https://img.shields.io/badge/ONNX_Runtime_Web-WebGPU%2BWASM-4c8bf5)
![Expo](https://img.shields.io/badge/Expo-54-000020?logo=expo&logoColor=white)
![Vercel](https://img.shields.io/badge/Vercel-serverless-000000?logo=vercel&logoColor=white)
![YOLO11n](https://img.shields.io/badge/YOLO11n-COCO-00bfff)

---

## Projects

### Web Prototype (`pulse-point`)

The public Vercel app deploys from `pulse-point` and runs in the browser with live camera object detection.

```bash
cd pulse-point
npm install
npm run dev        # development server
npm test           # run unit tests
npm run build      # production build
```

### Mobile App (`pulse-point-mobile`) — *Exploratory / Degraded Prototype*
 
> [!NOTE]
> `pulse-point-mobile/` is an **exploratory / degraded prototype** for testing mobile UX concepts and simulated guidance. It is **not** the active production client (which is `pulse-point/`). Contributors and agents should **not** waste time applying web updates or core improvements to this folder unless specifically requested.

The mobile prototype lives in `pulse-point-mobile`.

```bash
cd pulse-point-mobile
npm install --cache .npm-cache
npm start
```

Scan the Expo QR code with Expo Go on your phone. The app uses the phone camera, haptics, and motion sensors for the object-finding flow.

---

## Current Prototype

- Accepts a typed target and browser-supported voice input.
- Resolves supported targets against the bundled prompt pack and aliases.
- Runs local YOLOE/YOLO11n detection according to device capability and target class.
- Tracks a target box, estimates depth/distance, and emits prototype direction/proximity cues.
- Shows a camera overlay with the selected target; its grid and gradient are visual decoration, not model explanations.
- Can optionally query an experimental remote service when configured; this uploads camera frames while scanning, even when local detection succeeds.

It does not build a room map, detect obstacles, plan a route, or establish that moving/reaching is safe.

---

## Current Status

### Web

The Vercel app requests camera permission, downloads model and prompt assets, and runs the detector locally in the browser by default. WebGPU selects YOLOE; on CPU, YOLO11n handles COCO-mappable targets while YOLOE can load on demand for other supported prompt-pack targets. The app draws target boxes, estimates direction and distance, and triggers phone vibration where supported. A versioned service-worker cache can speed up repeat loads after assets download successfully.

The service worker is not an offline guarantee: camera permission, browser APIs, first model/runtime downloads, app-shell navigation, and device support can still require network access or fail. If `VITE_SERVER_URL` is set, camera frames and target text are also sent to that service on a periodic schedule during scanning. iPhone browsers do not expose reliable vibration APIs, and websites cannot access iPhone LiDAR room meshes directly.

The optional Python server detector is experimental and unvalidated. `/detect`
and `/v1/detect` share a handler. The service tries LocateAnything-3B for a
supplied target, then falls back to an indoor-object classifier with approximate
Grad-CAM localization if the grounding model is unavailable or returns no
result. The fallback uses raw, uncalibrated ImageNet probabilities and neither
path is evaluated as assistive sensing. Responses carry explicit
`assistiveReady: false` and `proof: false` metadata. See
[`server/README.md`](server/README.md) for the bounded request and response
contract.

### Mobile

`pulse-point-mobile/` is an exploratory/degraded UX prototype, not the supported app and not a source of production model behavior. Its implementation is intentionally not covered by the web-app architecture description above.

---

## Deploying to Vercel

Import this repo into Vercel. The included `vercel.json` builds the `pulse-point` site and publishes `pulse-point/dist`.

Keep the repository root as the root directory. Vercel will run:

```bash
cd pulse-point && npm install
cd pulse-point && npm run build
```

---

## Environment Variables

Set these in Vercel Project Settings → Environment Variables.

| Variable | Description |
|---|---|
| `VITE_SERVER_URL` | Optional base URL for the experimental Python vision service. When configured, scanning uploads a JPEG frame and target about every 2.5 seconds while the server is healthy; it is not limited to local-detection misses. Vite exposes `VITE_*` values to the browser, so this must not contain a secret. |
| `OPENROUTER_API_KEY` | Server-side key for the legacy `/api/ai` proxy. The web scanner does not call that proxy. Never prefix a secret with `VITE_`. |
| `ALLOWED_ORIGIN` | Optional allowed origin for the legacy `/api/ai` proxy (defaults to `https://pulse-point-steel.vercel.app`). |

Old prototype builds used `VITE_GEMINI_API_KEY` in browser code. Do not set it in production; the current scanner has no Gemini/OpenRouter fallback.

The `/api/ai` proxy already enforces an approved model list, output-token cap,
strict origin checks, per-IP in-memory rate limiting, and a 30-second Vercel
function duration. It is optional cloud-assist infrastructure and does not
validate the experimental Python detector.

`vercel dev` can run the legacy serverless functions alongside the app. `OPENROUTER_API_KEY` only enables the separate `/api/ai` endpoint; it does not enable the vision service. Set `VITE_SERVER_URL` only when intentionally testing the remote detector and its camera-frame upload behavior.

---

## Acknowledgments

Pulse Point uses [YOLOE](https://github.com/THU-MIG/yoloe) with a generated prompt pack, [Ultralytics YOLO11n](https://github.com/ultralytics/ultralytics) for CPU/COCO detection and a load-failure backup, [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) for scene depth, and [ONNX Runtime Web](https://onnxruntime.ai/docs/get-started/with-javascript.html). Export/build scripts are under `scripts/`. Review upstream model and runtime terms before redistributing or deploying modified weights. Pulse Point application code is licensed under the MIT License; see [LICENSE](LICENSE).
