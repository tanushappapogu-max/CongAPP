# Pulse Point

**Live demo:** https://pulse-point-steel.vercel.app

[![Tests](https://github.com/ketchup235/Pulse-Point/actions/workflows/test.yml/badge.svg)](https://github.com/ketchup235/Pulse-Point/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Pulse Point is a haptic-first prototype exploring an object-finding interface for blind and low-vision users. You say what you're looking for, the app can report detections and guidance through distinct vibration patterns — no spoken directions that compete with hearing. It is not a production assistive or safety-validated system.

---

## How It Works

The web app runs YOLO11n object detection in-browser with ONNX Runtime Web and its WASM execution provider (no server round-trip for supported COCO classes). Voice input names a target; the COCO label resolver maps natural language ("phone", "TV", "armchair", "loveseat") to the model's 80-class vocabulary through a curated synonym table. The "Trained Objects" list in the UI is scoped to exactly what that resolver can find — every entry there is guaranteed to map to a real, detectable class, so a tap never starts a scan that can't succeed. A bounding-box tracker with EMA-smoothed velocity keeps the lock stable across frames. A pinhole-camera distance model converts bbox size to meters. The guidance engine divides the frame into a center sweet spot and emits one of seven directional haptic signals until the target is centered and within reach.

A Vercel serverless proxy (`/api/ai`) is already deployed and working for a more capable cloud vision model (Gemini via OpenRouter) that could look up objects outside the local 80-class vocabulary, keeping the API key off the client. It is not yet called from the live detection loop — today it's available infrastructure, not an active fallback — so object-finding is currently bounded to the local vocabulary above.

```
Voice input ──► COCO resolver ──► YOLO loop ──► Box tracker
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
![ONNX Runtime Web](https://img.shields.io/badge/ONNX_Runtime_Web-WASM-4c8bf5)
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

## Prototype Goals

- Keep guidance tactile so it doesn't compete with hearing
- Run browser-based object detection for requests like "find my mouse"
- Show what LiDAR/camera spatial awareness could look like through a 3D room-map concept
- Walk through the full flow: request, scan, target lock, orientation, walking, and close-range handoff
- Visualize a 3×3 haptic matrix ring vocabulary for direction, proximity, and confirmation signals
- Provide a clean foundation for a TSA concept demo, pitch deck, or future hardware prototype

---

## Current Status

### Web

The Vercel app works in the browser. It requests camera permission, loads `/net.onnx`, runs inference through `/ort-wasm-simd.wasm`, draws boxes around detected objects, locks onto the requested target, estimates direction and distance from the camera frame, and triggers phone vibration where supported. A versioned service-worker cache keeps those immutable assets for faster repeat loads after they have downloaded successfully.

The service worker is not an offline guarantee: camera permission, browser APIs, the first model/runtime download, app-shell navigation, and device support can still require network access or fail. iPhone browsers don't expose reliable vibration APIs and websites can't access iPhone LiDAR room meshes directly, so true haptic guidance and LiDAR mapping belong in the native app.

The optional Python server detector is experimental and unvalidated. `/detect`
remains the mobile-compatible route and `/v1/detect` is its versioned alias.
It uses raw, uncalibrated ImageNet probabilities and approximate Grad-CAM
localization rather than a trained/evaluated bounding-box detector. Responses
carry explicit `assistiveReady: false` and `proof: false` metadata, so this
path must not be presented as validated assistive sensing. See
[`server/README.md`](server/README.md) for the bounded request and response
contract.

### Mobile

The Expo app opens the camera, reads compass heading, triggers haptics, and runs a target-finding state machine with spatial guidance on screen. The configured Python server detector is experimental and unvalidated. If its URL is missing, health checks fail, or a request errors, the app reports `UNAVAILABLE`/error and does not silently simulate detection. Simulation is available only through explicit opt-in (`EXPO_PUBLIC_PULSEPOINT_ALLOW_SIMULATION=true` or `extra.pulsepointAllowSimulation: true`), is labeled `DEMO ONLY`, and cannot produce a reach signal. This mobile flow must not be presented as production assistive sensing.

Real LiDAR mesh capture and live object recognition require a native build with iOS ARKit/CoreML or Android ARCore/ML Kit. The app is structured so those pieces can be swapped in later.

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
| `OPENROUTER_API_KEY` | Server-side key used by `api/ai.js`. **Do not** prefix with `VITE_` — Vite inlines those into the client bundle and exposes them publicly. |
| `ALLOWED_ORIGIN` | Optional. Restricts CORS to a specific origin (defaults to `https://pulse-point-steel.vercel.app`). |

Old prototype builds used `VITE_GEMINI_API_KEY` shipped to the browser. Remove that variable from production. The proxy at `/api/ai` is the supported path.

The `/api/ai` proxy already enforces an approved model list, output-token cap,
strict origin checks, per-IP in-memory rate limiting, and a 30-second Vercel
function duration. It is optional cloud-assist infrastructure and does not
validate the experimental Python detector.

For local development, `vercel dev` is recommended and runs the serverless function alongside Vite. Alternatively, set `VITE_GEMINI_API_KEY` in `pulse-point/.env.local` — the client falls back to direct OpenRouter calls only when the proxy returns 404. This is a dev-only convenience and must never be set in production.

---

## Acknowledgments

Pulse Point uses [Ultralytics YOLO11n](https://github.com/ultralytics/ultralytics) pretrained weights (Apache 2.0 license) for 80-class COCO object detection. The web deployment uses the checked-in `pulse-point/public/net.onnx` model with [ONNX Runtime Web](https://onnxruntime.ai/docs/get-started/with-javascript.html)'s WASM provider. All Pulse Point application code is original work by our team. Licensed under the MIT License — see [LICENSE](LICENSE).
