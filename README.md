# 6ixth Sense

[![Tests](https://github.com/tanushappapogu-max/CongAPP/actions/workflows/test.yml/badge.svg)](https://github.com/tanushappapogu-max/CongAPP/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Browser-based object finding for blind and low-vision users.** Say or type what you're looking for ("my glasses", "keys", "a cup") and sweep the phone around the room. 6ixth Sense guides you to it with speech and vibration: which way to turn, how far away the object is, and when it's within reach.

**Live demo:** https://6ixthsense.vercel.app

> [!IMPORTANT]
> 6ixth Sense is a research prototype. It has not been validated for independent mobility, obstacle avoidance, or any safety-critical use. A "reach" cue means the object is probably close, not that it is safe to move toward or touch.

## Highlights

- **Open-vocabulary detection.** One YOLOE model covers 113 items: 33 household objects plus the 80 COCO classes. Each item has a precomputed prompt vector, so no embedding API is called at runtime.
- **Confusion-aware matching.** Every target is scored against its look-alikes (eyeglasses vs. sunglasses vs. glasses case) before a detection counts.
- **Hybrid distance.** Metric depth from Depth Anything V2 is combined with box width and lens geometry, so distance updates every frame.
- **Server-first, offline-capable.** A GPU vision server runs inference at about 79 ms per frame. If it's unreachable, the app falls back to ONNX Runtime Web on the device without user action.
- **Built around iPhone Safari's limits.** The memory, threading and WebGPU constraints were profiled one by one and designed around (see [iPhone notes](#iphone-notes)).

## Contents

- [How it works](#how-it-works)
- [Results](#results)
- [Getting started](#getting-started)
- [Repository layout](#repository-layout)
- [Detection](#detection)
- [Distance estimation](#distance-estimation)
- [Vision server](#vision-server)
- [iPhone notes](#iphone-notes)
- [Configuration](#configuration)
- [Development](#development)
- [Deployment](#deployment)
- [Limitations](#limitations)
- [Team](#team)
- [Credits and license](#credits-and-license)

## How it works

```text
"find my glasses"
      │
      ▼
Target lookup ──── prompt pack: target vector + look-alikes (eyeglasses vs sunglasses)
      │
      ▼
Detection ──────── YOLOE (open vocabulary), YOLO11n as the COCO backup
      │              on the vision server when available, otherwise on the phone
      ▼
Tracking ───────── smooths boxes between frames
      │
      ▼
Distance ───────── Depth Anything V2 (meters, every ~1–4 s)
      │              box width × lens field of view in between
      ▼
Guidance ───────── turn left/right · tilt up/down · closer · reach
      │
      ▼
Speech + haptics
```

The web app (`pulse-point/`) owns the camera, tracking, guidance, speech and haptics. Detection and depth run on a GPU-backed [vision server](#vision-server) when it's available, and otherwise on the device with ONNX Runtime Web.

| Layer | Technology |
|---|---|
| Client | React 19, Vite 7, ONNX Runtime Web (WASM / WebGPU) |
| Models | YOLOE-11s (open vocabulary), YOLO11n (COCO backup), Depth Anything V2 Metric-Indoor-Small |
| Vision server | FastAPI, ONNX Runtime (CUDA / CPU), deployed on Modal |
| Testing | Vitest (web and shared core), `unittest` (server contract), GitHub Actions |

## Results

Measurements from internal development testing.

### Detection: prompt tuning

All 113 prompt-pack items were run through YOLOE on **904 real photos**. For some small objects, YOLOE barely responds to the literal item name, so those items use a blended vector of the phrases it does respond to (see [Tuned vectors](#yoloe-and-the-prompt-pack)). Three of the tuned items, showing the share of photos in which each item was found:

| Item | Plain name | Tuned vector | Prompt phrases used |
|---|---:|---:|---|
| Wristwatch | 12% | **75%** | "clock", "watch" |
| Eyeglasses | 12% | **62%** | "specs", "glasses", "my glasses" |
| Keys | 0% | **50%** | "keychain" |

16 of the 33 household items use tuned vectors (`prompts` in [`scripts/yoloe/pack_items.json`](scripts/yoloe/pack_items.json)).

### Latency and memory

| Measurement | Result | Conditions |
|---|---|---|
| End-to-end latency, vision server | ~79 ms/frame (~8 fps) | Server run locally; capture, upload, inference and response |
| Upload size per frame | ~35 KB | 640×640 letterboxed JPEG |
| Browser memory, vision server mode | ~10–20 MB | No models downloaded to the phone |
| On-device YOLOE, CPU | 178 ms/frame single-threaded; 51 ms/frame multi-threaded (3.5×) | iPhone Safari. Multi-threading is disabled in production (see [iPhone notes](#iphone-notes)). |
| Cold start, Modal T4 | 10–30 s | First request after ~5 idle minutes |

### Depth estimation

| Configuration | Peak memory | Agreement with full frame |
|---|---:|---|
| Full wide frame | ~550 MB | Reference |
| 518 px centred square (default) | ~280 MB | Within ~2% on indoor photos, after a ×1/1.03 correction |
| 392 px centred square (`?depthsize=392`) | ~200 MB | Slightly noisier, after a ×1/1.15 correction |

## Getting started

**Requirements:** Node 20+. Python 3.11+ is needed only to run the vision server or model scripts.

```bash
git clone https://github.com/tanushappapogu-max/CongAPP.git
cd CongAPP
./scripts/setup-git-hooks.sh     # commit-msg and pre-push hooks

cd pulse-point
npm install
npm run dev                      # serves on 0.0.0.0 so a phone on the same network can reach it
```

Camera access needs a secure context. On the desktop, `localhost` works. On a phone, open the dev server over HTTPS (for example with `cloudflared tunnel --url http://localhost:5173`) or use a Vercel preview.

By default the app uses the hosted vision server. Add `?server=off` to force on-device detection, or `?server=http://localhost:8788` to use a local server.

## Repository layout

```text
pulse-point/              Web app (the supported client)
  public/                 ONNX models, prompt pack, service worker, manifest
  src/detection/          Detector selection, server client, prompt pack, depth, tracker
  src/guidance/           Turn/tilt/reach logic, speech, haptics
  src/lib/                Camera, URL flags, settings, voice input
  src/scanner/            Scan session state
  src/ui/                 Overlays and settings sheet
packages/pulsepoint-core/ Shared target, tracking and guidance logic
server/vision/            Vision server (YOLOE + Depth Anything over HTTP)
scripts/yoloe/            YOLOE export and prompt pack build
scripts/depth/            Depth Anything export
server/                   Legacy experimental detector service (see Legacy components)
api/                      Legacy OpenRouter proxy, unused by the app
pulse-point-mobile/       Expo prototype with simulated sensing, not maintained
```

Key modules, relative to `pulse-point/`:

| File | Responsibility |
|---|---|
| `src/detection/engine.js` | Picks the detector (server, YOLOE, YOLO11n), captures frames, decodes boxes |
| `src/detection/remote.js` | Vision server client and automatic fallback |
| `src/detection/prompts.js` | Prompt pack loading, alias and typo matching, look-alike vectors |
| `src/detection/depth.js`, `depth.worker.js`, `depthSample.js` | Depth on the server or in a worker, square crop, box sampling |
| `src/detection/distance.js`, `src/guidance/compute.js` | Width-based distance and the turn/tilt/reach decisions |
| `src/lib/camera.js` | Wide-lens selection and its field of view |
| `src/lib/flags.js` | URL switches and Safari/iOS defaults |

## Detection

### YOLOE and the prompt pack

YOLOE is an open-vocabulary detector. Instead of a fixed class list, it takes a vector describing what to look for. The export script (`scripts/yoloe/export_promptable.py`) keeps that vector as a live ONNX input, so a single model covers every item.

The prompt pack (`pulse-point/public/prompts/pack.{json,bin}`, 226 KB) holds precomputed 512-dimensional vectors for 113 items: 33 household objects (glasses, keys, wallet, AirPods, pill bottle, white cane, door handle, and so on) plus the 80 COCO classes. It ships with the app, loads once, and is searched locally.

Plain class names alone work poorly for many items. Three mechanisms make up for that:

- **Look-alikes.** Each item lists confusable neighbours. A box counts as "eyeglasses" only if it scores higher for eyeglasses than for sunglasses or a glasses case.
- **Tuned vectors.** Where YOLOE barely responds to an item's name, the item's vector is the renormalized mean of the phrases it does respond to. The phrases were chosen by evaluating on real photos (see [Results](#detection-prompt-tuning)).
- **Aliases and typos.** "my glasses", "airpods", "medicine", "sofa" and near-misses such as "keyz" resolve to the right item (`src/detection/prompts.js`).

**YOLO11n** is kept as a lighter backup that only knows the 80 COCO classes. It's used for COCO targets on devices without WebGPU, and whenever YOLOE fails to load.

### Adding or changing an item

1. Edit `scripts/yoloe/pack_items.json`. Each item has a `name`, `aliases`, `widthCm` (used for distance) and `negatives` (look-alikes, which must also be items). Optional fields: `prompts` (phrases to blend instead of the name) and `coco` (the class YOLO11n should search for).
2. Rebuild the pack:
   ```bash
   scripts/yoloe/.venv/bin/python scripts/yoloe/build_prompt_pack.py
   ```
   This writes `pack.json` and `pack.bin` into `pulse-point/public/prompts/`.
3. If YOLOE is ever re-exported, rebuild the pack as well. Vectors only match the checkpoint they were built from (`pack.json` records its SHA-256).

## Distance estimation

Distance comes from two estimators that cover for each other.

**Depth Anything V2 (Metric-Indoor-Small)** gives a reading in meters. It runs at 518 px on the largest centred square of the frame instead of the full wide frame. The model's attention compares every patch with every other patch, so memory grows quickly with input size. The square crop roughly halves peak memory and stays within about 2% of the full-frame reading (see [Results](#depth-estimation)).

**Box width and lens geometry** fill the gaps between depth readings, so guidance updates every frame. Apparent width is inversely proportional to distance: a box twice as wide means the object is half as far away. The calculation uses the field of view of the lens actually in use and the frame's long side, so it holds in portrait and landscape.

How the two are combined:

- **Depth only runs within the last ~2 m.** Farther out, the user is still turning and walking, and the width estimate is good enough. In on-device mode, the depth model isn't loaded until then.
- **Small items use the nearer estimate.** Items under 15 cm (AirPods, keys, glasses) cover only a few depth patches until they're very close. The depth reading then picks up the surface behind the item and reports it as too far away. Without this rule, the "reach" cue came far too late.
- **The widest lens is used.** The camera opens the widest back lens available (the 0.5× ultra-wide on iPhone, about 103° versus about 64° for the main camera), so users find the target with less sweeping. See `lensFovDeg` in `src/lib/camera.js`.

## Vision server

Running YOLOE and Depth Anything in a phone browser is heavy enough that iPhone Safari repeatedly killed the tab (see [iPhone notes](#iphone-notes)). The vision server in `server/vision/` runs both models instead.

- The phone sends one JPEG per frame (~35 KB) and gets boxes back, plus a distance about once a second. No models are downloaded to the phone.
- The server loads the same model files and prompt pack as the app, so its detections match on-device detection.
- If the server is unreachable or a request fails, the app switches to on-device detection without user action.

| Endpoint | Input | Output |
|---|---|---|
| `GET /health` | – | `{"ok": true, "device": "CUDAExecutionProvider" \| "CPUExecutionProvider"}` |
| `POST /v1/detect` | `file`: 640×640 letterboxed JPEG; `target`: prompt pack item name | `{"detections": [{class, score, bbox}], "ms"}` |
| `POST /v1/depth` | `file`: centred square crop JPEG; `box`: `x,y,w,h` as crop fractions | `{"meters", "ms"}` |

**Hardening:** CORS is limited to the app's origins (`server/contract.py`), requests are rate-limited per IP, and image size is capped.

> [!NOTE]
> While a vision server is in use, camera frames and the target name leave the phone during scanning. The server doesn't store them.

### Hosting options

| Option | Cost | Trade-offs |
|---|---|---|
| Modal, T4 GPU (`server/vision/modal_vision.py`) | Covered by Modal's $30/month free credit | Fastest. Scales to zero after ~5 idle minutes, and the first request after that takes 10–30 s. The app sends a wake-up request when the page opens. |
| Local machine + Cloudflare Tunnel | Free, no account | Good for phone testing. Only available while your machine is on, and the URL changes on every run. |
| Hugging Face Spaces, free CPU | Free | Stable URL, but no GPU (so depth is slower), and it sleeps after ~2 days without traffic. |

### Running it yourself

**Locally:**

```bash
python3 -m venv .venv && .venv/bin/pip install -r server/vision/requirements.txt
cd server/vision && ../../.venv/bin/uvicorn vision_app:app --port 8788
```

Then open the app with `?server=http://localhost:8788`.

**On a phone, without deploying:**

```bash
brew install cloudflared
cloudflared tunnel --url http://localhost:8788
```

Then open `https://6ixthsense.vercel.app/?server=<trycloudflare.com URL>` on the phone.

**On Modal:**

```bash
pip install modal
python3 -m modal setup                          # one-time login
modal deploy server/vision/modal_vision.py      # prints the server URL
```

Set the printed URL as `VITE_VISION_URL` in Vercel and redeploy.

## iPhone notes

The vision server exists because of what profiling iPhone Safari showed. The team used Web Inspector memory recordings together with the [URL switches](#url-switches) below:

| Finding | Consequence |
|---|---|
| **WebGPU leaked memory** at ~150 MB/s while scanning, and the tab was killed within ~20 s. | On-device inference uses the CPU on every iOS browser (all run on WebKit) and on desktop Safari. `?gpu=1` turns WebGPU back on for testing. |
| **CPU detection alone was stable; detection plus depth was not.** The depth model is 27 MB on disk but peaked at ~550 MB on a wide frame. | Depth runs on a centred square crop, which halves peak memory. |
| **Multi-threaded inference** (via COOP/COEP) made detection 3.5× faster (178 → 51 ms/frame) but crashed Safari whenever depth also ran. | The site is deliberately not cross-origin isolated. |
| **Recreating the depth worker after each reading** to release memory also crashed Safari. | One depth worker stays alive for the whole session. |

The on-device path remains as the offline fallback.

## Configuration

### Environment variables

Set these in Vercel, or in `pulse-point/.env.local` for local builds.

> [!WARNING]
> Vite exposes every `VITE_*` value to the browser. Never put a secret in one.

| Variable | Purpose |
|---|---|
| `VITE_VISION_URL` | Vision server URL. Defaults to the Modal deployment, `https://tanush-appapogu--pulse-point-vision-fast.us-east.modal.direct`. |
| `VITE_SERVER_URL` | Optional legacy LocateAnything server. When it's healthy, it's queried about every 2.5 s while scanning. |
| `OPENROUTER_API_KEY` | Server-side key for the legacy `/api/ai` proxy only. The app doesn't call it. |
| `ALLOWED_ORIGIN` | Origin allowed by the legacy `/api/ai` proxy. Defaults to `https://6ixthsense.vercel.app`. |

### URL switches

These are for testing and diagnosis, and they can be combined, e.g. `https://6ixthsense.vercel.app/?debug=1&nodepth=1`.

| Switch | Effect |
|---|---|
| `?debug=1` | On-screen readout: server status, active detector and backend, depth backend, thread count, frame time, best target score, memory |
| `?server=URL` | Use this vision server. `?server=off` forces on-device detection. |
| `?nodepth=1` | Disable the depth model |
| `?cpu=1`, `?gpu=1` | Force CPU or WebGPU for on-device inference |
| `?threads=N` | Force the CPU backend's thread count |
| `?depthsize=392` | Smaller depth crop: ~200 MB peak, slightly noisier, corrected ×1/1.15 |
| `?nodetect=1` | Camera and overlay only; no detector is loaded |
| `?captureonly=1` | Load the detector and fill its input each frame, but never run it |

## Development

```bash
cd pulse-point
npm test              # unit tests (Vitest)
npm run test:watch
npm run build         # production build into pulse-point/dist
npm run preview       # serve the production build
```

**CI** (`.github/workflows/test.yml`) runs on every push to `main` and on every pull request. It runs the web and shared-core tests, a production build, a syntax check of `api/ai.js`, and the server contract tests (`python -m unittest discover -s server -p "test_contract.py"`).

**Model exports**, needed only when weights change. Each script directory has its own `requirements.txt`.

| Script | Output |
|---|---|
| `scripts/yoloe/export_promptable.py` | YOLOE-11s ONNX with the prompt vector as an input |
| `scripts/yoloe/build_prompt_pack.py` | `pack.json` and `pack.bin` from `pack_items.json` |
| `scripts/depth/export_depth.py` | Depth Anything V2 Metric-Indoor-Small, fp16 (WebGPU) and uint8 (CPU) |

**Contributing:** branching, commit format and the review process are in [CONTRIBUTING.md](CONTRIBUTING.md). The hooks installed by `scripts/setup-git-hooks.sh` check commit messages; see [COMMIT_POLICY.md](COMMIT_POLICY.md).

**Further reading:**

- [ARCHITECTURE.md](ARCHITECTURE.md): data flow and module boundaries
- [TECHNICAL_OVERVIEW.md](TECHNICAL_OVERVIEW.md): implementation detail
- [CHANGELOG.md](CHANGELOG.md): release history

## Deployment

The web app deploys to **Vercel** from the repository root. `vercel.json` installs and builds `pulse-point/` and publishes `pulse-point/dist`. It also sets long-lived caching on hashed assets and restricts camera and microphone permissions to the site's own origin.

The vision server deploys separately to **Modal** (see [Running it yourself](#running-it-yourself)).

### Legacy components

These remain in the repository but are not part of the main app path:

- **`server/`** (outside `server/vision/`): the earlier experimental detector, deployed with `server/modal_app.py`. It contains LocateAnything-3B, an ImageNet/Grad-CAM classifier and an unrelated tea-text classifier. It's used only when `VITE_SERVER_URL` is set. See [server/README.md](server/README.md).
- **`api/ai.js`:** an OpenRouter proxy that the app no longer calls.
- **`pulse-point-mobile/`:** an Expo prototype for mobile UX ideas, with simulated sensing. It is not the supported client and doesn't receive changes made to the web app.

## Limitations

- **No safety guarantees.** There is no obstacle detection, room mapping or route planning. Nothing in the app establishes that moving or reaching is safe.
- **Weak small items.** Headphones, sunglasses, chargers, door handles and smartwatches are still unreliable. Improving them will likely need vectors learned from labelled photos rather than text prompts.
- **Lens mismatch for depth.** Depth Anything was trained on normal-lens indoor images, so readings through the ultra-wide lens are somewhat less accurate.
- **Platform gaps.** iPhone browsers don't expose a reliable vibration API, and web pages can't access the iPhone's LiDAR.

## Team

| Name | GitHub |
|---|---|
| Tanush Appapogu | [@tanushappapogu-max](https://github.com/tanushappapogu-max) |
| Ahaan Nigam | [@AN00927](https://github.com/AN00927) |
| Tejas Karusala | [@tkarusala001](https://github.com/tkarusala001) |
| Rishabh Patel | [@rpatel-23](https://github.com/rpatel-23) |

## Credits and license

| Component | License / notes |
|---|---|
| [YOLOE](https://github.com/THU-MIG/yoloe) (YOLOE-11s via [Ultralytics](https://github.com/ultralytics/ultralytics)) and [Ultralytics YOLO11n](https://github.com/ultralytics/ultralytics) | AGPL-3.0. Because these weights are AGPL-3.0, the deployed app's source must remain public. |
| [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) Metric-Indoor-Small | Fine-tuned by its authors on Hypersim from the Apache-2.0 Small model. Its model card states no separate license. |
| [MobileCLIP](https://github.com/apple/ml-mobileclip) text encoder | Used offline only, to build the prompt pack |
| [ONNX Runtime Web](https://onnxruntime.ai/docs/get-started/with-javascript.html), [FastAPI](https://fastapi.tiangolo.com/) | Runtime and server frameworks |

The 6ixth Sense application code is released under the [MIT License](LICENSE). The project was previously named Pulse Point, which is why some directories and packages still use that name.
