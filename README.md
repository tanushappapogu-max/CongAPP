# 6ixth Sense

[![Tests](https://github.com/tanushappapogu-max/CongAPP/actions/workflows/test.yml/badge.svg)](https://github.com/tanushappapogu-max/CongAPP/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Object finding for blind and low-vision users, in the browser. Say or type what you're looking for ("my glasses", "keys", "a cup"), point the phone around the room, and 6ixth Sense guides you to it with speech and vibration: which way to turn, how far away it is, and when it's within reach.

**Live:** https://6ixthsense.vercel.app

> [!IMPORTANT]
> This is a research prototype. It has not been validated for independent mobility, obstacle avoidance, or any safety-critical use. A "reach" cue means the object is probably close, not that it is safe to move toward or touch.

## Contents

- [How it works](#how-it-works)
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

The web app (`pulse-point/`) owns the camera, tracking, guidance, speech and haptics. Detection and depth run either on a small GPU-backed [vision server](#vision-server) or, if that's unreachable, on the device with ONNX Runtime Web.

**Stack:** React 19, Vite 7, ONNX Runtime Web, Vitest. Vision server: FastAPI + ONNX Runtime, deployed on Modal.

## Getting started

Requirements: Node 20+, and Python 3.11+ if you want to run the vision server.

```bash
git clone https://github.com/tanushappapogu-max/CongAPP.git
cd CongAPP
./scripts/setup-git-hooks.sh     # commit-msg and pre-push hooks

cd pulse-point
npm install
npm run dev                      # serves on 0.0.0.0 so a phone on the same network can reach it
```

Camera access needs a secure context. `localhost` works on the desktop; on a phone, open the dev server over HTTPS (for example through `cloudflared tunnel --url http://localhost:5173`) or use a Vercel preview.

By default the app talks to the hosted vision server. Add `?server=off` to the URL to force on-device detection, or `?server=http://localhost:8788` to use a local one.

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
server/                   Legacy experimental detector service (see below)
api/                      Legacy OpenRouter proxy, unused by the app
pulse-point-mobile/       Expo prototype with simulated sensing, not maintained
```

Files you'll touch most often:

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

YOLOE is an open-vocabulary detector. Instead of a fixed class list, it takes a vector describing what to look for. We export it to ONNX with that vector as a live input (`scripts/yoloe/export_promptable.py`), so one model covers every item.

The prompt pack (`pulse-point/public/prompts/pack.{json,bin}`, 226 KB) holds precomputed vectors for 113 items: 33 household objects (glasses, keys, wallet, AirPods, pill bottle, white cane, door handle, and so on) plus the 80 COCO classes. It is served with the app, loaded once, and searched locally. There are no embedding API calls at runtime.

Three things make it work better than plain class names:

- **Look-alikes.** Each item lists confusable neighbours. A box only counts as "eyeglasses" if it scores higher for eyeglasses than for sunglasses or a glasses case.
- **Tuned vectors.** We evaluated every item on roughly 900 real photos. YOLOE ignores the literal names of some small objects (it sees a wristwatch as "clock"), so those items use a blend of the phrases it does respond to:

  | Item | Found, plain name | Found, tuned vector |
  |---|---|---|
  | Wristwatch | 12% | 75% |
  | Eyeglasses | 12% | 62% |
  | Keys | 0% | 50% |

- **Aliases and typos.** "my glasses", "airpods", "medicine", "sofa" and near-misses like "keyz" resolve to the right item (`src/detection/prompts.js`).

YOLO11n is kept as a backup. It's lighter but only knows the 80 COCO classes, so it's used for COCO targets on devices without WebGPU and whenever YOLOE fails to load.

### Adding or changing an item

1. Edit `scripts/yoloe/pack_items.json`. Each item has a `name`, `aliases`, `widthCm` (used for distance), `negatives` (look-alikes, which must also be items) and optionally `prompts` (phrases to blend instead of the name) and `coco` (the class YOLO11n should search for).
2. Rebuild the pack: `scripts/yoloe/.venv/bin/python scripts/yoloe/build_prompt_pack.py`. The script writes `pack.json` and `pack.bin` into `pulse-point/public/prompts/`.
3. If YOLOE is ever re-exported, rebuild the pack too. The vectors only match the checkpoint they came from.

## Distance estimation

Distance comes from two sources that cover for each other.

**Depth Anything V2 (Metric-Indoor-Small)** gives a reading in meters. It runs on the largest centred square of the frame at 518 px rather than the full wide frame. On indoor photos that stays within about 2% of the full-frame result (after a fixed ×1/1.03 correction) and roughly halves peak memory, from ~550 MB to ~280 MB, because the model's attention compares every patch with every other.

**Box width and lens geometry** fill the gaps between depth readings, so guidance updates every frame. If the box is twice as wide, the object is half as far. The math uses the field of view of the lens actually in use and the frame's long side, so it holds in portrait and landscape.

How the two are combined:

- Depth only runs within the last ~2 m. Farther out, the user is still turning and walking and width is good enough. In on-device mode the depth model isn't loaded until then.
- For small items under 15 cm (AirPods, keys, glasses), the object covers only a few depth patches until it's very close, so the reading picks up the table behind it and reports too far. For these, guidance uses whichever estimate is nearer. Without this, "reach" arrived far too late.
- The camera opens the widest back lens available (the 0.5× ultra-wide on iPhone), so users find the target with less sweeping. That lens is ~103° wide versus ~64° for the main camera (`lensFovDeg` in `src/lib/camera.js`).

## Vision server

Running YOLOE and Depth Anything inside a phone browser is heavy enough that iPhone Safari repeatedly killed the tab (see [iPhone notes](#iphone-notes)). The vision server in `server/vision/` runs both models instead.

- The phone sends a ~35 KB JPEG per frame and gets boxes back, plus a distance about once a second. No models are downloaded to the phone.
- The server loads the same model files and prompt pack as the app, so results match on-device detection.
- If the server is unreachable or a request fails, the app switches to on-device detection without user action.
- Measured locally: ~79 ms per frame end to end (~8 fps), with browser memory around 10–20 MB instead of several hundred.

| Endpoint | Input | Output |
|---|---|---|
| `GET /health` | | `{"ok": true, "device": "CUDAExecutionProvider" \| "CPUExecutionProvider"}` |
| `POST /v1/detect` | `file`: 640×640 letterboxed JPEG; `target`: prompt pack item name | `{"detections": [{class, score, bbox}], "ms"}` |
| `POST /v1/depth` | `file`: centred square crop JPEG; `box`: `x,y,w,h` as crop fractions | `{"meters", "ms"}` |

CORS is limited to the app's origins (`server/contract.py`), requests are rate-limited per IP, and image size is capped.

> [!NOTE]
> With a vision server in use, camera frames and the target name leave the phone while scanning. The server doesn't store them.

### Hosting options

| Option | Cost | Trade-offs |
|---|---|---|
| Modal, T4 GPU (`server/vision/modal_vision.py`) | Covered by Modal's $30/month free credit | Fastest. Scales to zero after ~5 idle minutes; the first request after that takes 10–30 s. The app sends a wake-up request when the page opens. |
| Your machine + Cloudflare Tunnel | Free, no account | Good for phone testing. Only up while your machine is on, and the URL changes each run. |
| Hugging Face Spaces, free CPU | Free | Stable URL, but no GPU (slower depth) and it sleeps after ~2 days without traffic. |

### Running it yourself

Locally:

```bash
python3 -m venv .venv && .venv/bin/pip install -r server/vision/requirements.txt
cd server/vision && ../../.venv/bin/uvicorn vision_app:app --port 8788
```

Open the app with `?server=http://localhost:8788`.

On a phone, without deploying anything:

```bash
brew install cloudflared
cloudflared tunnel --url http://localhost:8788
```

Then open `https://6ixthsense.vercel.app/?server=<trycloudflare.com URL>` on the phone.

On Modal:

```bash
pip install modal
python3 -m modal setup                          # one-time login
modal deploy server/vision/modal_vision.py      # prints the server URL
```

Set the printed URL as `VITE_VISION_URL` in Vercel and redeploy.

## iPhone notes

The vision server exists because of what we found profiling iPhone Safari with Web Inspector memory recordings and the URL switches below:

1. **WebGPU leaked memory** at ~150 MB/s while scanning, and the tab was killed within ~20 s. On-device inference now uses the CPU on every iOS browser (they all run WebKit) and on desktop Safari. `?gpu=1` turns WebGPU back on for testing.
2. **Detection alone on the CPU was stable; detection plus depth was not.** The depth model is 27 MB on disk but peaked at ~550 MB on a wide frame. The square crop halves that.
3. **Multi-threaded inference** via COOP/COEP made detection 3.5× faster (178 → 51 ms/frame) but crashed Safari whenever depth also ran. The site is deliberately not cross-origin isolated.
4. **Recreating the depth worker after each reading** to release memory also crashed Safari, so one worker stays alive for the session.

The on-device path remains as the offline fallback.

## Configuration

### Environment variables

Set these in Vercel (or `pulse-point/.env.local` for local builds). Vite exposes every `VITE_*` value to the browser, so none of them may hold a secret.

| Variable | Purpose |
|---|---|
| `VITE_VISION_URL` | Vision server URL. Defaults to the Modal deployment, `https://tanush-appapogu--pulse-point-vision-fast.us-east.modal.direct`. |
| `VITE_SERVER_URL` | Optional legacy LocateAnything server. When healthy, it's queried about every 2.5 s while scanning. |
| `OPENROUTER_API_KEY` | Server-side key for the legacy `/api/ai` proxy only. The app doesn't call it. |
| `ALLOWED_ORIGIN` | Origin allowed by the legacy `/api/ai` proxy. Defaults to `https://6ixthsense.vercel.app`. |

### URL switches

For testing and diagnosis. Combine them, e.g. `https://6ixthsense.vercel.app/?debug=1&nodepth=1`.

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

CI (`.github/workflows/test.yml`) runs on every push to `main` and every pull request: web and shared-core tests, a production build, a syntax check of `api/ai.js`, and the server contract tests (`python -m unittest discover -s server -p "test_contract.py"`).

Model exports, for when weights change:

| Script | Output |
|---|---|
| `scripts/yoloe/export_promptable.py` | YOLOE-11s ONNX with the prompt vector as an input |
| `scripts/yoloe/build_prompt_pack.py` | `pack.json` and `pack.bin` from `pack_items.json` |
| `scripts/depth/export_depth.py` | Depth Anything V2 Metric-Indoor-Small, fp16 (WebGPU) and uint8 (CPU) |

Each script directory has its own `requirements.txt`.

Branching, commit format and review process are in [CONTRIBUTING.md](CONTRIBUTING.md). Commit messages are checked by the hooks installed with `scripts/setup-git-hooks.sh`; see [COMMIT_POLICY.md](COMMIT_POLICY.md).

Further reading: [ARCHITECTURE.md](ARCHITECTURE.md) for data flow and module boundaries, [TECHNICAL_OVERVIEW.md](TECHNICAL_OVERVIEW.md) for implementation detail, [CHANGELOG.md](CHANGELOG.md) for release history.

## Deployment

The web app deploys to Vercel from the repository root. `vercel.json` installs and builds `pulse-point/`, publishes `pulse-point/dist`, sets long-lived caching on hashed assets, and restricts camera and microphone permissions to the site's own origin.

The vision server deploys separately to Modal (see [Running it yourself](#running-it-yourself)).

### Legacy components

These remain in the repository but aren't part of the main app path:

- `server/` (outside `server/vision/`): the earlier experimental detector (LocateAnything-3B, an ImageNet/Grad-CAM classifier, and an unrelated tea-text classifier), deployed with `server/modal_app.py`. Only used when `VITE_SERVER_URL` is set. Details in [server/README.md](server/README.md).
- `api/ai.js`: an OpenRouter proxy the app no longer calls.
- `pulse-point-mobile/`: an Expo prototype for mobile UX ideas with simulated sensing. It is not the supported client and doesn't receive web changes.

## Limitations

- No obstacle detection, room mapping or route planning. Nothing in the app establishes that moving or reaching is safe.
- Some small items are still weak: headphones, sunglasses, chargers, door handles, smartwatches. Improving them will need vectors learned from labelled photos rather than text prompts.
- Depth Anything was trained on normal-lens indoor images, so readings through the ultra-wide lens are somewhat less accurate.
- iPhone browsers don't expose a reliable vibration API, and web pages can't access the iPhone's LiDAR.

## Credits and license

- [YOLOE](https://github.com/THU-MIG/yoloe) (YOLOE-11s via [Ultralytics](https://github.com/ultralytics/ultralytics)) and [Ultralytics YOLO11n](https://github.com/ultralytics/ultralytics), AGPL-3.0. Because these weights are AGPL-3.0, the deployed app's source must remain public.
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) Metric-Indoor-Small, fine-tuned by its authors from the Apache-2.0 Small model on Hypersim. Its model card states no separate license.
- [MobileCLIP](https://github.com/apple/ml-mobileclip) text encoder (offline, to build the prompt pack), [ONNX Runtime Web](https://onnxruntime.ai/docs/get-started/with-javascript.html), [FastAPI](https://fastapi.tiangolo.com/).

6ixth Sense application code is released under the [MIT License](LICENSE). The project was previously named Pulse Point, which is why some directories and packages still use that name.
