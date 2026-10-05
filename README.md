# Pulse Point

**Live demo:** https://pulse-point-steel.vercel.app

[![Tests](https://github.com/tanushappapogu-max/CongAPP/actions/workflows/test.yml/badge.svg)](https://github.com/tanushappapogu-max/CongAPP/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Pulse Point is a browser-based prototype that helps a blind or low-vision user find a named object with haptics and speech. You say or type what you're looking for ("my glasses", "keys", "a cup"); the camera finds it; the phone tells you which way to turn, then how far away it is, until you can reach it.

The supported app is `pulse-point/` (React + Vite + ONNX Runtime Web). It is a prototype: it has not been validated for independent mobility, obstacle avoidance, or any safety-critical use.

---

## How it works

```text
"find my glasses"
      │
      ▼
Prompt pack lookup ── target vector + look-alike vectors (e.g. eyeglasses vs sunglasses)
      │
      ▼
Detection ─────────── YOLOE (open vocabulary) · YOLO11n backup (80 COCO objects)
      │                 runs on the vision server when one is configured, else on the phone
      ▼
Box tracker ────────── smooths boxes between frames
      │
      ▼
Distance ───────────── Depth Anything V2 (meters, every ~1–4 s)
      │                 + object width and the lens's field of view in between
      ▼
Guidance ──────────── turn left/right · tilt up/down · closer · reach
      │
      ▼
Haptics + speech
```

### Finding the object: YOLOE and the prompt pack

- **YOLOE** is an open-vocabulary detector: instead of a fixed class list it takes a *vector* describing what to look for. We export it to ONNX with that vector as a live input (`scripts/yoloe/export_promptable.py`).
- **The prompt pack** (`pulse-point/public/prompts/pack.{json,bin}`, 226 KB) holds precomputed vectors for 113 items: 33 household items (glasses, keys, wallet, AirPods, pill bottle, white cane, door handle, …) plus the 80 COCO classes. It's our self-hosted vector database: served from our own site, loaded once, and searched on the phone with no API calls.
- **Look-alikes:** each search scores the target against its look-alikes, so a box only counts as "eyeglasses" if it looks more like eyeglasses than sunglasses or a glasses case.
- **Tuned vectors:** we tested every item on ~900 real photos. YOLOE ignores the plain names of some small objects (it sees watches as "clock"), so those items use a blend of the phrases it actually responds to. Watch went from 12% → 75% found, eyeglasses 12% → 62%, keys 0% → 50%. See `scripts/yoloe/pack_items.json` (`prompts` and `coco` fields).
- **Aliases and typos:** "my glasses", "airpods", "medicine", "sofa", and close misspellings ("keyz") resolve to the right item (`pulse-point/src/detection/prompts.js`).
- **YOLO11n backup:** lighter, but only knows the 80 COCO objects. Used on devices without WebGPU for COCO targets and if YOLOE fails to load.

### How far away it is: Depth Anything + lens geometry

- **Depth Anything V2 Metric-Indoor-Small** returns distance in meters. It runs on the **largest centered square** of the frame (518 px) rather than the whole wide frame: on indoor photos that reads within ~2% of the full frame (after a fixed ×1/1.03 correction) at about **half the memory** (~280 MB peak vs ~550 MB), because the model compares every image patch with every other.
- **Between depth readings**, distance follows the box size (twice as wide = half as far), so guidance updates every frame.
- **The camera uses the widest back lens** (the 0.5× ultra-wide on iPhone) so the user finds the target without sweeping as far. The width-based distance math uses that lens's real field of view (~103° vs ~64° for the main camera; `lensFovDeg` in `pulse-point/src/lib/camera.js`) and the frame's long side, so it's right in portrait and landscape.

---

## The vision server (recommended for phones)

Running YOLOE and Depth Anything inside a phone browser is heavy. On an iPhone it repeatedly ran out of memory and Safari killed the tab (details below). The **vision server** (`server/vision/`) runs both models instead:

- **The phone** handles the camera, tracker, guidance, haptics and speech. It sends a ~35 KB JPEG per frame and gets boxes back (plus a distance about once a second). **No models are downloaded to the phone.**
- **The server** uses the exact same model files and prompt pack as the app, so results match on-device detection.
- **Fallback:** if the server is unreachable or a request fails, the app automatically switches to on-device detection.
- **Measured locally:** ~79 ms per frame (~8 frames/s) end to end, with browser memory around 10–20 MB instead of hundreds.

| Endpoint | Input | Output |
|---|---|---|
| `GET /health` | | `{"ok": true, "device": "CUDAExecutionProvider" or "CPUExecutionProvider"}` |
| `POST /v1/detect` | `file` = 640×640 letterboxed JPEG, `target` = prompt-pack item name | `{"detections": [{class, score, bbox}], "ms"}` |
| `POST /v1/depth` | `file` = centered square crop JPEG, `box` = `x,y,w,h` (crop fractions) | `{"meters", "ms"}` |

The server restricts CORS to the app's origins (`server/contract.py`), rate-limits per IP, and caps image size.

> **Privacy:** with a vision server configured, camera frames and the target name are sent to it while scanning. The server doesn't store them, but they do leave the phone.

### Where to run it

| Option | Cost | Notes |
|---|---|---|
| **Modal** (T4 GPU), `server/vision/modal_vision.py` | Free within Modal's $30/month credit | Fastest. Sleeps after ~5 idle minutes; the first request after that takes ~10–30 s while it wakes (the app wakes it when the page opens). |
| **Your own computer + Cloudflare Tunnel** | Free, no account | Good for testing on a phone today. Only works while your computer is on; the tunnel link changes each run. |
| **Hugging Face Spaces** (free CPU) | Free | Always-available link, but no GPU (slower depth) and it sleeps after ~2 days unused. |

**Run it locally:**

```bash
python3 -m venv .venv && .venv/bin/pip install -r server/vision/requirements.txt
cd server/vision && ../../.venv/bin/uvicorn vision_app:app --port 8788
```

Then open the app with `?server=http://localhost:8788`.

**Test on a phone without deploying** (your computer runs the server; `cloudflared` gives it an https link):

```bash
brew install cloudflared
cloudflared tunnel --url http://localhost:8788
```

Open `https://pulse-point-steel.vercel.app/?server=<the trycloudflare.com link>` on the phone.

**Deploy to Modal:**

```bash
pip install modal
python3 -m modal setup                          # one-time browser login
modal deploy server/vision/modal_vision.py      # prints the server URL
```

Set the printed URL as `VITE_VISION_URL` in Vercel (see below) and redeploy the site.

---

## What we learned about iPhone (why the server exists)

We diagnosed the iPhone crashes with Safari Web Inspector memory recordings and URL switches that turn parts of the app on and off:

1. **Safari's WebGPU path leaked memory** (~150 MB/s while scanning; the tab was killed in ~20 s). On-device detection now uses the CPU on every iPhone browser (they all run Safari's engine) and desktop Safari. `?gpu=1` re-enables WebGPU for testing.
2. **Detection alone on the CPU was stable; detection + Depth Anything wasn't.** Depth Anything's file is 27 MB but it peaks at ~550 MB while running on a wide frame. The centered square crop halves that.
3. **Multi-threaded CPU inference** (COOP/COEP headers) made detection 3.5× faster (178 → 51 ms/frame) but crashed iPhone Safari whenever depth also ran, so the site is not cross-origin isolated.
4. **Closing and reopening the depth worker after every reading** to free memory also crashed Safari, so one worker stays alive.

The vision server sidesteps all of this. The on-device path remains as the offline fallback.

---

## URL switches (testing and diagnosis)

Add these to the URL, e.g. `https://pulse-point-steel.vercel.app/?debug=1&nodepth=1`.

| Switch | Effect |
|---|---|
| `?debug=1` | Shows a readout: server status, which detector/backend is running, depth backend, thread count, frame time, best score for the target, memory |
| `?server=URL` | Use this vision server; `?server=off` forces on-device detection |
| `?nodepth=1` | No depth model |
| `?cpu=1` / `?gpu=1` | Force CPU / force WebGPU for on-device inference |
| `?threads=N` | Force the CPU backend's thread count |
| `?depthsize=392` | Smaller depth crop (~200 MB peak, slightly noisier, corrected ×1/1.15) |
| `?nodetect=1` | Camera and overlay only; no detection model is loaded or run |
| `?captureonly=1` | Load the detector and copy each frame into its input, but never run it |

---

## Projects

### Web app (`pulse-point/`), the supported app

```bash
cd pulse-point
npm install
npm run dev        # development server
npm test           # unit tests
npm run build      # production build
```

Key files:

| File | What it does |
|---|---|
| `src/detection/engine.js` | Detector selection (server / YOLOE / YOLO11n), frame capture, box decoding |
| `src/detection/remote.js` | Vision server client and fallback |
| `src/detection/prompts.js` | Prompt pack loading, alias/typo matching, target + look-alike vectors |
| `src/detection/depth.js`, `depth.worker.js`, `depthSample.js` | Depth Anything (server or on-device worker), square crop, box sampling |
| `src/detection/distance.js`, `src/guidance/compute.js` | Width/FOV distance and turn/tilt/reach guidance |
| `src/lib/camera.js` | Wide-lens selection and field of view |
| `src/lib/flags.js` | URL switches and the Safari/iOS defaults |

### Model and data scripts (`scripts/`)

| Script | What it does |
|---|---|
| `scripts/yoloe/export_promptable.py` | Exports YOLOE-11s to ONNX with prompt vectors as a live input |
| `scripts/yoloe/build_prompt_pack.py` | Builds the prompt pack from `scripts/yoloe/pack_items.json` (rebuild whenever the model is re-exported) |
| `scripts/depth/export_depth.py` | Exports Depth Anything V2 Metric-Indoor-Small (fp16 for WebGPU, uint8 for CPU) |

### Mobile app (`pulse-point-mobile/`): exploratory, degraded prototype

> [!NOTE]
> `pulse-point-mobile/` is an exploratory prototype for mobile UX concepts with simulated sensing. It is **not** the supported client. Don't port web changes to it unless specifically asked.

### Legacy services

- `server/` (outside `server/vision/`) is the older experimental Python service (LocateAnything-3B, an ImageNet/Grad-CAM classifier, and an unrelated tea-text classifier), deployed by `server/modal_app.py`. The web app only calls it when `VITE_SERVER_URL` is set. See [`server/README.md`](server/README.md).
- `api/ai.js` is a legacy OpenRouter proxy that the web app no longer calls.

---

## Deploying the web app (Vercel)

Import the repo into Vercel and keep the repository root as the root directory. `vercel.json` builds `pulse-point/` and publishes `pulse-point/dist`.

### Environment variables

| Variable | Description |
|---|---|
| `VITE_VISION_URL` | Overrides the vision server URL (defaults to the Modal deployment, `https://tanush-appapogu--pulse-point-vision-web.modal.run`). When healthy, detection and depth run there and frames are uploaded while scanning. Vite exposes `VITE_*` values to the browser, so this must not contain a secret. |
| `VITE_SERVER_URL` | Optional URL for the legacy LocateAnything server. When healthy, it is queried about every 2.5 s while scanning. |
| `OPENROUTER_API_KEY` | Only for the legacy `/api/ai` proxy, which the app doesn't call. Never prefix a secret with `VITE_`. |
| `ALLOWED_ORIGIN` | Optional origin for the legacy `/api/ai` proxy (defaults to `https://pulse-point-steel.vercel.app`). |

---

## Limitations

- Prototype only: no obstacle detection, room mapping or route planning, and nothing establishes that moving or reaching is safe.
- Some small household items are still weak (headphones, sunglasses, chargers, door handles, smartwatches); improving them needs vectors learned from labeled photos.
- Depth Anything was trained on normal-lens indoor photos, so readings on the ultra-wide lens are somewhat less exact.
- iPhone browsers don't expose reliable vibration APIs, and websites can't access the iPhone's LiDAR.

---

## Acknowledgments and licenses

- [YOLOE](https://github.com/THU-MIG/yoloe) (YOLOE-11s via [Ultralytics](https://github.com/ultralytics/ultralytics)) and [Ultralytics YOLO11n](https://github.com/ultralytics/ultralytics): AGPL-3.0. Because these weights are AGPL-3.0, the deployed app's source must stay public.
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) Metric-Indoor-Small: fine-tuned by its authors from the Apache-2.0 Small model on the Hypersim dataset; its model card states no separate license.
- [MobileCLIP](https://github.com/apple/ml-mobileclip) text encoder (used offline to build the prompt pack), [ONNX Runtime Web](https://onnxruntime.ai/docs/get-started/with-javascript.html), [FastAPI](https://fastapi.tiangolo.com/).

Pulse Point application code is licensed under the MIT License; see [LICENSE](LICENSE).
