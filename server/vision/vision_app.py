"""6ixth Sense vision server: runs YOLOE detection and Depth Anything for the phone.

The phone keeps the camera, tracker, guidance, haptics and speech; it sends a small JPEG here and
gets boxes (and, every couple of seconds, a distance) back. It uses the exact model files and
prompt pack the web app ships, so results match on-device detection.

  GET  /health      -> {"ok": true, "device": ...}
  POST /v1/detect   multipart: file=JPEG (letterboxed to 640x640 by the phone), target=<pack item>
                    -> {"detections": [{"class", "score", "bbox": [x, y, w, h] in 640px input}], "ms"}
  POST /v1/depth    multipart: file=JPEG (the centered square crop, any size), box="x,y,w,h" in
                    crop fractions -> {"meters": float|null, "ms"}

Run locally (from repo root):
  pip install -r server/vision/requirements.txt
  cd server/vision && uvicorn vision_app:app --port 8788
"""
import json
import os
import sys
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from contract import cors_origins  # noqa: E402

ASSETS = Path(os.getenv("PULSEPOINT_ASSETS", HERE.parent.parent / "pulse-point" / "public"))
INPUT = 640
CONF_THRESH = 0.25
IOU_THRESH = 0.45
DEPTH_SIZE = 518
DEPTH_CORRECTION = 1.03  # centered-square crop vs full frame, measured on indoor photos
ROI_SHRINK = 0.2
FOREGROUND_PCTL = 0.3
MAX_IMAGE_BYTES = 1_500_000
RATE_LIMIT_PER_MIN = 900  # ~15 req/s per IP covers one phone scanning at full speed
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

app = FastAPI(title="6ixth Sense Vision")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins(),
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Accept", "Content-Type"],
)

_providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
_yoloe = ort.InferenceSession(str(ASSETS / "yoloe-11s.onnx"), providers=_providers)
_depth = ort.InferenceSession(str(ASSETS / "depth-indoor-small.fp16.onnx"), providers=_providers)
_meta = json.loads((ASSETS / "prompts" / "pack.json").read_text())
_vectors = np.frombuffer((ASSETS / "prompts" / "pack.bin").read_bytes(), np.float32).reshape(len(_meta["items"]), _meta["dim"])
_items = {item["name"]: (i, item) for i, item in enumerate(_meta["items"])}
_requests = {}


def _warmup():
    _yoloe.run(None, {"images": np.zeros((1, 3, INPUT, INPUT), np.float32), "pe": _vectors[None, :1]})
    _depth.run(None, {"pixel_values": np.zeros((1, 3, DEPTH_SIZE, DEPTH_SIZE), np.float32)})


_warmup()


def _rate_ok(request):
    ip = request.client.host if request.client else "unknown"
    now = time.monotonic()
    recent = [t for t in _requests.get(ip, []) if now - t < 60]
    if len(_requests) > 10_000:
        _requests.clear()
    if len(recent) >= RATE_LIMIT_PER_MIN:
        _requests[ip] = recent
        return False
    recent.append(now)
    _requests[ip] = recent
    return True


async def _read_image(file):
    data = await file.read(MAX_IMAGE_BYTES + 1)
    if file.content_type not in ("image/jpeg", "image/png", "image/webp") or not data or len(data) > MAX_IMAGE_BYTES:
        raise ValueError("expected a JPEG/PNG/WebP image under 1.5 MB")
    try:
        img = Image.open(BytesIO(data))
        if img.width * img.height > 4_000_000:
            raise ValueError("image too large")
        return img.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as e:
        raise ValueError("malformed image") from e


def prompt_set(target):
    """Target first, then its look-alike negatives — same as prompts.js buildPromptSet."""
    if target not in _items:
        raise ValueError("unknown target")
    _, item = _items[target]
    names = [target] + [n for n in item.get("negatives", []) if n in _items]
    return names, _vectors[[_items[n][0] for n in names]][None]


def decode(raw, names):
    """Argmax over prompts per anchor, threshold, NMS — same as engine.js."""
    scores = raw[4:]
    cls = scores.argmax(0)
    best = scores.max(0)
    keep = np.where(best >= CONF_THRESH)[0]
    order = keep[np.argsort(-best[keep])]
    out = []
    for i in order:
        cx, cy, w, h = raw[:4, i]
        box = [float(cx - w / 2), float(cy - h / 2), float(w), float(h)]
        if all(_iou(box, o["bbox"]) <= IOU_THRESH for o in out):
            out.append({"class": names[cls[i]], "score": float(best[i]), "bbox": box})
    return out


def _iou(a, b):
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / (a[2] * a[3] + b[2] * b[3] - inter or 1)


def sample_box_depth(depth, box):
    """Foreground-biased depth inside the box — same as depthSample.js sampleBoxDepth."""
    dh, dw = depth.shape
    x, y, w, h = box
    x0, x1 = max(0, int(np.floor((x + w * ROI_SHRINK) * dw))), min(dw, int(np.ceil((x + w * (1 - ROI_SHRINK)) * dw)))
    y0, y1 = max(0, int(np.floor((y + h * ROI_SHRINK) * dh))), min(dh, int(np.ceil((y + h * (1 - ROI_SHRINK)) * dh)))
    if x1 <= x0 or y1 <= y0:
        return None
    v = depth[y0:y1, x0:x1].ravel()
    v = np.sort(v[np.isfinite(v) & (v > 0)])
    if not v.size:
        return None
    return float(np.clip(v[min(v.size - 1, int(v.size * FOREGROUND_PCTL))], 0.1, 20.0))


@app.get("/health")
async def health():
    return {"ok": True, "device": _yoloe.get_providers()[0]}


@app.post("/v1/detect")
async def detect(request: Request, file: UploadFile = File(...), target: str = Form(...)):
    if not _rate_ok(request):
        return JSONResponse({"error": "rate limited"}, status_code=429)
    try:
        img = await _read_image(file)
        names, pe = prompt_set(target)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    t0 = time.perf_counter()
    if img.size != (INPUT, INPUT):
        img = img.resize((INPUT, INPUT))
    x = (np.asarray(img, np.float32) / 255).transpose(2, 0, 1)[None]
    # Inference runs in a worker thread so a slow depth request never blocks detection.
    raw = (await run_in_threadpool(_yoloe.run, None, {"images": x, "pe": pe.astype(np.float32)}))[0][0]
    return {"detections": decode(raw, names), "ms": round((time.perf_counter() - t0) * 1000)}


@app.post("/v1/depth")
async def depth(request: Request, file: UploadFile = File(...), box: str = Form(...)):
    if not _rate_ok(request):
        return JSONResponse({"error": "rate limited"}, status_code=429)
    try:
        img = await _read_image(file)
        parts = [float(v) for v in box.split(",")]
        if len(parts) != 4 or not all(np.isfinite(parts)) or parts[2] <= 0 or parts[3] <= 0:
            raise ValueError("box must be four frame-relative numbers")
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    t0 = time.perf_counter()
    x = ((np.asarray(img.resize((DEPTH_SIZE, DEPTH_SIZE), Image.BICUBIC), np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)[None]
    pred = (await run_in_threadpool(_depth.run, None, {"pixel_values": x.astype(np.float32)}))[0][0]
    meters = sample_box_depth(pred, parts)
    return {"meters": None if meters is None else meters / DEPTH_CORRECTION, "ms": round((time.perf_counter() - t0) * 1000)}
