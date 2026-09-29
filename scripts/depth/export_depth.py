"""Export Depth Anything V2 Metric-Indoor-Small to ONNX for the web app.

Writes pulse-point/public/depth-indoor-small.fp16.onnx (fp16 weights, ~50 MB). Depth only runs on
WebGPU: on the CPU it takes seconds per frame, so CPU-only devices use width-based distance.

Input:  pixel_values (1, 3, H, W) float32, ImageNet-normalized RGB, H and W multiples of 14 (dynamic)
Output: predicted_depth in meters (indoor head, max 20 m). The legacy exporter bakes the final
        upsample to the dummy input's size, so the map is always 266x350 over the full frame;
        callers sample it with frame-relative coordinates. Keep inputs at a 518 px short side:
        the model drifts 10-25% in meters at smaller sizes (same in PyTorch).

The DINOv2 position-embedding resize is exported as bilinear instead of bicubic because ONNX
Runtime Web's WebGPU ResizeBiCubic shader fails to compile; this changes depth by ~0.2%.

Usage (from repo root):
  python3 -m venv scripts/depth/.venv --system-site-packages
  scripts/depth/.venv/bin/pip install -r scripts/depth/requirements.txt
  scripts/depth/.venv/bin/python scripts/depth/export_depth.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForDepthEstimation

sys.path.insert(0, str(Path(__file__).parent.parent))
from onnx_fp16 import to_fp16  # noqa: E402

MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
HERE = Path(__file__).parent
BUILD = HERE / "build"
PUBLIC = HERE.parent.parent / "pulse-point" / "public"
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class DepthOnly(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, pixel_values):
        return self.m(pixel_values=pixel_values).predicted_depth


def preprocess(img: Image.Image, short_side: int) -> np.ndarray:
    w, h = img.size
    s = short_side / min(w, h)
    nw, nh = (max(14, round(w * s / 14) * 14), max(14, round(h * s / 14) * 14))
    x = (np.asarray(img.convert("RGB").resize((nw, nh), Image.BICUBIC), dtype=np.float32) / 255 - MEAN) / STD
    return x.transpose(2, 0, 1)[None]


def export_bilinear(model, path):
    original = F.interpolate

    def interpolate(*args, **kwargs):
        if kwargs.get("mode") == "bicubic":
            kwargs["mode"] = "bilinear"
        return original(*args, **kwargs)

    F.interpolate = interpolate
    try:
        torch.onnx.export(
            model, (torch.randn(1, 3, 266, 350),), path,
            input_names=["pixel_values"], output_names=["predicted_depth"],
            dynamic_axes={"pixel_values": {2: "height", 3: "width"}, "predicted_depth": {1: "height", 2: "width"}},
            opset_version=17, dynamo=False,
        )
    finally:
        F.interpolate = original


def main():
    BUILD.mkdir(exist_ok=True)
    fp32 = BUILD / "depth-indoor-small.fp32.onnx"
    fp16 = PUBLIC / "depth-indoor-small.fp16.onnx"

    model = DepthOnly(AutoModelForDepthEstimation.from_pretrained(MODEL_ID).eval())
    export_bilinear(model, fp32)
    to_fp16(str(fp32), str(fp16))
    for p in (fp32, fp16):
        print(f"{p.name}: {p.stat().st_size / 1e6:.1f} MB")

    sample = HERE / "sample.jpg"
    if not sample.exists():
        return
    x = preprocess(Image.open(sample), 518)
    with torch.no_grad():
        ref = model(torch.from_numpy(x))[0].numpy()
    for p in (fp16,):
        sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
        t0 = time.time()
        d = sess.run(None, {"pixel_values": x})[0][0]
        ms = (time.time() - t0) * 1000
        r = np.asarray(Image.fromarray(ref).resize(d.shape[::-1], Image.BILINEAR))
        err = np.abs(d - r) / np.maximum(r, 1e-3)
        print(f"  {p.name}: median rel err vs PyTorch {np.median(err) * 100:.2f}%, "
              f"p90 {np.percentile(err, 90) * 100:.1f}%, {ms:.0f} ms CPU")


if __name__ == "__main__":
    main()
