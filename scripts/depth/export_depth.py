"""Export Depth Anything V2 Metric-Indoor-Small to ONNX (fp32 + uint8-weight quantized) for the web app.

Input:  pixel_values (1, 3, H, W) float32, ImageNet-normalized RGB, H and W multiples of 14 (dynamic)
Output: predicted_depth in meters (indoor head, max 20 m). The legacy exporter bakes the final
        upsample to the dummy input's size, so the map is always 266x350 over the full frame;
        callers sample it with frame-relative coordinates. Keep inputs at a 518 px short side:
        the model drifts 10-25% in meters at smaller sizes (same in PyTorch).

Usage (from repo root):
  python3 -m venv scripts/depth/.venv --system-site-packages
  scripts/depth/.venv/bin/pip install -r scripts/depth/requirements.txt
  scripts/depth/.venv/bin/python scripts/depth/export_depth.py
"""
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from PIL import Image
from transformers import AutoModelForDepthEstimation

MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
BUILD = Path(__file__).parent / "build"
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


def main():
    BUILD.mkdir(exist_ok=True)
    fp32 = BUILD / "depth-anything-v2-metric-indoor-small.onnx"
    int8 = BUILD / "depth-anything-v2-metric-indoor-small.uint8.onnx"

    model = DepthOnly(AutoModelForDepthEstimation.from_pretrained(MODEL_ID).eval())
    dummy = torch.randn(1, 3, 266, 350)
    torch.onnx.export(
        model, (dummy,), fp32,
        input_names=["pixel_values"], output_names=["predicted_depth"],
        dynamic_axes={"pixel_values": {2: "height", 3: "width"}, "predicted_depth": {1: "height", 2: "width"}},
        opset_version=17, dynamo=False,
    )
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QUInt8)
    for p in (fp32, int8):
        print(f"{p.name}: {p.stat().st_size / 1e6:.1f} MB")

    sample = Path(__file__).parent / "sample.jpg"
    if not sample.exists():
        return
    img = Image.open(sample)
    with torch.no_grad():
        ref = model(torch.from_numpy(preprocess(img, 518)))[0].numpy()
    print(f"reference (torch fp32 @518): median {np.median(ref):.2f} m")
    for p in (fp32, int8):
        sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
        for short in (518, 266, 196):
            x = preprocess(img, short)
            t0 = time.time()
            d = sess.run(None, {"pixel_values": x})[0][0]
            ms = (time.time() - t0) * 1000
            ref_small = np.asarray(Image.fromarray(ref).resize(d.shape[::-1], Image.BILINEAR))
            rel = np.median(np.abs(d - ref_small) / np.maximum(ref_small, 1e-3))
            print(f"  {p.name.split('.')[-2]:>5} @{short}: {d.shape}, median {np.median(d):.2f} m, "
                  f"median rel err vs ref {rel * 100:.1f}%, {ms:.0f} ms CPU")


if __name__ == "__main__":
    main()
