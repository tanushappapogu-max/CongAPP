"""Export YOLOE-11s to ONNX with prompt embeddings as a live input.

Inputs:  images (1, 3, 640, 640) float32 in [0, 1]; pe (1, K, 512) prompt embeddings, K is dynamic
Output:  output0 (1, 4 + K, 8400) — cx, cy, w, h rows then one sigmoid score row per prompt

The head is left unfused so the browser can swap prompts per search instead of baking classes in.

Usage (from repo root):
  python3 -m venv scripts/yoloe/.venv --system-site-packages
  scripts/yoloe/.venv/bin/pip install -r scripts/yoloe/requirements.txt
  scripts/yoloe/.venv/bin/python scripts/yoloe/export_promptable.py
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch import nn
from ultralytics import YOLOE
from ultralytics.nn.modules.head import YOLOEDetect

BUILD = Path(__file__).parent / "build"
IMGSZ = 640


class PromptableYOLOE(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, images, pe):
        y, x = [], images
        for layer in self.m.model:
            if layer.f != -1:
                x = y[layer.f] if isinstance(layer.f, int) else [x if j == -1 else y[j] for j in layer.f]
            if isinstance(layer, YOLOEDetect):
                x = [*x, pe.expand(images.shape[0], -1, -1)]
            x = layer(x)
            y.append(x if layer.i in self.m.save else None)
        return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="yoloe-11s-seg.pt")
    parser.add_argument("--out", default="yoloe-11s-promptable.onnx")
    args = parser.parse_args()

    BUILD.mkdir(exist_ok=True)
    os.chdir(BUILD)  # Ultralytics downloads weights into the working directory

    net = YOLOE(args.weights.replace("-seg.pt", ".yaml")).load(args.weights).model.eval()
    net.fuse(verbose=False)  # conv+bn only; text embeddings stay a runtime input
    head = net.model[-1]
    assert isinstance(head, YOLOEDetect) and not head.is_fused
    head.export, head.format, head.dynamic = True, "onnx", True

    def text_pe(names):
        return net.get_text_pe(names, cache_clip_model=True)

    wrapped = PromptableYOLOE(net).eval()
    img = torch.rand(1, 3, IMGSZ, IMGSZ)
    out_path = BUILD / args.out
    torch.onnx.export(
        wrapped,
        (img, text_pe(["eyeglasses", "keys", "wallet"])),
        out_path,
        input_names=["images", "pe"],
        output_names=["output0"],
        dynamic_axes={"pe": {1: "num_prompts"}, "output0": {1: "4_plus_prompts"}},
        opset_version=17,
        dynamo=False,
    )
    print(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    for names in (["eyeglasses", "sunglasses", "keys"], ["eyeglasses", "keys", "remote control", "wallet", "mug"]):
        pe = text_pe(names)
        with torch.no_grad():
            ref = wrapped(img, pe).numpy()
        t0 = time.time()
        got = sess.run(None, {"images": img.numpy(), "pe": pe.numpy()})[0]
        print(f"K={len(names)}: shape {got.shape}, max|onnx-torch| {np.abs(got - ref).max():.1e}, "
              f"{(time.time() - t0) * 1000:.0f} ms CPU")


if __name__ == "__main__":
    main()
