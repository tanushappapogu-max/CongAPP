"""Modal deployment for the Pulse Point vision server (YOLOE + Depth Anything on a T4 GPU).

Deploy (from repo root, after `pip install modal && python3 -m modal setup`):
  modal deploy server/vision/modal_vision.py

The printed URL is the web app's vision server (pulse-point/src/detection/remote.js). The container
sleeps after a few idle minutes and Modal bills only while it runs.
"""
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent.parent
PUBLIC = ROOT / "pulse-point" / "public"

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04", add_python="3.11")
    .pip_install(
        "fastapi==0.115.0",
        "python-multipart==0.0.9",
        "numpy<2.3",
        "Pillow==11.0.0",
        "onnxruntime-gpu==1.20.2",
    )
    .env({"PULSEPOINT_ASSETS": "/assets"})
    .add_local_file(PUBLIC / "yoloe-11s.onnx", "/assets/yoloe-11s.onnx")
    .add_local_file(PUBLIC / "depth-indoor-small.fp16.onnx", "/assets/depth-indoor-small.fp16.onnx")
    .add_local_dir(PUBLIC / "prompts", "/assets/prompts")
    .add_local_file(ROOT / "server" / "contract.py", "/app/server/contract.py")
    .add_local_file(ROOT / "server" / "vision" / "vision_app.py", "/app/server/vision/vision_app.py")
)

app = modal.App("pulse-point-vision")


@app.function(image=image, gpu="T4", timeout=600, scaledown_window=300)
@modal.concurrent(max_inputs=16)
@modal.asgi_app()
def web():
    import sys

    sys.path.insert(0, "/app/server/vision")
    from vision_app import app as fastapi_app

    return fastapi_app
