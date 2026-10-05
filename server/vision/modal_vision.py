"""Modal deployment for the 6ixth Sense vision server (YOLOE + Depth Anything on a T4 GPU).

Deploy (from repo root, after `pip install modal && python3 -m modal setup`):
  modal deploy server/vision/modal_vision.py

The printed *.modal.direct URL is the web app's vision server (pulse-point/src/detection/remote.js). The container
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
        "uvicorn==0.32.0",
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


# Modal's direct HTTP routing: requests go straight to uvicorn in the container. A detection round trip
# was ~45 ms this way vs ~215 ms through @modal.asgi_app, whose per-request dispatch added ~190 ms.
@app.server(image=image, gpu="T4", port=8000, routing_region="us-east", scaledown_window=300, unauthenticated=True)
class Fast:
    @modal.enter()
    def start(self):
        import subprocess

        self.proc = subprocess.Popen(
            ["python", "-m", "uvicorn", "vision_app:app", "--host", "0.0.0.0", "--port", "8000"],
            cwd="/app/server/vision",
        )
