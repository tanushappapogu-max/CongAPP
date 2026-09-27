"""Build the browser prompt pack: YOLOE text-prompt embeddings for every item in pack_items.json.

Writes pulse-point/public/prompts/pack.json (item metadata) and pack.bin (float32 [N, 512],
row i = embedding for items[i].name). Vectors must come from the same YOLOE checkpoint as the
exported ONNX, so rebuild the pack whenever the model is re-exported.

Usage (from repo root, after export_promptable.py's venv setup):
  scripts/yoloe/.venv/bin/python scripts/yoloe/build_prompt_pack.py
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from ultralytics import YOLOE

HERE = Path(__file__).parent
REPO = HERE.parent.parent
OUT = REPO / "pulse-point" / "public" / "prompts"
WEIGHTS = "yoloe-11s-seg.pt"


def main():
    src = json.loads((HERE / "pack_items.json").read_text())
    items = [dict(i) for i in src["items"]]
    custom = {i["name"] for i in items}
    items += [{"name": n, "aliases": [], "negatives": []} for n in src["coco"] if n not in custom]

    names = [i["name"] for i in items]
    assert len(names) == len(set(names)), "duplicate item names"
    for item in items:
        for neg in item["negatives"]:
            assert neg in names, f"{item['name']}: unknown negative {neg!r}"

    (HERE / "build").mkdir(exist_ok=True)
    os.chdir(HERE / "build")
    net = YOLOE(WEIGHTS.replace("-seg.pt", ".yaml")).load(WEIGHTS).model.eval()
    vectors = net.get_text_pe(names, cache_clip_model=True)[0].numpy().astype(np.float32)
    assert vectors.shape == (len(names), 512)

    OUT.mkdir(parents=True, exist_ok=True)
    blob = vectors.tobytes()
    (OUT / "pack.bin").write_bytes(blob)
    meta = {
        "model": "yoloe-11s",
        "dim": 512,
        "sha256": hashlib.sha256(blob).hexdigest(),
        "items": items,
    }
    (OUT / "pack.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"wrote {len(items)} prompts ({len(blob) / 1024:.0f} KB) to {OUT}")


if __name__ == "__main__":
    main()
