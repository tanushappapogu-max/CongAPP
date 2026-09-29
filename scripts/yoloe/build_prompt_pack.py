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
import torch
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
    # An item's vector is the renormalized mean of its prompt phrases (default: just its name).
    prompt_lists = [i.get("prompts") or [i["name"]] for i in items]
    phrases = sorted({p for ps in prompt_lists for p in ps})
    base = net.get_text_pe(phrases, cache_clip_model=True)[0]
    rows = [torch.nn.functional.normalize(base[[phrases.index(p) for p in ps]].mean(0), dim=-1) for ps in prompt_lists]
    vectors = torch.stack(rows).numpy().astype(np.float32)
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
