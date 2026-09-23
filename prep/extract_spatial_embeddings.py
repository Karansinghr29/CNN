# -*- coding: utf-8 -*-
"""Deterministic spatial views through the SAME frozen ResNet-50 (IMAGENET1K_V2).

Views, all computed on the EXIF-corrected portrait image (2250x4000), each then
letterboxed to 384x384 with the existing 384 config:

  whole  : full frame                      (identical to the existing baseline input)
  upper  : y 0.00-0.60   (walls, beds, desks, upper clutter)
  lower  : y 0.40-1.00   (floor region: debris, spills, shoes, bags)
  q1..q4 : 2x2 quadrants with 10% overlap  (keeps context, no random cropping)

No fine-tuning: eval mode, requires_grad_(False), torch.no_grad().
No PCA/scaler/classifier fitted here. Output is raw per-view embeddings.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
from PIL import Image

from dataset_prep import CONFIGS, PROJECT_ROOT, load_manifest, load_image, _resize
from extract_embeddings import build_backbone, to_tensor

OUT_DIR = os.path.join(PROJECT_ROOT, "artifacts", "embeddings")

# (name, x0, y0, x1, y1) as fractions of the EXIF-corrected image
VIEWS = [
    ("whole", 0.00, 0.00, 1.00, 1.00),
    ("upper", 0.00, 0.00, 1.00, 0.60),
    ("lower", 0.00, 0.40, 1.00, 1.00),
    ("q1",    0.00, 0.00, 0.55, 0.55),
    ("q2",    0.45, 0.00, 1.00, 0.55),
    ("q3",    0.00, 0.45, 0.55, 1.00),
    ("q4",    0.45, 0.45, 1.00, 1.00),
]


def crop_view(img: Image.Image, box) -> Image.Image:
    w, h = img.size
    _, x0, y0, x1, y1 = box
    return img.crop((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)))


def main() -> None:
    cfg = CONFIGS["384"]
    rows = load_manifest()                       # 136 supervised rows
    model, dim = build_backbone("resnet50")      # frozen feature extractor

    embs = np.zeros((len(rows), len(VIEWS), dim), dtype=np.float32)
    with torch.no_grad():
        for i, r in enumerate(rows):
            base = load_image(r["image_path"])   # EXIF-corrected, full resolution
            tensors = [to_tensor(_resize(crop_view(base, v), cfg), cfg) for v in VIEWS]
            out = model(torch.stack(tensors))
            embs[i] = out.reshape(len(VIEWS), -1).cpu().numpy()
            if (i + 1) % 10 == 0 or i + 1 == len(rows):
                print(f"  {i+1}/{len(rows)}", end="\r")

    os.makedirs(OUT_DIR, exist_ok=True)
    npz = os.path.join(OUT_DIR, "embeddings_resnet50_384_spatial.npz")
    np.savez_compressed(
        npz,
        embedding=embs,                                   # (136, 7, 2048)
        view_names=np.array([v[0] for v in VIEWS]),
        image_path=np.array([r["image_path"] for r in rows]),
        image_file=np.array([r["image_file"] for r in rows]),
        room_id=np.array([r["room_id"] for r in rows]),
        property_group=np.array([r["property_group"] for r in rows]),
        sub_area=np.array([r["sub_area"] for r in rows]),
        final_label=np.array([r["final_label"] for r in rows]),
    )
    with open(npz.replace(".npz", "_meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "backbone": "resnet50 IMAGENET1K_V2", "frozen": True, "fine_tuned": False,
            "classifier_trained": False, "config": cfg.as_dict(),
            "views": [{"name": v[0], "box_fractions": list(v[1:])} for v in VIEWS],
            "n_images": len(rows), "per_view_dim": dim, "shape": list(embs.shape),
            "fitted_transforms": "none - raw backbone outputs",
            "metadata_columns_are_not_features": ["room_id", "property_group", "sub_area",
                                                  "image_file", "image_path"],
        }, f, indent=2)
    print(f"\nwrote {npz}  shape={embs.shape}")

    # sanity: the whole-image view must reproduce the existing 384 baseline embeddings
    base = np.load(os.path.join(OUT_DIR, "embeddings_resnet50_384.npz"))["embedding"]
    d = float(np.abs(base - embs[:, 0, :]).max())
    print(f"max abs diff vs existing 384 whole-image embeddings: {d:.6f}  "
          f"({'MATCH' if d < 1e-3 else 'MISMATCH - investigate'})")


if __name__ == "__main__":
    main()
