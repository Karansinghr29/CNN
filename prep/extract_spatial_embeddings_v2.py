# -*- coding: utf-8 -*-
r"""Spatial embeddings for the combined 163-image set.

Reuses the EXISTING 136 embeddings unchanged and computes only the 27 new
images with the identical frozen pipeline (same VIEWS, same 384 config, same
ResNet-50 IMAGENET1K_V2, eval + no_grad). Writes a NEW npz; the original
embedding files are not touched.
"""
from __future__ import annotations

import csv
import json
import os

import numpy as np
import torch

from dataset_prep import CONFIGS, PROJECT_ROOT, load_image, _resize
from extract_embeddings import build_backbone, to_tensor
from extract_spatial_embeddings import VIEWS, crop_view

OUT_DIR = os.path.join(PROJECT_ROOT, "artifacts", "embeddings")
OLD_NPZ = os.path.join(OUT_DIR, "embeddings_resnet50_384_spatial.npz")
NEW_NPZ = os.path.join(OUT_DIR, "embeddings_resnet50_384_spatial_v2.npz")
COMBINED = os.path.join(PROJECT_ROOT, "artifacts", "combined", "combined_manifest_v2.csv")


def main() -> None:
    cfg = CONFIGS["384"]
    with open(COMBINED, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 163

    old = np.load(OLD_NPZ, allow_pickle=False)
    cache = {os.path.normpath(p).lower(): old["embedding"][i]
             for i, p in enumerate(old["image_path"])}
    print(f"reusing {len(cache)} existing embeddings from {os.path.basename(OLD_NPZ)}")

    model, dim = build_backbone("resnet50")
    embs = np.zeros((len(rows), len(VIEWS), dim), dtype=np.float32)
    reused = computed = 0
    with torch.no_grad():
        for i, r in enumerate(rows):
            abs_path = os.path.normpath(os.path.join(PROJECT_ROOT,
                                                     r["image_path"].replace("/", os.sep)))
            hit = cache.get(abs_path.lower())
            if hit is not None:
                embs[i] = hit
                reused += 1
                continue
            base = load_image(abs_path)
            batch = torch.stack([to_tensor(_resize(crop_view(base, v), cfg), cfg) for v in VIEWS])
            embs[i] = model(batch).reshape(len(VIEWS), -1).cpu().numpy()
            computed += 1
            print(f"  computed {computed}/27  {r['image_file']}", end="\r")

    print(f"\nreused={reused}  newly computed={computed}")
    assert reused == 136 and computed == 27, (reused, computed)

    np.savez_compressed(
        NEW_NPZ,
        embedding=embs,
        view_names=np.array([v[0] for v in VIEWS]),
        image_path=np.array([r["image_path"] for r in rows]),
        image_file=np.array([r["image_file"] for r in rows]),
        room_id=np.array([r["room_id"] for r in rows]),
        property_group=np.array([r["property_group"] for r in rows]),
        sub_area=np.array([r["sub_area"] for r in rows]),
        label=np.array([r["label"] for r in rows]),
        dataset_version=np.array([r["dataset_version"] for r in rows]),
    )
    with open(NEW_NPZ.replace(".npz", "_meta.json"), "w", encoding="utf-8") as f:
        json.dump({"backbone": "resnet50 IMAGENET1K_V2 (frozen)", "config": cfg.as_dict(),
                   "views": [v[0] for v in VIEWS], "shape": list(embs.shape),
                   "reused_from_v1": reused, "newly_computed": computed,
                   "original_embeddings_modified": False}, f, indent=2)
    print(f"wrote {NEW_NPZ} shape={embs.shape}")


if __name__ == "__main__":
    main()
