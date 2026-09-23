# -*- coding: utf-8 -*-
r"""Inference path for the demo prototype.

Reuses the EXACT modules from the completed experiment - nothing is reimplemented:
    dataset_prep.CONFIGS["384"], dataset_prep._resize   (preprocessing)
    extract_spatial_embeddings.VIEWS, crop_view         (view geometry)
    extract_embeddings.build_backbone, to_tensor        (frozen ResNet-50)
    run_spatial_experiment.build_representations        (tile2x2_concat)

Loads the fitted scaler + classifier from artifacts\model\.
"""
from __future__ import annotations

import json
import os

import joblib
import numpy as np
import torch
from PIL import Image, ImageOps

from dataset_prep import CONFIGS, _resize
from extract_embeddings import build_backbone, to_tensor
from extract_spatial_embeddings import VIEWS, crop_view
from run_spatial_experiment import build_representations

PROJECT = r"D:\data science\CNN"
MODEL_DIR = os.path.join(PROJECT, "artifacts", "model")
MODEL_PATH = os.path.join(MODEL_DIR, "demo_cleanliness_model.joblib")
META_PATH = os.path.join(MODEL_DIR, "demo_cleanliness_model_meta.json")

_CFG = CONFIGS["384"]
_VIEW_NAMES = [v[0] for v in VIEWS]


def load_artifacts():
    """Return (bundle, meta). Raises FileNotFoundError with a clear message."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Demo model artifact missing: {MODEL_PATH}\n"
            f"Run:  python prep\\fit_demo_model.py")
    bundle = joblib.load(MODEL_PATH)
    meta = json.load(open(META_PATH, encoding="utf-8"))
    return bundle, meta


def load_backbone():
    model, _ = build_backbone("resnet50")   # frozen, eval mode
    return model


def preprocess_views(pil_image: Image.Image) -> tuple[torch.Tensor, list[Image.Image]]:
    """EXIF-correct, cut the 7 deterministic views, letterbox each to 384."""
    base = ImageOps.exif_transpose(pil_image).convert("RGB")
    crops = [_resize(crop_view(base, v), _CFG) for v in VIEWS]
    batch = torch.stack([to_tensor(c, _CFG) for c in crops])
    return batch, crops


def embed(model, batch: torch.Tensor) -> np.ndarray:
    """(7, 2048) frozen ResNet-50 features - no grad, no fine-tuning."""
    with torch.no_grad():
        out = model(batch)
    return out.reshape(len(VIEWS), -1).cpu().numpy().astype(np.float64)


def predict(pil_image: Image.Image, model, bundle, threshold: float | None = None) -> dict:
    """Full demo prediction for one uploaded image."""
    batch, crops = preprocess_views(pil_image)
    per_view = embed(model, batch)                       # (7, 2048)
    X = build_representations(per_view[None, ...], _VIEW_NAMES)[bundle["representation"]]
    Xs = bundle["scaler"].transform(X)
    prob = float(bundle["classifier"].predict_proba(Xs)[0, 1])
    thr = bundle["default_threshold"] if threshold is None else float(threshold)
    label = bundle["positive_class"] if prob >= thr else "CLEAN"
    return {
        "label": label,
        "probability_not_clean": prob,
        "probability_clean": 1.0 - prob,
        "threshold": thr,
        "confidence": prob if label == bundle["positive_class"] else 1.0 - prob,
        "crops": crops,
        "view_names": _VIEW_NAMES,
        "feature_dim": int(X.shape[1]),
    }


if __name__ == "__main__":
    # smoke test on one existing dataset image (read-only)
    import csv
    rows = list(csv.DictReader(open(os.path.join(PROJECT, "cleanliness_manifest.csv"),
                                    encoding="utf-8-sig")))
    bundle, meta = load_artifacts()
    backbone = load_backbone()
    for want in ("NOT_CLEAN", "CLEAN"):
        r = next(x for x in rows if x["final_label"] == want)
        with Image.open(r["image_path"]) as im:
            out = predict(im, backbone, bundle)
        print(f"{r['image_file']:<30} manifest={want:<10} demo={out['label']:<10} "
              f"p(NOT_CLEAN)={out['probability_not_clean']:.3f} thr={out['threshold']} "
              f"dim={out['feature_dim']}")
