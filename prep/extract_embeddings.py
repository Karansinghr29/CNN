# -*- coding: utf-8 -*-
"""Step 6/7 - pretrained backbone used ONLY as a frozen feature extractor.

Pipeline:  image -> EXIF correction -> letterbox resize -> normalize
           -> frozen pretrained backbone -> embedding -> (classifier: NOT here)

This module does NOT train or fine-tune anything. The backbone runs under
torch.no_grad() in eval mode. No classifier head is fitted.

Leakage discipline: NO normalization statistics, PCA, feature selection or
thresholds are fitted here. Embeddings are saved raw. Any learned transform must
be fitted inside the training portion of each LOPO fold, later.

Usage:
    python extract_embeddings.py --config 224 --backbone resnet50
    python extract_embeddings.py --config 384 --backbone resnet50
    python extract_embeddings.py --check          # report availability only
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from dataset_prep import CONFIGS, PROJECT_ROOT, load_manifest, load_image

# Generated artifacts live apart from the original dataset and the manifest.
OUT_DIR = os.path.join(PROJECT_ROOT, "artifacts", "embeddings")

# Backbones considered, smallest useful first. Weights download on first use.
BACKBONES = {
    "mobilenet_v3_small": {"dim": 576,  "weights_mb": 10,  "note": "smallest; weakest features"},
    "efficientnet_b0":    {"dim": 1280, "weights_mb": 21,  "note": "good size/quality trade-off"},
    "resnet50":           {"dim": 2048, "weights_mb": 98,  "note": "standard baseline, IMAGENET1K_V2"},
    "convnext_tiny":      {"dim": 768,  "weights_mb": 114, "note": "stronger, modern"},
}


def check_environment() -> dict:
    """Report what is installed. Downloads nothing."""
    status = {"torch": None, "torchvision": None, "timm": None,
              "onnxruntime": None, "cached_checkpoints": []}
    for mod in ("torch", "torchvision", "timm", "onnxruntime"):
        try:
            m = __import__(mod)
            status[mod] = getattr(m, "__version__", "?")
        except ImportError:
            status[mod] = None
    if status["torch"]:
        import torch
        hub = torch.hub.get_dir()
        ck = os.path.join(hub, "checkpoints")
        status["hub_dir"] = hub
        status["cached_checkpoints"] = sorted(os.listdir(ck)) if os.path.isdir(ck) else []
        status["cuda"] = torch.cuda.is_available()
    return status


def build_backbone(name: str):
    """Return (feature_extractor_module, embedding_dim). Frozen, eval mode."""
    import torch
    import torchvision.models as tvm

    if name == "resnet50":
        m = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
        m.fc = torch.nn.Identity()
        dim = 2048
    elif name == "efficientnet_b0":
        m = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1)
        m.classifier = torch.nn.Identity()
        dim = 1280
    elif name == "convnext_tiny":
        m = tvm.convnext_tiny(weights=tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        m.classifier[2] = torch.nn.Identity()
        dim = 768
    elif name == "mobilenet_v3_small":
        m = tvm.mobilenet_v3_small(weights=tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        m.classifier = torch.nn.Identity()
        dim = 576
    else:
        raise ValueError(f"unknown backbone: {name}")

    m.eval()
    for p in m.parameters():      # frozen: no training, no fine-tuning
        p.requires_grad_(False)
    return m, dim


def to_tensor(img, cfg):
    """PIL -> normalized CHW float tensor using the config's fixed ImageNet stats."""
    import torch
    arr = np.asarray(img, dtype=np.float32) / 255.0          # HWC
    arr = (arr - np.array(cfg.mean, dtype=np.float32)) / np.array(cfg.std, dtype=np.float32)
    return torch.from_numpy(arr.transpose(2, 0, 1))


def extract(config_name: str, backbone_name: str, batch_size: int = 8) -> str:
    import torch

    cfg = CONFIGS[config_name]
    rows = load_manifest()                       # 136 supervised rows
    model, dim = build_backbone(backbone_name)

    embs = np.zeros((len(rows), dim), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            chunk = rows[start:start + batch_size]
            batch = torch.stack([to_tensor(load_image(r["image_path"], cfg), cfg) for r in chunk])
            out = model(batch)
            embs[start:start + len(chunk)] = out.reshape(len(chunk), -1).cpu().numpy()
            print(f"  {min(start + batch_size, len(rows))}/{len(rows)}", end="\r")

    os.makedirs(OUT_DIR, exist_ok=True)
    stem = f"embeddings_{backbone_name}_{config_name}"
    npz = os.path.join(OUT_DIR, stem + ".npz")
    np.savez_compressed(
        npz,
        embedding=embs,
        image_path=np.array([r["image_path"] for r in rows]),
        image_file=np.array([r["image_file"] for r in rows]),
        room_id=np.array([r["room_id"] for r in rows]),
        property_group=np.array([r["property_group"] for r in rows]),
        sub_area=np.array([r["sub_area"] for r in rows]),
        final_label=np.array([r["final_label"] for r in rows]),
    )
    with open(os.path.join(OUT_DIR, stem + "_meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "backbone": backbone_name, "frozen": True, "fine_tuned": False,
            "classifier_trained": False,
            "config": cfg.as_dict(), "n_images": len(rows), "embedding_dim": dim,
            "metadata_columns_are_not_features": ["room_id", "property_group", "sub_area",
                                                  "image_file", "image_path"],
            "fitted_transforms": "none - raw backbone outputs; scaling/PCA must be fitted per-fold later",
        }, f, indent=2)
    print(f"\nwrote {npz}  shape={embs.shape}")
    return npz


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=sorted(CONFIGS), default="224")
    ap.add_argument("--backbone", choices=sorted(BACKBONES), default="resnet50")
    ap.add_argument("--check", action="store_true", help="report environment only")
    args = ap.parse_args()

    st = check_environment()
    print("environment:", json.dumps(st, indent=2))
    if args.check:
        return
    if not st["torch"] or not st["torchvision"]:
        print("\ntorch/torchvision not installed - cannot extract embeddings.", file=sys.stderr)
        print("Install a CPU-only build, then re-run:", file=sys.stderr)
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu",
              file=sys.stderr)
        sys.exit(2)
    extract(args.config, args.backbone)


if __name__ == "__main__":
    main()
