# -*- coding: utf-8 -*-
"""Reproducible image loading / preprocessing for the room-cleanliness dataset.

READ-ONLY with respect to the dataset: images are opened, never written.
No augmentation here. No training here.

Usage:
    from dataset_prep import load_manifest, load_image, PreprocessConfig, CONFIGS
"""
from __future__ import annotations

import csv
import hashlib
import os
from dataclasses import dataclass, asdict
from typing import Iterator

from PIL import Image, ImageOps

PROJECT_ROOT = r"D:\data science\CNN"
MANIFEST = os.path.join(PROJECT_ROOT, "cleanliness_manifest.csv")

SUPERVISED_LABELS = ("CLEAN", "NOT_CLEAN")   # UNCERTAIN excluded from supervised use
EXCLUDED_LABEL = "UNCERTAIN"

# Columns that are METADATA ONLY - never model input features.
METADATA_ONLY = ("room_id", "property_group", "sub_area", "image_file", "image_path")


def property_group(room_id: str) -> str:
    """A33 C double common -> A33.  Grouping metadata only, never a feature."""
    return room_id.split()[0].upper()


def load_manifest(path: str = MANIFEST, supervised_only: bool = True) -> list[dict]:
    """Read the manifest. supervised_only=True keeps CLEAN + NOT_CLEAN (136 rows)."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["property_group"] = property_group(r["room_id"])
    if supervised_only:
        rows = [r for r in rows if r["final_label"] in SUPERVISED_LABELS]
    return rows


def excluded_rows(path: str = MANIFEST) -> list[dict]:
    """The UNCERTAIN rows: retained in the manifest, excluded from supervised use."""
    return [r for r in load_manifest(path, supervised_only=False)
            if r["final_label"] == EXCLUDED_LABEL]


@dataclass(frozen=True)
class PreprocessConfig:
    """Deterministic preprocessing. Augmentation is NOT part of this stage."""
    name: str
    size: int                 # square target edge, e.g. 224 or 384
    resize_mode: str          # "letterbox" (pad, no crop) or "center_crop"
    interpolation: str        # "bicubic" | "bilinear" | "lanczos"
    pad_color: tuple = (0, 0, 0)
    # ImageNet statistics - applied by the embedding step, recorded here for reproducibility
    mean: tuple = (0.485, 0.456, 0.406)
    std: tuple = (0.229, 0.224, 0.225)

    def as_dict(self) -> dict:
        return asdict(self)


_INTERP = {
    "bicubic": Image.BICUBIC,
    "bilinear": Image.BILINEAR,
    "lanczos": Image.LANCZOS,
}

# Letterbox is the default: every source image is 4000x2250 (16:9) and portrait after
# EXIF correction. A center crop at that aspect ratio would discard ~44% of the frame,
# and evidence in this dataset sits at frame edges (floor corners, doorways).
CONFIGS = {
    "224": PreprocessConfig(name="224", size=224, resize_mode="letterbox", interpolation="bicubic"),
    "384": PreprocessConfig(name="384", size=384, resize_mode="letterbox", interpolation="bicubic"),
}


def load_image(path: str, config: PreprocessConfig | None = None) -> Image.Image:
    """Open an image read-only, apply EXIF orientation, optionally resize.

    The file on disk is never written. exif_transpose returns a NEW image.
    """
    with Image.open(path) as im:
        im.load()
        fixed = ImageOps.exif_transpose(im).convert("RGB")
    if config is None:
        return fixed
    return _resize(fixed, config)


def _resize(im: Image.Image, cfg: PreprocessConfig) -> Image.Image:
    interp = _INTERP[cfg.interpolation]
    if cfg.resize_mode == "letterbox":
        w, h = im.size
        scale = cfg.size / max(w, h)
        new = (max(1, round(w * scale)), max(1, round(h * scale)))
        im = im.resize(new, interp)
        canvas = Image.new("RGB", (cfg.size, cfg.size), cfg.pad_color)
        canvas.paste(im, ((cfg.size - new[0]) // 2, (cfg.size - new[1]) // 2))
        return canvas
    if cfg.resize_mode == "center_crop":
        w, h = im.size
        scale = cfg.size / min(w, h)
        im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))), interp)
        w, h = im.size
        left, top = (w - cfg.size) // 2, (h - cfg.size) // 2
        return im.crop((left, top, left + cfg.size, top + cfg.size))
    raise ValueError(f"unknown resize_mode: {cfg.resize_mode}")


def exif_orientation(path: str):
    """Raw EXIF orientation tag (274), or None."""
    with Image.open(path) as im:
        return im.getexif().get(274)


def raw_size(path: str) -> tuple[int, int]:
    """Stored pixel dimensions, before EXIF correction (no decode)."""
    with Image.open(path) as im:
        return im.size


def corrected_size(path: str) -> tuple[int, int]:
    """Dimensions after EXIF correction - what a human sees."""
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).size


def file_sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def iter_images(rows: list[dict], config: PreprocessConfig | None = None
                ) -> Iterator[tuple[dict, Image.Image]]:
    """Yield (manifest_row, preprocessed PIL image) pairs."""
    for r in rows:
        yield r, load_image(r["image_path"], config)


def verify_all(rows: list[dict]) -> dict:
    """Open every image once and report load status + dimensions before/after EXIF."""
    ok, failures, records = 0, [], []
    for r in rows:
        try:
            rw, rh = raw_size(r["image_path"])
            cw, ch = corrected_size(r["image_path"])
            load_image(r["image_path"])  # full decode
            ok += 1
            records.append({
                "image_path": r["image_path"], "room_id": r["room_id"],
                "property_group": r["property_group"], "sub_area": r["sub_area"],
                "final_label": r["final_label"],
                "raw_w": rw, "raw_h": rh, "corrected_w": cw, "corrected_h": ch,
                "exif_orientation": exif_orientation(r["image_path"]),
                "orientation_after": "portrait" if ch > cw else "landscape",
                "aspect_ratio_corrected": round(cw / ch, 4),
            })
        except Exception as e:  # noqa: BLE001
            failures.append((r["image_path"], repr(e)))
    return {"loaded_ok": ok, "failed": failures, "records": records}


if __name__ == "__main__":
    rows = load_manifest()
    print(f"supervised rows (CLEAN+NOT_CLEAN): {len(rows)}")
    print(f"excluded UNCERTAIN rows: {len(excluded_rows())}")
    res = verify_all(rows)
    print(f"loaded ok: {res['loaded_ok']} / {len(rows)}   failures: {len(res['failed'])}")
    for cfg in CONFIGS.values():
        im = load_image(rows[0]["image_path"], cfg)
        print(f"config {cfg.name}: sample output size {im.size}, mode {im.mode}")
