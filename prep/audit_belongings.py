# -*- coding: utf-8 -*-
r"""STEP 1 + STEP 2 - audit the belongings-type NOT_CLEAN cases. Read-only.

Identifies the exact matched belongings positives, tabulates every model's
prediction and probability, and separates representation quality (threshold-free
ranking: ROC-AUC / average precision) from operating-point choice (threshold).
"""
from __future__ import annotations

import collections
import csv
import json
import os
import re

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from dataset_prep import PROJECT_ROOT

ART = os.path.join(PROJECT_ROOT, "artifacts")
V1 = os.path.join(ART, "experiments", "spatial_lopo_predictions.csv")
V2F = os.path.join(ART, "experiments_v2", "lopo_v2_frozen_predictions.csv")
V2C = os.path.join(ART, "experiments_v2", "lopo_v2_cnn_predictions.csv")
COMBINED = os.path.join(ART, "combined", "combined_manifest_v2.csv")
OUT = os.path.join(ART, "experiments_v2", "belongings_audit.csv")

GARBAGE = re.compile(r"debris|waste|discarded|packing cover|wrapper|tissue|litter|garbage", re.I)
DIRT = re.compile(r"spill|residue|grim|stain|unwashed|greasy|soiled|crumbs|not been (?:cleaned|wiped|swept)", re.I)
KITCHEN = re.compile(r"kitchen|counter|slab|sink|vessels|fridge", re.I)

PATTERNS = {
    "clothes": re.compile(r"clothes|garment|jacket|jeans|towel", re.I),
    "bags": re.compile(r"\bbags?\b|backpack|tote|suitcase|luggage", re.I),
    "shoes": re.compile(r"shoes|slipper|footwear", re.I),
    "cables_electronics": re.compile(r"cable|charger|laptop|power strip|wire", re.I),
    "books_bottles_small": re.compile(r"book|bottle|jar|packet|toiletries|vessel", re.I),
    "bedding": re.compile(r"bedding|blanket|quilt|mattress|bedcover|pillow", re.I),
    "boxes_packing": re.compile(r"carton|box|packaging|packing|plastic", re.I),
    "cookware": re.compile(r"cookware|vessels|racket|utensil", re.I),
}


def subtype(sub_area, reason):
    if sub_area == "Kitchen" or KITCHEN.search(reason):
        if DIRT.search(reason) or GARBAGE.search(reason):
            return "kitchen_dirt"
    if GARBAGE.search(reason):
        return "garbage_debris"
    if DIRT.search(reason):
        return "dirty_floor_or_surface"
    return "belongings_unorganized_floor_clean"


def load(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    v1 = {r["image_path"]: r for r in load(V1)
          if r["representation"] == "tile2x2_concat" and r["classifier_variant"] == "class_weighted"}
    v2f = {r["image_path"]: r for r in load(V2F)}
    v2c = {r["image_path"]: r for r in load(V2C)}
    comb = {r["image_path"]: r for r in load(COMBINED)}

    # every NOT_CLEAN image, with its subtype
    pos = [(p, r) for p, r in comb.items() if r["label"] == "NOT_CLEAN"]
    for p, r in pos:
        r["subtype"] = subtype(r["sub_area"], r["human_reason"])
    belong = [(p, r) for p, r in pos if r["subtype"] == "belongings_unorganized_floor_clean"]
    matched16 = [(p, r) for p, r in belong if p in v1]

    print("=" * 112)
    print("STEP 1 - THE MATCHED BELONGINGS POSITIVES")
    print("=" * 112)
    print(f"  all NOT_CLEAN positives (163-set)      : {len(pos)}")
    print(f"  belongings-type positives              : {len(belong)}")
    print(f"  of which in the original 136 (matched) : {len(matched16)}")
    print(f"  of which new A11/A12                    : {len(belong) - len(matched16)}")
    print()

    hdr = ("  %-26s %-5s %-9s | %-9s %-6s | %-9s %-6s | %-9s %-6s"
           % ("image", "prop", "human", "v1 pred", "p", "v2f pred", "p", "v2cnn pred", "p"))
    print(hdr)
    print("  " + "-" * 108)
    rows_out = []
    for p, r in sorted(matched16, key=lambda x: (x[1]["property_group"], x[1]["image_file"])):
        a, b, c = v1[p], v2f[p], v2c[p]
        pa = float(a["predicted_probability"]); pb = float(b["p_not_clean"]); pc = float(c["p_not_clean"])
        print("  %-26s %-5s %-9s | %-9s %.3f | %-9s %.3f | %-9s %.3f"
              % (r["image_file"][:26], r["property_group"], "NOT_CLEAN",
                 a["predicted_label"], pa, b["model_prediction"], pb, c["model_prediction"], pc))
        tags = [t for t, rx in PATTERNS.items() if rx.search(r["human_reason"])]
        rows_out.append({
            "image_file": r["image_file"], "image_path": p, "room_id": r["room_id"],
            "property_group": r["property_group"], "dataset_version": r["dataset_version"],
            "human_label": "NOT_CLEAN", "human_reason": r["human_reason"],
            "visual_pattern_tags": "|".join(sorted(tags)) or "unspecified",
            "in_matched_16": "yes",
            "v1_prediction": a["predicted_label"], "v1_probability": pa, "v1_threshold": a["threshold_used"],
            "v2frozen_prediction": b["model_prediction"], "v2frozen_probability": pb, "v2frozen_threshold": b["threshold"],
            "v2cnn_prediction": c["model_prediction"], "v2cnn_probability": pc, "v2cnn_threshold": c["threshold"],
        })

    for p, r in sorted(belong, key=lambda x: x[1]["image_file"]):
        if p in v1:
            continue
        b, c = v2f[p], v2c[p]
        tags = [t for t, rx in PATTERNS.items() if rx.search(r["human_reason"])]
        rows_out.append({
            "image_file": r["image_file"], "image_path": p, "room_id": r["room_id"],
            "property_group": r["property_group"], "dataset_version": r["dataset_version"],
            "human_label": "NOT_CLEAN", "human_reason": r["human_reason"],
            "visual_pattern_tags": "|".join(sorted(tags)) or "unspecified",
            "in_matched_16": "no",
            "v1_prediction": "", "v1_probability": "", "v1_threshold": "",
            "v2frozen_prediction": b["model_prediction"], "v2frozen_probability": float(b["p_not_clean"]),
            "v2frozen_threshold": b["threshold"],
            "v2cnn_prediction": c["model_prediction"], "v2cnn_probability": float(c["p_not_clean"]),
            "v2cnn_threshold": c["threshold"],
        })

    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader(); w.writerows(rows_out)

    det = lambda key, pred: sum(1 for r in rows_out if r["in_matched_16"] == "yes" and r[pred] == "NOT_CLEAN")
    print()
    print("  detected on the matched 16:")
    print(f"    v1 frozen  : {det('v1', 'v1_prediction')}/16")
    print(f"    v2 frozen  : {det('v2f', 'v2frozen_prediction')}/16")
    print(f"    v2 cnn     : {det('v2c', 'v2cnn_prediction')}/16")

    print()
    print("  visual pattern mix of the 16:")
    tag_count = collections.Counter()
    for r in rows_out:
        if r["in_matched_16"] == "yes":
            for t in r["visual_pattern_tags"].split("|"):
                tag_count[t] += 1
    for t, c in tag_count.most_common():
        print(f"    {t:<22} {c}")

    # ---------- threshold vs representation ----------
    print()
    print("=" * 112)
    print("DIAGNOSTIC - is the v2 drop a representation failure or a threshold shift?")
    print("=" * 112)
    print("  thresholds actually used per fold:")
    for name, src, key in (("v1 frozen", v1, "threshold_used"),
                           ("v2 frozen", v2f, "threshold"),
                           ("v2 cnn", v2c, "threshold")):
        th = sorted({float(r[key]) for r in src.values()})
        print(f"    {name:<12} {th}")

    y_all, p_v1, p_v2f, p_v2c = [], [], [], []
    for p, r in comb.items():
        if p in v1:
            y_all.append(1 if r["label"] == "NOT_CLEAN" else 0)
            p_v1.append(float(v1[p]["predicted_probability"]))
            p_v2f.append(float(v2f[p]["p_not_clean"]))
            p_v2c.append(float(v2c[p]["p_not_clean"]))
    y_all = np.array(y_all)
    print("\n  threshold-free ranking quality on the SAME 136 images (held-out probabilities):")
    for name, pr in (("v1 frozen", p_v1), ("v2 frozen", p_v2f), ("v2 cnn", p_v2c)):
        pr = np.array(pr)
        print(f"    {name:<12} ROC-AUC={roc_auc_score(y_all, pr):.3f}  "
              f"average_precision={average_precision_score(y_all, pr):.3f}")

    yb = np.ones(len(matched16))
    pb_v1 = np.array([float(v1[p]["predicted_probability"]) for p, _ in matched16])
    pb_v2f = np.array([float(v2f[p]["p_not_clean"]) for p, _ in matched16])
    pb_v2c = np.array([float(v2c[p]["p_not_clean"]) for p, _ in matched16])
    print("\n  on the matched 16 belongings positives, how many would be detected at a FIXED threshold:")
    print("    %-8s %-10s %-10s %-10s" % ("thr", "v1", "v2 frozen", "v2 cnn"))
    for t in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50):
        print("    %-8.2f %-10s %-10s %-10s"
              % (t, f"{int((pb_v1>=t).sum())}/16", f"{int((pb_v2f>=t).sum())}/16",
                 f"{int((pb_v2c>=t).sum())}/16"))
    print("\n    (this is diagnostic only - a fixed threshold picked by looking at these")
    print("     labels would be tuning on the evaluation set and is NOT used for selection)")

    # ---------- STEP 2 data inventory ----------
    print()
    print("=" * 112)
    print("STEP 2 - AVAILABLE BELONGINGS TRAINING DATA")
    print("=" * 112)
    print(f"  belongings-type positives total : {len(belong)}")
    by_prop = collections.Counter(r["property_group"] for _, r in belong)
    by_room = collections.Counter(r["room_id"] for _, r in belong)
    print(f"  properties ({len(by_prop)}) : {dict(by_prop)}")
    print(f"  rooms ({len(by_room)}) :")
    for room, c in sorted(by_room.items()):
        print(f"    {room:<26} {c}")
    print("\n  all NOT_CLEAN positives by subtype:")
    for s, c in collections.Counter(r["subtype"] for _, r in pos).most_common():
        print(f"    {s:<40} {c}")
    total_clean = sum(1 for r in comb.values() if r["label"] == "CLEAN")
    print(f"\n  overall label balance: CLEAN={total_clean}  NOT_CLEAN={len(pos)}  total={len(comb)}")

    # unlabeled images anywhere in the project?
    known = set(comb) | {r["image_path"] for r in load(
        os.path.join(PROJECT_ROOT, "cleanliness_manifest.csv"))}
    extra = []
    for root, dirs, files in os.walk(PROJECT_ROOT):
        dirs[:] = [d for d in dirs if d not in (".git", "artifacts", "prep", "__pycache__")]
        for fn in files:
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                rel = os.path.relpath(os.path.join(root, fn), PROJECT_ROOT).replace("\\", "/")
                if rel not in known:
                    extra.append(rel)
    print(f"\n  unlabeled images found in the project: {len(extra)}")
    for e in extra[:10]:
        print(f"    {e}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
