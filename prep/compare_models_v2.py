# -*- coding: utf-8 -*-
r"""STEP 6 + STEP 7 - old vs new comparison and evidence-subtype error analysis.

Compares, on grouped (leave-one-property-out) held-out predictions only:
  v1_frozen_136   : previous best, 136 images / 5 properties  (historical artifact)
  v2_frozen_163   : same method retrained on 163 images / 7 properties
  v2_cnn_163      : fine-tuned ResNet-50, 163 images / 7 properties

Also reports a strictly matched view: the same 136 original images, scored by
each model when that image was in its held-out property fold.

Evidence subtypes are derived from the human reason text - metadata used for
analysis only, never a model feature.
"""
from __future__ import annotations

import collections
import csv
import json
import os
import re

from dataset_prep import PROJECT_ROOT

ART = os.path.join(PROJECT_ROOT, "artifacts")
V1_PRED = os.path.join(ART, "experiments", "spatial_lopo_predictions.csv")
V1_METRICS = os.path.join(ART, "experiments", "spatial_lopo_metrics.json")
V2F_PRED = os.path.join(ART, "experiments_v2", "lopo_v2_frozen_predictions.csv")
V2F_METRICS = os.path.join(ART, "experiments_v2", "lopo_v2_frozen_metrics.json")
V2C_PRED = os.path.join(ART, "experiments_v2", "lopo_v2_cnn_predictions.csv")
V2C_METRICS = os.path.join(ART, "experiments_v2", "lopo_v2_cnn_metrics.json")
OUT_JSON = os.path.join(ART, "experiments_v2", "model_comparison_v1_vs_v2.json")
OUT_MD = os.path.join(ART, "experiments_v2", "model_comparison_v1_vs_v2.md")
OUT_ERR = os.path.join(ART, "experiments_v2", "error_analysis_subtypes.csv")

GARBAGE = re.compile(r"debris|waste|discarded|packing cover|wrapper|tissue|litter|garbage", re.I)
DIRT = re.compile(r"spill|residue|grim|stain|unwashed|greasy|soiled|crumbs|not been (?:cleaned|wiped|swept)", re.I)
KITCHEN = re.compile(r"kitchen|counter|slab|sink|vessels|fridge", re.I)
TAGS = {
    "scattered_clothes": re.compile(r"clothes|garment|quilt", re.I),
    "scattered_bags": re.compile(r"\bbags?\b|backpack|tote|suitcase|luggage", re.I),
    "shoes": re.compile(r"shoes|slipper|footwear", re.I),
    "cables_electronics": re.compile(r"cable|charger|laptop|power strip|wires?", re.I),
    "books_bottles_personal": re.compile(r"book|bottle|jar|packet|toiletries|personal items", re.I),
    "bedding": re.compile(r"bedding|blanket|quilt|mattress|bedcover|pillow", re.I),
    "packing_material": re.compile(r"packing cover|packaging|carton|box|plastic wrap", re.I),
    "garbage_debris": GARBAGE,
    "kitchen_dirt": KITCHEN,
    "dirty_floor_surface": DIRT,
}


def primary_subtype(row) -> str:
    """One bucket per NOT_CLEAN image, in priority order."""
    reason = row.get("human_reason", "")
    if row.get("sub_area") == "Kitchen" or KITCHEN.search(reason):
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


def prf(rows):
    tp = sum(1 for r in rows if r["y"] == 1 and r["p"] == 1)
    fn = sum(1 for r in rows if r["y"] == 1 and r["p"] == 0)
    fp = sum(1 for r in rows if r["y"] == 0 and r["p"] == 1)
    tn = sum(1 for r in rows if r["y"] == 0 and r["p"] == 0)
    n = len(rows)
    acc = (tp + tn) / n if n else 0.0
    crec = tn / (tn + fp) if (tn + fp) else 0.0
    nrec = tp / (tp + fn) if (tp + fn) else 0.0
    cprec = tn / (tn + fn) if (tn + fn) else 0.0
    nprec = tp / (tp + fp) if (tp + fp) else 0.0
    cf1 = 2 * cprec * crec / (cprec + crec) if (cprec + crec) else 0.0
    nf1 = 2 * nprec * nrec / (nprec + nrec) if (nprec + nrec) else 0.0
    return {"n": n, "accuracy": acc, "balanced_accuracy": (crec + nrec) / 2,
            "clean_precision": cprec, "clean_recall": crec,
            "notclean_precision": nprec, "notclean_recall": nrec, "notclean_f1": nf1,
            "macro_f1": (cf1 + nf1) / 2, "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def normalise(rows, label_col, pred_col):
    out = []
    for r in rows:
        out.append(dict(r,
                        y=1 if r[label_col] == "NOT_CLEAN" else 0,
                        p=1 if r[pred_col] == "NOT_CLEAN" else 0))
    return out


def main() -> None:
    # v1: best config only
    v1 = [r for r in load(V1_PRED)
          if r["representation"] == "tile2x2_concat" and r["classifier_variant"] == "class_weighted"]
    v1 = normalise(v1, "actual_label", "predicted_label")
    for r in v1:
        r["human_reason"] = r.get("evidence_reason", "")
    v2f = normalise(load(V2F_PRED), "human_label", "model_prediction")
    v2c = normalise(load(V2C_PRED), "human_label", "model_prediction")

    models = {"v1_frozen_136": v1, "v2_frozen_163": v2f, "v2_cnn_163": v2c}
    overall = {k: prf(v) for k, v in models.items()}

    # strictly matched: the same 136 original images
    v1_paths = {r["image_path"] for r in v1}
    matched = {
        "v1_frozen_136": prf(v1),
        "v2_frozen_163": prf([r for r in v2f if r["image_path"] in v1_paths]),
        "v2_cnn_163": prf([r for r in v2c if r["image_path"] in v1_paths]),
    }
    new_only = {
        "v2_frozen_163": prf([r for r in v2f if r["dataset_version"] == "v2_new_a11_a12"]),
        "v2_cnn_163": prf([r for r in v2c if r["dataset_version"] == "v2_new_a11_a12"]),
    }

    # subtypes
    for name, rows in models.items():
        for r in rows:
            r["subtype"] = primary_subtype(r) if r["y"] == 1 else "clean"
    subtypes = sorted({r["subtype"] for rows in models.values() for r in rows if r["y"] == 1})
    sub_table = {}
    for s in subtypes:
        sub_table[s] = {}
        for name, rows in models.items():
            g = [r for r in rows if r["subtype"] == s]
            hit = sum(1 for r in g if r["p"] == 1)
            sub_table[s][name] = {"n": len(g), "detected": hit,
                                  "recall": (hit / len(g)) if g else None}

    # per-fold NC recall
    per_fold = {}
    for name, path in (("v2_frozen_163", V2F_METRICS), ("v2_cnn_163", V2C_METRICS)):
        if os.path.exists(path):
            m = json.load(open(path, encoding="utf-8"))
            per_fold[name] = {f["held_out_property"]: {
                "n": f["n"], "n_pos": f["n_pos"], "accuracy": round(f["accuracy"], 4),
                "balanced_accuracy": round(f["balanced_accuracy"], 4),
                "notclean_recall": round(f["notclean_recall"], 4),
                "threshold": f["threshold"]} for f in m["per_fold"]}
    v1m = json.load(open(V1_METRICS, encoding="utf-8"))
    per_fold["v1_frozen_136"] = {f["held_out_property"]: {
        "n": f["n"], "n_pos": f["n_pos"], "accuracy": round(f["accuracy"], 4),
        "balanced_accuracy": round(f["balanced_accuracy"], 4),
        "notclean_recall": round(f["notclean_recall"], 4), "threshold": f["threshold"]}
        for f in v1m["per_fold"]
        if f["representation"] == "tile2x2_concat" and f["classifier_variant"] == "class_weighted"}

    # per-image error analysis csv
    err_rows = []
    for name, rows in models.items():
        for r in rows:
            tags = [t for t, rx in TAGS.items() if rx.search(r.get("human_reason", ""))]
            err_rows.append({
                "model": name, "room_id": r["room_id"], "property_group": r["property_group"],
                "image_file": r["image_file"], "image_path": r["image_path"],
                "human_label": "NOT_CLEAN" if r["y"] == 1 else "CLEAN",
                "model_prediction": "NOT_CLEAN" if r["p"] == 1 else "CLEAN",
                "outcome": ("correct" if r["y"] == r["p"] else
                            ("false_positive" if r["p"] == 1 else "false_negative")),
                "primary_subtype": r["subtype"], "evidence_tags": "|".join(sorted(tags)),
                "p_not_clean": r.get("p_not_clean", r.get("predicted_probability", "")),
                "human_reason": r.get("human_reason", ""),
            })
    with open(OUT_ERR, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(err_rows[0].keys()))
        w.writeheader(); w.writerows(err_rows)

    result = {"overall_all_images": overall, "matched_original_136": matched,
              "new_27_only": new_only, "subtype_recall": sub_table, "per_fold": per_fold}
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    # console
    def row(label, m):
        return ("  %-18s %6.3f %8.3f %8.3f %7.3f %8.3f %7.3f %7.3f %8.3f  %3d %3d %3d %3d"
                % (label, m["accuracy"], m["balanced_accuracy"], m["clean_precision"],
                   m["clean_recall"], m["notclean_precision"], m["notclean_recall"],
                   m["notclean_f1"], m["macro_f1"], m["tn"], m["fp"], m["fn"], m["tp"]))

    hdr = ("  %-18s %6s %8s %8s %7s %8s %7s %7s %8s  %3s %3s %3s %3s"
           % ("model", "acc", "bal_acc", "C_prec", "C_rec", "NC_prec", "NC_rec", "NC_F1",
              "macroF1", "TN", "FP", "FN", "TP"))
    print("=" * 110)
    print("OVERALL (each model on its own grouped held-out predictions)")
    print("=" * 110); print(hdr)
    for k in ("v1_frozen_136", "v2_frozen_163", "v2_cnn_163"):
        print(row(k, overall[k]))

    print()
    print("=" * 110)
    print("MATCHED SUBSET - the same 136 original images, held-out predictions only")
    print("=" * 110); print(hdr)
    for k in ("v1_frozen_136", "v2_frozen_163", "v2_cnn_163"):
        print(row(k, matched[k]))

    print()
    print("=" * 110)
    print("NEW 27 A11/A12 IMAGES ONLY (held out as their own property folds)")
    print("=" * 110); print(hdr)
    for k in ("v2_frozen_163", "v2_cnn_163"):
        print(row(k, new_only[k]))
    print("  note: v1 has no LOPO prediction for these - it was externally validated on them")

    print()
    print("=" * 110)
    print("NOT_CLEAN RECALL BY EVIDENCE SUBTYPE")
    print("=" * 110)
    print("  %-40s %-16s %-16s %-16s" % ("subtype", "v1_frozen_136", "v2_frozen_163", "v2_cnn_163"))
    for s, d in sub_table.items():
        cells = []
        for k in ("v1_frozen_136", "v2_frozen_163", "v2_cnn_163"):
            e = d[k]
            cells.append("%-16s" % (f"{e['detected']}/{e['n']}"
                                    + (f" ({e['recall']:.0%})" if e["n"] else "")))
        print("  %-40s %s" % (s, "".join(cells)))

    print()
    print("per-fold NOT_CLEAN recall:")
    for name in ("v1_frozen_136", "v2_frozen_163", "v2_cnn_163"):
        if name in per_fold:
            cells = " ".join(f"{p}={d['notclean_recall']:.2f}({d['n_pos']})"
                             for p, d in sorted(per_fold[name].items()))
            print(f"  {name:<16} {cells}")

    print(f"\nwrote {OUT_JSON}\nwrote {OUT_ERR}")


if __name__ == "__main__":
    main()
