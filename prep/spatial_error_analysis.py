# -*- coding: utf-8 -*-
"""Error analysis for the spatial/tiled experiment, against the whole-image baseline.

Evidence subtypes come from manifest metadata and are used for ANALYSIS ONLY.
"""
from __future__ import annotations

import collections
import csv
import os

PRED = r"D:\data science\CNN\artifacts\experiments\spatial_lopo_predictions.csv"
DIFFICULT = {
    "A34 B double attach": "scattered belongings",
    "B41 C double common": "floor debris + kitchen dirt",
    "C31 B double attach": "scattered belongings",
    "A33 C double common": "discarded packing cover",
    "C21 A single common": "spill/residue",
    "C21 C double common": "kitchen dirt",
}


def subtype(row: dict) -> str:
    ev = row["evidence_reason"].lower()
    if row["floor_tile_condition"] == "DIRTY":
        return "floor_debris_or_spill"
    if row["sub_area"] == "Kitchen":
        return "kitchen_surface_dirt"
    if "packing cover" in ev or "discarded" in ev:
        return "discarded_waste_object"
    if "crushed empty bottle" in ev or "tissue" in ev or "wrapper" in ev:
        return "small_waste_items"
    return "scattered_belongings"


def main() -> None:
    rows = list(csv.DictReader(open(PRED, encoding="utf-8-sig")))
    for r in rows:
        r["subtype"] = subtype(r) if r["actual_label"] == "NOT_CLEAN" else "clean"

    combos = []
    for rep in ("whole", "spatial3_concat", "spatial3_mean", "tile2x2_concat", "tile2x2_mean"):
        for v in ("unweighted", "class_weighted"):
            if any(r["representation"] == rep and r["classifier_variant"] == v for r in rows):
                combos.append((rep, v))

    def sel(rep, var):
        return [r for r in rows if r["representation"] == rep and r["classifier_variant"] == var]

    print("=" * 118)
    print("NOT_CLEAN RECALL BY EVIDENCE SUBTYPE (out-of-fold)")
    print("=" * 118)
    subs = ["floor_debris_or_spill", "scattered_belongings", "kitchen_surface_dirt",
            "small_waste_items", "discarded_waste_object"]
    print("%-24s %4s  " % ("subtype", "n") + "  ".join("%-14s" % f"{r[:9]}/{v[:3]}" for r, v in combos))
    for s in subs:
        cells, n = [], None
        for rep, var in combos:
            g = [r for r in sel(rep, var) if r["subtype"] == s]
            n = len(g)
            hit = sum(1 for r in g if r["predicted_label"] == "NOT_CLEAN")
            cells.append("%-14s" % f"{hit}/{len(g)} ({hit/len(g):.0%})")
        print("%-24s %4d  " % (s, n) + "  ".join(cells))

    print()
    print("=" * 118)
    print("DIFFICULT ROOMS - NOT_CLEAN images detected (diagnostic only, all folds reported)")
    print("=" * 118)
    print("%-24s %4s  " % ("room", "n") + "  ".join("%-14s" % f"{r[:9]}/{v[:3]}" for r, v in combos))
    for room in sorted(DIFFICULT):
        cells, n = [], None
        for rep, var in combos:
            g = [r for r in sel(rep, var) if r["room_id"] == room and r["actual_label"] == "NOT_CLEAN"]
            n = len(g)
            hit = sum(1 for r in g if r["predicted_label"] == "NOT_CLEAN")
            cells.append("%-14s" % f"{hit}/{len(g)}")
        print("%-24s %4d  " % (room, n) + "  ".join(cells))

    print()
    print("=" * 118)
    print("FALSE POSITIVES (CLEAN flagged NOT_CLEAN): count, and breakdown")
    print("=" * 118)
    for rep, var in combos:
        g = sel(rep, var)
        fps = [r for r in g if r["correct_or_error"] == "false_positive"]
        cl = [r for r in g if r["actual_label"] == "CLEAN"]
        with_m = [r for r in cl if r["maintenance_note"].strip()]
        fp_m = sum(1 for r in with_m if r["predicted_label"] == "NOT_CLEAN")
        fp_nm = len(fps) - fp_m
        top = ", ".join(f"{room} {c}" for room, c in
                        collections.Counter(r["room_id"] for r in fps).most_common(3))
        print("  %-16s/%-14s FP=%2d | maintenance-note CLEAN %d/%d (%.0f%%) vs plain CLEAN %d/%d (%.0f%%) | top: %s"
              % (rep, var, len(fps), fp_m, len(with_m), 100*fp_m/max(1, len(with_m)),
                 fp_nm, len(cl)-len(with_m), 100*fp_nm/max(1, len(cl)-len(with_m)), top))

    print()
    print("=" * 118)
    print("KITCHEN vs ROOM sub-area accuracy")
    print("=" * 118)
    for rep, var in combos:
        g = sel(rep, var)
        out = []
        for sa in ("Room", "Kitchen"):
            s = [r for r in g if r["sub_area"] == sa]
            corr = sum(1 for r in s if r["correct_or_error"] == "correct")
            nc = [r for r in s if r["actual_label"] == "NOT_CLEAN"]
            hit = sum(1 for r in nc if r["predicted_label"] == "NOT_CLEAN")
            out.append(f"{sa}: acc={corr/len(s):.3f} NCrec={hit}/{len(nc)}")
        print("  %-16s/%-14s %s" % (rep, var, "  |  ".join(out)))

    print()
    print("=" * 118)
    print("PER-IMAGE FLIPS vs whole-image baseline (same variant)")
    print("=" * 118)
    for var in ("unweighted", "class_weighted"):
        base = {r["image_path"]: r for r in sel("whole", var)}
        for rep in ("spatial3_concat", "spatial3_mean", "tile2x2_concat", "tile2x2_mean"):
            cur = sel(rep, var)
            fixed = [r for r in cur if base[r["image_path"]]["correct_or_error"] != "correct"
                     and r["correct_or_error"] == "correct"]
            broke = [r for r in cur if base[r["image_path"]]["correct_or_error"] == "correct"
                     and r["correct_or_error"] != "correct"]
            fn_fixed = sum(1 for r in fixed if r["actual_label"] == "NOT_CLEAN")
            fp_fixed = sum(1 for r in fixed if r["actual_label"] == "CLEAN")
            print("  %-16s/%-14s fixed=%2d (FN->TP %d, FP->TN %d)   broke=%2d"
                  % (rep, var, len(fixed), fn_fixed, fp_fixed, len(broke)))

    print()
    print("=" * 118)
    print("REMAINING FALSE NEGATIVES - best spatial config by NC recall (unweighted)")
    print("=" * 118)
    best = max(((rep, var) for rep, var in combos if rep != "whole"),
               key=lambda rv: sum(1 for r in sel(*rv)
                                  if r["actual_label"] == "NOT_CLEAN" and r["predicted_label"] == "NOT_CLEAN"))
    print(f"  (best = {best[0]} / {best[1]})")
    for r in sorted([r for r in sel(*best) if r["correct_or_error"] == "false_negative"],
                    key=lambda x: (x["room_id"], x["image_file"])):
        print("   %-22s %-28s p=%.3f thr=%-5s [%s]" % (r["room_id"], r["image_file"],
              float(r["predicted_probability"]), r["threshold_used"], r["subtype"]))


if __name__ == "__main__":
    main()
