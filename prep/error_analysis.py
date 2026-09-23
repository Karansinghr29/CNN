# -*- coding: utf-8 -*-
"""Error analysis over the LOPO out-of-fold predictions.

Evidence subtypes are derived from MANIFEST METADATA for analysis only.
They were never features and never touched training.
"""
from __future__ import annotations

import collections
import csv
import os

PRED = r"D:\data science\CNN\artifacts\experiments\lopo_predictions_error_analysis.csv"


def subtype(row: dict) -> str:
    """Categorise a NOT_CLEAN image by the evidence its label rests on."""
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

    configs = sorted({(r["embedding_config"], r["classifier_variant"]) for r in rows})

    print("=" * 96)
    print("NOT_CLEAN RECALL BY EVIDENCE SUBTYPE (out-of-fold)")
    print("=" * 96)
    subs = sorted({r["subtype"] for r in rows if r["actual_label"] == "NOT_CLEAN"})
    print("%-26s %5s  " % ("subtype", "n") + "  ".join("%-16s" % f"{c}/{v[:8]}" for c, v in configs))
    for s in subs:
        sr = [r for r in rows if r["subtype"] == s]
        n = len({r["image_path"] for r in sr})
        cells = []
        for c, v in configs:
            g = [r for r in sr if r["embedding_config"] == c and r["classifier_variant"] == v]
            hit = sum(1 for r in g if r["predicted_label"] == "NOT_CLEAN")
            cells.append("%-16s" % f"{hit}/{len(g)} ({hit/len(g):.0%})")
        print("%-26s %5d  " % (s, n) + "  ".join(cells))

    print()
    print("=" * 96)
    print("PER-ROOM NOT_CLEAN RECALL (out-of-fold)")
    print("=" * 96)
    nc_rooms = sorted({r["room_id"] for r in rows if r["actual_label"] == "NOT_CLEAN"})
    print("%-26s %5s  " % ("room", "n") + "  ".join("%-14s" % f"{c}/{v[:8]}" for c, v in configs))
    for room in nc_rooms:
        rr = [r for r in rows if r["room_id"] == room and r["actual_label"] == "NOT_CLEAN"]
        n = len({r["image_path"] for r in rr})
        cells = []
        for c, v in configs:
            g = [r for r in rr if r["embedding_config"] == c and r["classifier_variant"] == v]
            hit = sum(1 for r in g if r["predicted_label"] == "NOT_CLEAN")
            cells.append("%-14s" % f"{hit}/{len(g)}")
        print("%-26s %5d  " % (room, n) + "  ".join(cells))

    print()
    print("=" * 96)
    print("FALSE POSITIVES by room (CLEAN images predicted NOT_CLEAN)")
    print("=" * 96)
    for c, v in configs:
        fps = [r for r in rows if r["embedding_config"] == c and r["classifier_variant"] == v
               and r["correct_or_error"] == "false_positive"]
        print(f"\n{c}/{v}: {len(fps)} false positives")
        for room, cnt in collections.Counter(r["room_id"] for r in fps).most_common():
            tot = sum(1 for r in rows if r["embedding_config"] == c and r["classifier_variant"] == v
                      and r["room_id"] == room and r["actual_label"] == "CLEAN")
            print("   %-26s %2d / %2d clean images flagged" % (room, cnt, tot))

    print()
    print("=" * 96)
    print("FALSE NEGATIVES - 384/unweighted (missed NOT_CLEAN, with the evidence that was missed)")
    print("=" * 96)
    fns = [r for r in rows if r["embedding_config"] == "384" and r["classifier_variant"] == "unweighted"
           and r["correct_or_error"] == "false_negative"]
    for r in sorted(fns, key=lambda x: (x["room_id"], x["image_file"])):
        print("  %-24s %-28s p=%.3f thr=%s [%s]" % (r["room_id"], r["image_file"],
              float(r["predicted_probability"]), r["threshold_used"], r["subtype"]))
        print("      %s" % r["evidence_reason"][:110])

    print()
    print("=" * 96)
    print("MAINTENANCE-ONLY CHECK: are CLEAN images with maintenance notes over-flagged?")
    print("=" * 96)
    for c, v in configs:
        g = [r for r in rows if r["embedding_config"] == c and r["classifier_variant"] == v
             and r["actual_label"] == "CLEAN"]
        with_m = [r for r in g if r["maintenance_note"].strip()]
        without_m = [r for r in g if not r["maintenance_note"].strip()]
        fp_w = sum(1 for r in with_m if r["predicted_label"] == "NOT_CLEAN")
        fp_wo = sum(1 for r in without_m if r["predicted_label"] == "NOT_CLEAN")
        print("  %s/%-15s  CLEAN w/ maintenance note: %2d/%2d flagged (%.0f%%) | "
              "without: %2d/%2d flagged (%.0f%%)"
              % (c, v, fp_w, len(with_m), 100*fp_w/max(1, len(with_m)),
                 fp_wo, len(without_m), 100*fp_wo/max(1, len(without_m))))

    print()
    print("=" * 96)
    print("SUB-AREA BREAKDOWN (Kitchen vs Room) - 384/unweighted")
    print("=" * 96)
    g = [r for r in rows if r["embedding_config"] == "384" and r["classifier_variant"] == "unweighted"]
    for sa in ("Room", "Kitchen"):
        s = [r for r in g if r["sub_area"] == sa]
        corr = sum(1 for r in s if r["correct_or_error"] == "correct")
        nc = [r for r in s if r["actual_label"] == "NOT_CLEAN"]
        nc_hit = sum(1 for r in nc if r["predicted_label"] == "NOT_CLEAN")
        print("  %-8s n=%3d  accuracy=%.3f  NOT_CLEAN recall=%s"
              % (sa, len(s), corr/len(s), f"{nc_hit}/{len(nc)}" if nc else "n/a"))


if __name__ == "__main__":
    main()
