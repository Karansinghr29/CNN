# -*- coding: utf-8 -*-
"""Step 3 - deterministic Leave-One-Property-Out fold definitions.

Writes lopo_folds.csv: one row per (fold, image), split = train|test.
No randomness, no shuffling: folds are fully determined by the property prefix.
Read-only on the dataset.
"""
from __future__ import annotations

import csv
import os

from dataset_prep import load_manifest

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lopo_folds.csv")

COLS = ["fold", "split", "property_group", "room_id", "sub_area",
        "final_label", "image_file", "image_path"]


def main() -> None:
    rows = load_manifest()                       # 136 supervised rows, UNCERTAIN excluded
    groups = sorted({r["property_group"] for r in rows})   # A33 A34 B41 C21 C31
    assert len(groups) == 5, groups

    out = []
    for i, held_out in enumerate(groups, start=1):
        for r in sorted(rows, key=lambda x: (x["property_group"], x["room_id"],
                                             x["sub_area"], x["image_file"])):
            out.append({
                "fold": i,
                "split": "test" if r["property_group"] == held_out else "train",
                "property_group": r["property_group"],
                "room_id": r["room_id"],
                "sub_area": r["sub_area"],
                "final_label": r["final_label"],
                "image_file": r["image_file"],
                "image_path": r["image_path"],
            })

    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(out)

    print(f"folds: {len(groups)} ({', '.join(groups)})")
    print(f"rows written: {len(out)}  ->  {OUT}")
    for i, held_out in enumerate(groups, start=1):
        te = [r for r in out if r["fold"] == i and r["split"] == "test"]
        tr = [r for r in out if r["fold"] == i and r["split"] == "train"]
        tn = sum(1 for r in te if r["final_label"] == "NOT_CLEAN")
        rn = sum(1 for r in tr if r["final_label"] == "NOT_CLEAN")
        print("  fold %d | test=%-4s %3d imgs (%2d NC / %2d C) | train %3d imgs (%2d NC / %2d C)"
              % (i, held_out, len(te), tn, len(te) - tn, len(tr), rn, len(tr) - rn))


if __name__ == "__main__":
    main()
