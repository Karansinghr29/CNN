# -*- coding: utf-8 -*-
r"""STEP 3 - combined labeled dataset (v2) for retraining.

Creates a NEW artifact. The original cleanliness_manifest.csv/.jsonl are read
only and never modified. The 5 UNCERTAIN images stay excluded.

  original labeled : 136 (105 CLEAN / 31 NOT_CLEAN)   dataset_version = v1_original
  new labeled      :  27 ( 18 CLEAN /  9 NOT_CLEAN)   dataset_version = v2_new_a11_a12
  total            : 163 (123 CLEAN / 40 NOT_CLEAN)
"""
from __future__ import annotations

import collections
import csv
import json
import os
from datetime import datetime, timezone

from dataset_prep import MANIFEST, PROJECT_ROOT, load_manifest, to_repo_relative

OUT_DIR = os.path.join(PROJECT_ROOT, "artifacts", "combined")
OUT_CSV = os.path.join(OUT_DIR, "combined_manifest_v2.csv")
OUT_JSON = os.path.join(OUT_DIR, "combined_manifest_v2_meta.json")
VAL_INV = os.path.join(PROJECT_ROOT, "artifacts", "validation",
                       "external_validation_inventory.csv")

COLS = ["dataset_version", "source_dataset", "room_id", "property_group", "sub_area",
        "image_file", "image_path", "label", "human_reason", "orientation",
        "previously_used_for", "in_original_manifest"]


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []

    # --- v1: the original 136 labeled images (UNCERTAIN excluded) ---
    for r in load_manifest(MANIFEST, supervised_only=True):
        rows.append({
            "dataset_version": "v1_original",
            "source_dataset": "cleanliness_manifest.csv",
            "room_id": r["room_id"], "property_group": r["property_group"],
            "sub_area": r["sub_area"], "image_file": r["image_file"],
            "image_path": to_repo_relative(r["image_path"]),
            "label": r["final_label"], "human_reason": r["evidence_reason"],
            "orientation": "portrait",
            "previously_used_for": "LOPO training+evaluation (5 properties)",
            "in_original_manifest": "yes",
        })

    # --- v2: the 27 new A11/A12 images with owner-confirmed labels ---
    with open(VAL_INV, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                "dataset_version": "v2_new_a11_a12",
                "source_dataset": "external_validation_inventory.csv",
                "room_id": r["room_id"], "property_group": r["property_group"],
                "sub_area": "Room", "image_file": r["image_file"],
                "image_path": r["image_path"],
                "label": r["human_label"], "human_reason": r["human_reason"],
                "orientation": r["orientation"],
                "previously_used_for": "external validation of the v1 demo model",
                "in_original_manifest": "no",
            })

    assert len(rows) == 163, len(rows)
    counts = collections.Counter(r["label"] for r in rows)
    assert counts == {"CLEAN": 123, "NOT_CLEAN": 40}, counts
    for r in rows:
        assert os.path.exists(os.path.join(PROJECT_ROOT, r["image_path"].replace("/", os.sep)))
    assert len({r["image_path"] for r in rows}) == 163

    rows.sort(key=lambda r: (r["property_group"], r["room_id"], r["sub_area"], r["image_file"]))
    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader(); w.writerows(rows)

    by_prop = {}
    for p in sorted({r["property_group"] for r in rows}):
        pr = [r for r in rows if r["property_group"] == p]
        c = collections.Counter(r["label"] for r in pr)
        by_prop[p] = {"images": len(pr), "CLEAN": c["CLEAN"], "NOT_CLEAN": c["NOT_CLEAN"],
                      "rooms": sorted({r["room_id"] for r in pr}),
                      "dataset_version": sorted({r["dataset_version"] for r in pr})}

    meta = {
        "artifact": "combined_manifest_v2.csv",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "total_labeled_images": len(rows),
        "counts": dict(counts),
        "composition": {
            "v1_original": {"images": 136, "CLEAN": 105, "NOT_CLEAN": 31,
                            "note": "UNCERTAIN images (5) remain excluded and are NOT in this file"},
            "v2_new_a11_a12": {"images": 27, "CLEAN": 18, "NOT_CLEAN": 9,
                               "note": "previously used to externally validate the v1 demo model"},
        },
        "property_groups": by_prop,
        "grouping_columns_are_not_features": ["room_id", "property_group", "sub_area",
                                              "image_file", "image_path", "dataset_version"],
        "original_manifest_modified": False,
        "uncertain_images_excluded": 5,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"wrote {OUT_CSV}  rows={len(rows)}  {dict(counts)}")
    print(f"wrote {OUT_JSON}")
    print("\nper property:")
    for p, d in by_prop.items():
        print("  %-5s n=%3d  CLEAN=%3d NOT_CLEAN=%2d  rooms=%d  %s"
              % (p, d["images"], d["CLEAN"], d["NOT_CLEAN"], len(d["rooms"]),
                 ",".join(d["dataset_version"])))


if __name__ == "__main__":
    main()
