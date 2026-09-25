# -*- coding: utf-8 -*-
r"""Owner label correction for room "A12 B Double attached".

Rule (owner, clarified): the room is generally neat; clothes left untidily on the
cot are localized cot-level untidiness, not room-level NOT_CLEAN. All A12 B images
are CLEAN.

Scope: A12 B only. No other room is touched. No image file is touched. No model is
retrained and no previous metric file is recomputed - affected results are marked
SUPERSEDED instead.

Usage:  python prep\relabel_a12b.py --dry-run     (show before -> after)
        python prep\relabel_a12b.py               (apply)
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
from datetime import datetime, timezone

from dataset_prep import PROJECT_ROOT

ROOM = "A12 B Double attached"
NEW_LABEL = "CLEAN"
NEW_REASON_SCATTERED = ("The room is generally neat and clean, but some clothes are "
                        "scattered on the cot and are not properly arranged.")

ART = os.path.join(PROJECT_ROOT, "artifacts")
INVENTORY = os.path.join(ART, "validation", "external_validation_inventory.csv")
COMBINED = os.path.join(ART, "combined", "combined_manifest_v2.csv")
COMBINED_META = os.path.join(ART, "combined", "combined_manifest_v2_meta.json")
FINAL_MARKER = os.path.join(ART, "final_test", "FINAL_TEST_IMAGES.csv")

# files whose numbers were computed with the OLD A12 B labels
SUPERSEDED = [
    ("artifacts/validation/external_validation_predictions.csv", "external validation of the v1 demo model"),
    ("artifacts/validation/external_validation_report.json", "external validation metrics"),
    ("artifacts/validation/external_validation_report.md", "external validation report"),
    ("artifacts/experiments_v2/lopo_v2_frozen_predictions.csv", "v2 frozen LOPO predictions"),
    ("artifacts/experiments_v2/lopo_v2_frozen_metrics.json", "v2 frozen LOPO metrics"),
    ("artifacts/experiments_v2/lopo_v2_cnn_predictions.csv", "v2 CNN LOPO predictions"),
    ("artifacts/experiments_v2/lopo_v2_cnn_metrics.json", "v2 CNN LOPO metrics"),
    ("artifacts/experiments_v2/model_comparison_v1_vs_v2.json", "v1 vs v2 comparison"),
    ("artifacts/experiments_v2/error_analysis_subtypes.csv", "subtype error analysis"),
    ("artifacts/experiments_v2/belongings_audit.csv", "belongings audit (24 positives)"),
    ("artifacts/experiments_v3/candidates_v3_predictions.csv", "candidate A/B/C predictions"),
    ("artifacts/experiments_v3/candidates_v3_metrics.json", "candidate A/B/C metrics"),
    ("artifacts/experiments_v3/candidate_comparison.json", "candidate regression gate"),
    ("artifacts/final_test/final_test_predictions.csv", "final test predictions"),
    ("artifacts/final_test/final_test_metrics.json", "final test metrics"),
    ("artifacts/final_test/final_test_report.md", "final test report"),
    ("artifacts/embeddings/embeddings_resnet50_384_spatial_v2.npz",
     "embedding file carries a copy of the labels in its 'label' array (features unaffected)"),
]

NOTE_PATHS = [os.path.join(ART, d, "SUPERSEDED_A12B_RELABEL.md")
              for d in ("validation", "experiments_v2", "experiments_v3", "final_test")]


def show_changes(rows, label_key, reason_key, name):
    print(f"\n--- {name} ---")
    changes = []
    for r in rows:
        if r.get("room_id") != ROOM:
            continue
        old_label, old_reason = r[label_key], r[reason_key]
        new_reason = NEW_REASON_SCATTERED if old_label == "NOT_CLEAN" else old_reason
        changed = (old_label != NEW_LABEL) or (new_reason != old_reason)
        changes.append((r["image_file"], old_label, NEW_LABEL, old_reason, new_reason, changed))
        flag = "CHANGED" if changed else "unchanged"
        print(f"  {r['image_file']:<28} {old_label:<10} -> {NEW_LABEL:<10} [{flag}]")
        if changed:
            print(f"      before: {old_reason[:96]}")
            print(f"      after : {new_reason[:96]}")
    return changes


def apply(rows, label_key, reason_key):
    n = 0
    for r in rows:
        if r.get("room_id") != ROOM:
            continue
        if r[label_key] == "NOT_CLEAN":
            r[reason_key] = NEW_REASON_SCATTERED
            n += 1
        r[label_key] = NEW_LABEL
    return n


def rewrite(path, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    inv = list(csv.DictReader(open(INVENTORY, encoding="utf-8-sig")))
    comb = list(csv.DictReader(open(COMBINED, encoding="utf-8-sig")))
    marker = list(csv.DictReader(open(FINAL_MARKER, encoding="utf-8-sig")))

    print("=" * 100)
    print(f"BEFORE -> AFTER for room: {ROOM}")
    print("=" * 100)
    show_changes(inv, "human_label", "human_reason", "external_validation_inventory.csv")
    show_changes(comb, "label", "human_reason", "combined_manifest_v2.csv")

    # counts before
    def counts(rows, key):
        return dict(collections.Counter(r[key] for r in rows))
    before = {
        "inventory_27": counts(inv, "human_label"),
        "combined_163": counts(comb, "label"),
        "a12b": counts([r for r in inv if r["room_id"] == ROOM], "human_label"),
    }

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return

    n1 = apply(inv, "human_label", "human_reason")
    n2 = apply(comb, "label", "human_reason")
    for r in marker:
        if r["room_id"] == ROOM:
            r["human_label"] = NEW_LABEL
    rewrite(INVENTORY, inv); rewrite(COMBINED, comb); rewrite(FINAL_MARKER, marker)
    print(f"\nupdated: inventory ({n1} reasons rewritten), combined manifest ({n2}), final-test marker")

    after = {
        "inventory_27": counts(inv, "human_label"),
        "combined_163": counts(comb, "label"),
        "a12b": counts([r for r in inv if r["room_id"] == ROOM], "human_label"),
    }

    print("\n" + "=" * 100)
    print("RECALCULATED COUNTS (label files only - no model metric was recomputed)")
    print("=" * 100)
    for k in before:
        print(f"  {k:<16} before={before[k]}   after={after[k]}")

    # combined meta refresh
    meta = json.load(open(COMBINED_META, encoding="utf-8"))
    meta["counts"] = after["combined_163"]
    meta["composition"]["v2_new_a11_a12"] = {
        "images": 27, "CLEAN": after["inventory_27"].get("CLEAN", 0),
        "NOT_CLEAN": after["inventory_27"].get("NOT_CLEAN", 0),
        "note": "A12 B relabelled to CLEAN by owner rule on "
                + datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    }
    for p in meta["property_groups"]:
        rows_p = [r for r in comb if r["property_group"] == p]
        c = collections.Counter(r["label"] for r in rows_p)
        meta["property_groups"][p]["CLEAN"] = c["CLEAN"]
        meta["property_groups"][p]["NOT_CLEAN"] = c["NOT_CLEAN"]
    meta["label_corrections"] = meta.get("label_corrections", []) + [{
        "room_id": ROOM, "applied_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "change": "all images -> CLEAN",
        "rule": "localized cot-level untidiness is not room-level NOT_CLEAN",
        "images_changed": n2,
    }]
    with open(COMBINED_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"  updated {os.path.relpath(COMBINED_META, PROJECT_ROOT)}")

    # supersede notes
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    body = ["# SUPERSEDED — A12 B label correction\n",
            f"Applied: {stamp}\n",
            f"Room **{ROOM}** was relabelled: 4 images moved NOT_CLEAN -> CLEAN "
            "(localized cot-level untidiness is not room-level NOT_CLEAN).\n",
            "Every metric below was computed with the OLD A12 B labels and is therefore "
            "**superseded**. The numbers have deliberately NOT been recomputed - rerunning "
            "is a separate, explicitly approved step.\n",
            "| artifact | what it contained |", "|---|---|"]
    for path, what in SUPERSEDED:
        body.append(f"| `{path}` | {what} |")
    body += ["\n## Which conclusions are affected\n",
             "- The 27-image external validation of the v1 demo model (was 18 CLEAN / 9 NOT_CLEAN; "
             "now 22 CLEAN / 5 NOT_CLEAN). Its NOT_CLEAN recall of 2/9 no longer applies.",
             "- All v2 / v3 LOPO runs that trained or evaluated on the combined 163-image set.",
             "- The final test on the 27 images: both models scored 0/4 on A12 B belongings "
             "positives that are no longer positives.",
             "- The belongings-subtype counts (were 24 positives; A12 B contributed 4).",
             "\n## Not affected\n",
             "- The original 136-image manifest, its labels and the v1 LOPO baseline "
             "(accuracy 0.794, NOT_CLEAN recall 0.677, belongings 11/16) - no A12 B image is in it.",
             "- The deployed demo model and the Streamlit app, which were not changed.",
             "- The embedding *features*; only the label array copied inside the v2 npz is stale.\n"]
    for p in NOTE_PATHS:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write("\n".join(body))
        print(f"  wrote {os.path.relpath(p, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
