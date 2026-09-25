# -*- coding: utf-8 -*-
r"""STEP 1 + STEP 2 - read-only audit before any retraining.

Verifies the previous LOPO baseline from the saved experiment artifacts, and
verifies the 27 new A11/A12 images (existence, labels, duplicates, grouping).
Writes nothing.
"""
from __future__ import annotations

import collections
import csv
import hashlib
import json
import os

from PIL import Image, ImageOps

from dataset_prep import MANIFEST, PROJECT_ROOT, load_manifest

ART = os.path.join(PROJECT_ROOT, "artifacts")
SPATIAL_METRICS = os.path.join(ART, "experiments", "spatial_lopo_metrics.json")
VAL_INV = os.path.join(ART, "validation", "external_validation_inventory.csv")
VAL_REPORT = os.path.join(ART, "validation", "external_validation_report.json")

QUOTED_BASELINE = {
    "accuracy": 0.794, "balanced_accuracy": 0.753,
    "clean_precision": 0.897, "clean_recall": 0.829,
    "notclean_precision": 0.538, "notclean_recall": 0.677, "notclean_f1": 0.600,
    "macro_f1": 0.731, "tn": 87, "fp": 18, "fn": 10, "tp": 21,
}


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def dhash(path, size=8):
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("L").resize((size + 1, size), Image.LANCZOS)
    px = list(im.getdata())
    bits = 0
    for row in range(size):
        base = row * (size + 1)
        for col in range(size):
            bits = (bits << 1) | int(px[base + col] > px[base + col + 1])
    return bits


def hamming(a, b):
    return bin(a ^ b).count("1")


def main() -> None:
    print("=" * 96)
    print("STEP 1 - PREVIOUS BASELINE, VERIFIED FROM SAVED ARTIFACTS")
    print("=" * 96)
    m = json.load(open(SPATIAL_METRICS, encoding="utf-8"))
    agg = [a for a in m["aggregate"]
           if a["representation"] == "tile2x2_concat"
           and a["classifier_variant"] == "class_weighted"][0]
    print(f"  source: {os.path.relpath(SPATIAL_METRICS, PROJECT_ROOT)}")
    print(f"  config: tile2x2_concat / class_weighted  (n={agg['n']}, positives={agg['n_pos']})")
    print()
    print("  %-22s %10s %10s %s" % ("metric", "saved", "quoted", "match"))
    ok = True
    for k, q in QUOTED_BASELINE.items():
        v = agg[k]
        match = abs(v - q) <= (0.0006 if isinstance(q, float) else 0)
        ok &= match
        print("  %-22s %10s %10s %s" % (k, round(v, 4), q, "OK" if match else "MISMATCH"))
    print(f"\n  baseline verification: {'ALL QUOTED NUMBERS MATCH THE SAVED ARTIFACT' if ok else 'DISCREPANCY - see above'}")

    print()
    print("  per-fold (the five original properties):")
    for r in m["per_fold"]:
        if r["representation"] == "tile2x2_concat" and r["classifier_variant"] == "class_weighted":
            print("    fold %d %-5s n=%2d pos=%d thr=%.2f acc=%.3f bal=%.3f NCrec=%.3f"
                  % (r["fold"], r["held_out_property"], r["n"], r["n_pos"],
                     r["threshold"], r["accuracy"], r["balanced_accuracy"], r["notclean_recall"]))

    print()
    print("  external validation of that model on the 27 new images:")
    vr = json.load(open(VAL_REPORT, encoding="utf-8"))
    o = vr["overall"]
    print("    accuracy=%.4f  clean_recall=%.4f  NC_precision=%.4f  NC_recall=%.4f  cm=%s"
          % (o["accuracy"], o["clean_recall"], o["notclean_precision"],
             o["notclean_recall"], o["confusion_matrix"]))
    print("    threshold used: %s | retrained: %s"
          % (vr["model"]["threshold_used"], vr["model"]["retrained_or_refitted"]))

    print()
    print("=" * 96)
    print("STEP 2 - VERIFY THE 27 NEW IMAGES")
    print("=" * 96)
    with open(VAL_INV, encoding="utf-8-sig", newline="") as f:
        new_rows = list(csv.DictReader(f))
    print(f"  rows in validation inventory: {len(new_rows)}")

    missing = [r for r in new_rows
               if not os.path.exists(os.path.join(PROJECT_ROOT, r["image_path"].replace("/", os.sep)))]
    print(f"  paths that exist: {len(new_rows) - len(missing)}/{len(new_rows)}")
    assert not missing, missing

    labs = collections.Counter(r["human_label"] for r in new_rows)
    print(f"  labels: {dict(labs)}")
    assert set(labs) == {"CLEAN", "NOT_CLEAN"}, labs
    assert labs["CLEAN"] == 18 and labs["NOT_CLEAN"] == 9, labs
    blank = [r["image_file"] for r in new_rows if not r["human_reason"].strip()]
    print(f"  rows with a human reason: {len(new_rows) - len(blank)}/{len(new_rows)}")

    print("\n  grouping:")
    for room in sorted({r["room_id"] for r in new_rows}):
        rr = [r for r in new_rows if r["room_id"] == room]
        c = collections.Counter(r["human_label"] for r in rr)
        print("    %-24s property=%-4s n=%d  CLEAN=%d NOT_CLEAN=%d"
              % (room, rr[0]["property_group"], len(rr), c["CLEAN"], c["NOT_CLEAN"]))
    props = sorted({r["property_group"] for r in new_rows})
    print(f"    new properties: {props}")

    # original set
    orig = load_manifest(MANIFEST, supervised_only=False)
    orig_lab = [r for r in orig if r["final_label"] in ("CLEAN", "NOT_CLEAN")]
    orig_props = sorted({r["property_group"] for r in orig})
    print(f"\n  original manifest: {len(orig)} rows, {len(orig_lab)} labeled, properties {orig_props}")
    assert not (set(props) & set(orig_props)), "property overlap with the original set"
    print("  property overlap with original: NONE (A11/A12 are new properties)")

    # duplicate checks
    print("\n  duplicate checks (new 27 vs original 141, and within the new set):")
    orig_hash = {sha256(r["image_path"]): r["image_path"] for r in orig}
    new_paths = [os.path.join(PROJECT_ROOT, r["image_path"].replace("/", os.sep)) for r in new_rows]
    new_hash = {}
    exact_cross, exact_within = [], []
    for p in new_paths:
        h = sha256(p)
        if h in orig_hash:
            exact_cross.append((p, orig_hash[h]))
        if h in new_hash:
            exact_within.append((p, new_hash[h]))
        new_hash[h] = p
    print(f"    exact duplicates new-vs-original : {len(exact_cross)}")
    print(f"    exact duplicates within new set  : {len(exact_within)}")

    nh = {p: dhash(p) for p in new_paths}
    oh = {r["image_path"]: dhash(r["image_path"]) for r in orig}
    near_cross = [(os.path.basename(p), os.path.basename(q), hamming(nh[p], oh[q]))
                  for p in new_paths for q in oh if hamming(nh[p], oh[q]) <= 6]
    keys = list(nh)
    near_within = [(os.path.basename(keys[i]), os.path.basename(keys[j]),
                    hamming(nh[keys[i]], nh[keys[j]]))
                   for i in range(len(keys)) for j in range(i + 1, len(keys))
                   if hamming(nh[keys[i]], nh[keys[j]]) <= 6]
    print(f"    near-duplicates (dHash<=6) new-vs-original: {len(near_cross)}")
    for a, b, d in near_cross[:10]:
        print(f"      {a} ~ {b} (d={d})")
    print(f"    near-duplicates (dHash<=6) within new set : {len(near_within)}")
    for a, b, d in near_within[:10]:
        print(f"      {a} ~ {b} (d={d})")

    print("\n  image properties of the new set:")
    sizes, orients = collections.Counter(), collections.Counter()
    for r in new_rows:
        sizes[(int(r["corrected_w"]), int(r["corrected_h"]))] += 1
        orients[r["orientation"]] += 1
    print(f"    corrected sizes : {dict(sizes)}")
    print(f"    orientation     : {dict(orients)}")

    print()
    print("  COMBINED TOTALS IF MERGED:")
    print(f"    original labeled : {len(orig_lab)}  (CLEAN {sum(1 for r in orig_lab if r['final_label']=='CLEAN')}, "
          f"NOT_CLEAN {sum(1 for r in orig_lab if r['final_label']=='NOT_CLEAN')})")
    print(f"    new labeled      : {len(new_rows)}  (CLEAN {labs['CLEAN']}, NOT_CLEAN {labs['NOT_CLEAN']})")
    print(f"    total            : {len(orig_lab) + len(new_rows)}  "
          f"(CLEAN {sum(1 for r in orig_lab if r['final_label']=='CLEAN') + labs['CLEAN']}, "
          f"NOT_CLEAN {sum(1 for r in orig_lab if r['final_label']=='NOT_CLEAN') + labs['NOT_CLEAN']})")
    print(f"    UNCERTAIN excluded: {sum(1 for r in orig if r['final_label']=='UNCERTAIN')}")
    print(f"    property groups   : {sorted(set(orig_props) | set(props))} ({len(set(orig_props) | set(props))} groups)")


if __name__ == "__main__":
    main()
