# -*- coding: utf-8 -*-
r"""STEP 5 + STEP 7 - regression gate, matched comparison, error analysis."""
from __future__ import annotations

import collections
import csv
import json
import os

import numpy as np

from dataset_prep import PROJECT_ROOT

ART = os.path.join(PROJECT_ROOT, "artifacts")
V1 = os.path.join(ART, "experiments", "spatial_lopo_predictions.csv")
V2F = os.path.join(ART, "experiments_v2", "lopo_v2_frozen_predictions.csv")
V2C = os.path.join(ART, "experiments_v2", "lopo_v2_cnn_predictions.csv")
V3 = os.path.join(ART, "experiments_v3", "candidates_v3_predictions.csv")
OUT = os.path.join(ART, "experiments_v3", "candidate_comparison.json")


def load(p):
    with open(p, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def prf(rows):
    tp = sum(1 for r in rows if r["y"] == 1 and r["p"] == 1)
    fn = sum(1 for r in rows if r["y"] == 1 and r["p"] == 0)
    fp = sum(1 for r in rows if r["y"] == 0 and r["p"] == 1)
    tn = sum(1 for r in rows if r["y"] == 0 and r["p"] == 0)
    n = len(rows)
    crec = tn / (tn + fp) if tn + fp else 0
    nrec = tp / (tp + fn) if tp + fn else 0
    cprec = tn / (tn + fn) if tn + fn else 0
    nprec = tp / (tp + fp) if tp + fp else 0
    cf1 = 2 * cprec * crec / (cprec + crec) if cprec + crec else 0
    nf1 = 2 * nprec * nrec / (nprec + nrec) if nprec + nrec else 0
    return {"n": n, "accuracy": (tp + tn) / n, "balanced_accuracy": (crec + nrec) / 2,
            "clean_precision": cprec, "clean_recall": crec, "notclean_precision": nprec,
            "notclean_recall": nrec, "notclean_f1": nf1, "macro_f1": (cf1 + nf1) / 2,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def main() -> None:
    v1 = [dict(r, y=1 if r["actual_label"] == "NOT_CLEAN" else 0,
               p=1 if r["predicted_label"] == "NOT_CLEAN" else 0,
               prob=float(r["predicted_probability"]))
          for r in load(V1)
          if r["representation"] == "tile2x2_concat" and r["classifier_variant"] == "class_weighted"]
    v1_paths = {r["image_path"] for r in v1}

    v3 = load(V3)
    for r in v3:
        r["y"] = 1 if r["human_label"] == "NOT_CLEAN" else 0
        r["p"] = 1 if r["model_prediction"] == "NOT_CLEAN" else 0
        r["prob"] = float(r["p_not_clean"])
    cands = sorted({r["candidate"] for r in v3})

    older = {}
    for name, path, lab, pred in (("v2_frozen_163", V2F, "human_label", "model_prediction"),
                                  ("v2_cnn_163", V2C, "human_label", "model_prediction")):
        rows = load(path)
        for r in rows:
            r["y"] = 1 if r[lab] == "NOT_CLEAN" else 0
            r["p"] = 1 if r[pred] == "NOT_CLEAN" else 0
            r["prob"] = float(r["p_not_clean"])
        older[name] = rows

    belong16 = {r["image_path"] for r in v3
                if r["candidate"] == cands[0] and r["in_matched_16"] == "yes"}

    def det16(rows):
        return sum(1 for r in rows if r["image_path"] in belong16 and r["p"] == 1)

    print("=" * 118)
    print("REGRESSION GATE - belongings detection on the matched 16 (gate = 11/16)")
    print("=" * 118)
    hdr = ("  %-18s %-10s %7s %8s %8s %8s %8s %7s %5s %5s %5s %5s"
           % ("model", "belong16", "NC_rec", "NC_prec", "bal_acc", "macroF1", "accuracy",
              "n", "TN", "FP", "FN", "TP"))
    print(hdr)

    table = {}
    rowsets = {"v1_frozen_136": v1, "v2_frozen_163": older["v2_frozen_163"],
               "v2_cnn_163": older["v2_cnn_163"]}
    for c in cands:
        rowsets[c] = [r for r in v3 if r["candidate"] == c]

    for name, rows in rowsets.items():
        m = prf(rows)
        d = det16(rows)
        gate = "PASS" if d >= 11 else "FAIL"
        table[name] = dict(m, belongings_16=d, gate=gate)
        print("  %-18s %-10s %7.3f %8.3f %8.3f %8.3f %8.3f %7d %5d %5d %5d %5d  %s"
              % (name, f"{d}/16", m["notclean_recall"], m["notclean_precision"],
                 m["balanced_accuracy"], m["macro_f1"], m["accuracy"], m["n"],
                 m["tn"], m["fp"], m["fn"], m["tp"], gate))

    print()
    print("=" * 118)
    print("MATCHED SUBSET - the same 136 original images only (like-for-like vs v1)")
    print("=" * 118)
    print(hdr)
    matched = {}
    for name, rows in rowsets.items():
        sub = [r for r in rows if r["image_path"] in v1_paths]
        m = prf(sub)
        d = det16(sub)
        matched[name] = dict(m, belongings_16=d)
        print("  %-18s %-10s %7.3f %8.3f %8.3f %8.3f %8.3f %7d %5d %5d %5d %5d"
              % (name, f"{d}/16", m["notclean_recall"], m["notclean_precision"],
                 m["balanced_accuracy"], m["macro_f1"], m["accuracy"], m["n"],
                 m["tn"], m["fp"], m["fn"], m["tp"]))

    print()
    print("=" * 118)
    print("FP-MATCHED DIAGNOSTIC - hold false positives at v1's level on the same 136 images")
    print("  (labels are consulted here, so this is ANALYSIS ONLY - never used to pick a model)")
    print("=" * 118)
    v1_fp = prf([r for r in v1 if r["image_path"] in v1_paths])["fp"]
    print(f"  v1 false positives on 136 = {v1_fp}")
    print("  %-18s %-14s %-14s %-12s" % ("model", "thr for FP<=%d" % v1_fp, "belongings16", "NC recall"))
    for name, rows in rowsets.items():
        sub = [r for r in rows if r["image_path"] in v1_paths]
        best = None
        for t in np.round(np.arange(0.01, 0.99, 0.01), 2):
            fp = sum(1 for r in sub if r["y"] == 0 and r["prob"] >= t)
            if fp <= v1_fp:
                tp = sum(1 for r in sub if r["y"] == 1 and r["prob"] >= t)
                d = sum(1 for r in sub if r["image_path"] in belong16 and r["prob"] >= t)
                npos = sum(1 for r in sub if r["y"] == 1)
                best = (t, d, tp / npos)
                break
        if best:
            print("  %-18s %-14s %-14s %-12s"
                  % (name, f"{best[0]:.2f}", f"{best[1]}/16", f"{best[2]:.3f}"))

    print()
    print("=" * 118)
    print("STEP 7 - REMAINING BELONGINGS MISSES (candidate C_lower_f2)")
    print("=" * 118)
    c_rows = [r for r in v3 if r["candidate"] == "C_lower_f2"]
    for r in sorted(c_rows, key=lambda x: (x["room_id"], x["image_file"])):
        if r["subtype"] == "belongings_unorganized_floor_clean" and r["p"] == 0:
            print("  %-22s %-28s p=%.3f thr=%s  %s"
                  % (r["room_id"], r["image_file"], r["prob"], r["threshold"],
                     ("[matched16]" if r["in_matched_16"] == "yes" else "[new A11/A12]")))
            print(f"      {r['human_reason'][:104]}")

    print()
    print("  false positives introduced by C_lower_f2 (CLEAN flagged NOT_CLEAN):")
    fps = [r for r in c_rows if r["correct_or_error"] == "false_positive"]
    for room, cnt in collections.Counter(r["room_id"] for r in fps).most_common():
        tot = sum(1 for r in c_rows if r["room_id"] == room and r["y"] == 0)
        print(f"    {room:<26} {cnt:2d}/{tot:2d} clean images flagged")

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"overall": table, "matched_136": matched,
                   "gate": "belongings detected on the matched 16, baseline 11/16"}, f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
