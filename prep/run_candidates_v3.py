# -*- coding: utf-8 -*-
r"""STEP 8 - targeted candidates for the belongings failure mode.

Diagnosis driving these candidates: the v2 representation ranks belongings cases
at least as well as v1 (ROC-AUC 0.802 vs 0.793), but the macro-F1 threshold rule
selected a much higher operating point (0.24-0.30 vs 0.05-0.17), which suppressed
detections. So the candidates change the OPERATING POINT and the VIEW SET, not
the architecture.

  A  tile2x2_concat (whole + q1..q4, 10240-d)          + F2 threshold rule
  B  allviews_concat (whole + upper + lower + q1..q4)  + F2 threshold rule
  C  lower-emphasis (whole + lower + q3 + q4)          + F2 threshold rule

F2 weights recall twice as heavily as precision. It is computed ONLY on inner
leave-one-property-out folds of the training properties - the test property is
never inspected when choosing the threshold.

Classifier, scaler and hyperparameters are unchanged from the validated pipeline:
StandardScaler + LogisticRegression(C=1.0, class_weight="balanced", lbfgs).
Pixels only; metadata is used for grouping and reporting, never as a feature.
"""
from __future__ import annotations

import csv
import json
import os
import re

import numpy as np
from sklearn.metrics import fbeta_score
from sklearn.preprocessing import StandardScaler

from dataset_prep import PROJECT_ROOT
from run_lopo_experiment import fit_classifier, metrics

EMB = os.path.join(PROJECT_ROOT, "artifacts", "embeddings",
                   "embeddings_resnet50_384_spatial_v2.npz")
COMBINED = os.path.join(PROJECT_ROOT, "artifacts", "combined", "combined_manifest_v2.csv")
V1_PRED = os.path.join(PROJECT_ROOT, "artifacts", "experiments", "spatial_lopo_predictions.csv")
OUT_DIR = os.path.join(PROJECT_ROOT, "artifacts", "experiments_v3")
THRESHOLDS = np.round(np.arange(0.03, 0.96, 0.01), 2)

GARBAGE = re.compile(r"debris|waste|discarded|packing cover|wrapper|tissue|litter|garbage", re.I)
DIRT = re.compile(r"spill|residue|grim|stain|unwashed|greasy|soiled|crumbs|not been (?:cleaned|wiped|swept)", re.I)
KITCHEN = re.compile(r"kitchen|counter|slab|sink|vessels|fridge", re.I)


def subtype(sub_area, reason):
    if sub_area == "Kitchen" or KITCHEN.search(reason):
        if DIRT.search(reason) or GARBAGE.search(reason):
            return "kitchen_dirt"
    if GARBAGE.search(reason):
        return "garbage_debris"
    if DIRT.search(reason):
        return "dirty_floor_or_surface"
    return "belongings_unorganized_floor_clean"


def build(E, views, spec):
    idx = {v: i for i, v in enumerate(views)}
    return np.concatenate([E[:, idx[v], :] for v in spec], axis=1)


CANDIDATES = {
    "A_tile2x2_f2":   {"views": ["whole", "q1", "q2", "q3", "q4"], "beta": 2.0},
    "B_allviews_f2":  {"views": ["whole", "upper", "lower", "q1", "q2", "q3", "q4"], "beta": 2.0},
    "C_lower_f2":     {"views": ["whole", "lower", "q3", "q4"], "beta": 2.0},
    "A0_tile2x2_f1":  {"views": ["whole", "q1", "q2", "q3", "q4"], "beta": 1.0},  # reference
}


def pick_threshold(X_tr, y_tr, g_tr, beta):
    """Inner LOPO over the training properties only; maximise F-beta of NOT_CLEAN."""
    oof = np.full(len(y_tr), np.nan)
    for g in np.unique(g_tr):
        inner = g_tr != g
        if len(np.unique(y_tr[inner])) < 2:
            continue
        sc = StandardScaler().fit(X_tr[inner])
        clf = fit_classifier(sc.transform(X_tr[inner]), y_tr[inner], "class_weighted")
        oof[~inner] = clf.predict_proba(sc.transform(X_tr[~inner]))[:, 1]
    m = ~np.isnan(oof)
    if m.sum() == 0 or len(np.unique(y_tr[m])) < 2:
        return 0.5, None
    best_t, best_s = 0.5, -1.0
    for t in THRESHOLDS:
        s = fbeta_score(y_tr[m], (oof[m] >= t).astype(int), beta=beta, zero_division=0)
        if s > best_s + 1e-12 or (abs(s - best_s) <= 1e-12 and t < best_t):
            best_t, best_s = float(t), s
    return best_t, float(best_s)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    d = np.load(EMB, allow_pickle=False)
    E = d["embedding"].astype(np.float64)
    views = [str(v) for v in d["view_names"]]
    y = (d["label"] == "NOT_CLEAN").astype(int)
    groups = d["property_group"]
    paths = list(d["image_path"])

    with open(COMBINED, encoding="utf-8-sig", newline="") as f:
        comb = {r["image_path"]: r for r in csv.DictReader(f)}
    for r in comb.values():
        r["subtype"] = subtype(r["sub_area"], r["human_reason"]) if r["label"] == "NOT_CLEAN" else "clean"

    with open(V1_PRED, encoding="utf-8-sig", newline="") as f:
        v1_paths = {r["image_path"] for r in csv.DictReader(f)
                    if r["representation"] == "tile2x2_concat"
                    and r["classifier_variant"] == "class_weighted"}

    belong_all = [p for p in paths if comb[p]["subtype"] == "belongings_unorganized_floor_clean"]
    belong_16 = [p for p in belong_all if p in v1_paths]
    print(f"belongings positives: {len(belong_all)} total, {len(belong_16)} in the matched 16\n")

    results, all_rows = {}, []
    for name, spec in CANDIDATES.items():
        X = build(E, views, spec["views"])
        oof_pred = np.full(len(y), -1, dtype=int)
        oof_prob = np.full(len(y), np.nan)
        thr_used = {}
        for held in sorted(np.unique(groups)):
            te = groups == held
            tr = ~te
            thr, inner = pick_threshold(X[tr], y[tr], groups[tr], spec["beta"])
            thr_used[held] = thr
            sc = StandardScaler().fit(X[tr])
            clf = fit_classifier(sc.transform(X[tr]), y[tr], "class_weighted")
            prob = clf.predict_proba(sc.transform(X[te]))[:, 1]
            oof_prob[te] = prob
            oof_pred[te] = (prob >= thr).astype(int)

        agg = metrics(y, oof_pred)
        det16 = sum(1 for p in belong_16 if oof_pred[paths.index(p)] == 1)
        det_all = sum(1 for p in belong_all if oof_pred[paths.index(p)] == 1)
        agg.update({"candidate": name, "views": spec["views"], "dim": int(X.shape[1]),
                    "beta": spec["beta"], "thresholds": thr_used,
                    "belongings_detected_matched16": det16,
                    "belongings_recall_matched16": round(det16 / len(belong_16), 4),
                    "belongings_detected_all24": det_all,
                    "belongings_recall_all24": round(det_all / len(belong_all), 4)})
        results[name] = agg
        print("%-16s dim=%5d thr=%s" % (name, X.shape[1],
              {k: round(v, 2) for k, v in thr_used.items()}))
        print("    belongings 16: %2d/16 (%.0f%%)   all 24: %2d/24 (%.0f%%)   "
              "NCrec=%.3f NCprec=%.3f bal=%.3f macroF1=%.3f acc=%.3f FP=%d"
              % (det16, 100 * det16 / 16, det_all, 100 * det_all / len(belong_all),
                 agg["notclean_recall"], agg["notclean_precision"],
                 agg["balanced_accuracy"], agg["macro_f1"], agg["accuracy"], agg["fp"]))

        for i, p in enumerate(paths):
            r = comb[p]
            all_rows.append({
                "candidate": name, "room_id": r["room_id"], "property_group": r["property_group"],
                "dataset_version": r["dataset_version"], "image_file": r["image_file"],
                "image_path": p, "subtype": r["subtype"], "in_matched_16": "yes" if p in belong_16 else "no",
                "human_label": r["label"],
                "model_prediction": "NOT_CLEAN" if oof_pred[i] == 1 else "CLEAN",
                "p_not_clean": round(float(oof_prob[i]), 6),
                "threshold": thr_used[groups[i]],
                "correct_or_error": ("correct" if oof_pred[i] == y[i] else
                                     ("false_positive" if oof_pred[i] == 1 else "false_negative")),
                "human_reason": r["human_reason"],
            })

    with open(os.path.join(OUT_DIR, "candidates_v3_predictions.csv"), "w",
              newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader(); w.writerows(all_rows)
    with open(os.path.join(OUT_DIR, "candidates_v3_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {OUT_DIR}")


if __name__ == "__main__":
    main()
