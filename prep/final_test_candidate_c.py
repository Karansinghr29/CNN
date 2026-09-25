# -*- coding: utf-8 -*-
r"""FINAL TEST - candidate C locked on the 136, evaluated once on the 27 A11/A12 images.

Isolation contract enforced in code (asserted, not assumed):
  * training data            = the original 136 labeled images only
  * threshold selection      = F2 on inner property-grouped folds of those 136 only
  * model/candidate selection = re-verified on the 136 only (see step 2)
  * the 27 A11/A12 images are loaded ONLY at the final scoring step

Also scores the unchanged v1 demo artifact on exactly the same 27 images.
Writes artifacts/final_test/. Does not touch the demo model or the app.
"""
from __future__ import annotations

import csv
import json
import os
import re
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.metrics import fbeta_score
from sklearn.preprocessing import StandardScaler

from dataset_prep import PROJECT_ROOT
from run_lopo_experiment import fit_classifier

ART = os.path.join(PROJECT_ROOT, "artifacts")
EMB_V2 = os.path.join(ART, "embeddings", "embeddings_resnet50_384_spatial_v2.npz")
COMBINED = os.path.join(ART, "combined", "combined_manifest_v2.csv")
DEMO_MODEL = os.path.join(ART, "model", "demo_cleanliness_model.joblib")
OUT_DIR = os.path.join(ART, "final_test")

CANDIDATE_C = {"views": ["whole", "lower", "q3", "q4"], "beta": 2.0}
OTHER_CANDIDATES = {
    "A_tile2x2_f2": {"views": ["whole", "q1", "q2", "q3", "q4"], "beta": 2.0},
    "B_allviews_f2": {"views": ["whole", "upper", "lower", "q1", "q2", "q3", "q4"], "beta": 2.0},
    "A0_tile2x2_f1": {"views": ["whole", "q1", "q2", "q3", "q4"], "beta": 1.0},
}
THRESHOLDS = np.round(np.arange(0.03, 0.96, 0.01), 2)
TEST_PROPERTIES = {"A11", "A12"}

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


def pick_threshold_f2(X, y, g, beta):
    oof = np.full(len(y), np.nan)
    for prop in np.unique(g):
        inner = g != prop
        if len(np.unique(y[inner])) < 2:
            continue
        sc = StandardScaler().fit(X[inner])
        clf = fit_classifier(sc.transform(X[inner]), y[inner], "class_weighted")
        oof[~inner] = clf.predict_proba(sc.transform(X[~inner]))[:, 1]
    m = ~np.isnan(oof)
    best_t, best_s = 0.5, -1.0
    for t in THRESHOLDS:
        s = fbeta_score(y[m], (oof[m] >= t).astype(int), beta=beta, zero_division=0)
        if s > best_s + 1e-12 or (abs(s - best_s) <= 1e-12 and t < best_t):
            best_t, best_s = float(t), s
    return best_t, float(best_s), oof


def prf(y, p):
    y, p = np.asarray(y), np.asarray(p)
    tp = int(((y == 1) & (p == 1)).sum()); fn = int(((y == 1) & (p == 0)).sum())
    fp = int(((y == 0) & (p == 1)).sum()); tn = int(((y == 0) & (p == 0)).sum())
    n = len(y)
    crec = tn / (tn + fp) if tn + fp else 0.0
    nrec = tp / (tp + fn) if tp + fn else 0.0
    cprec = tn / (tn + fn) if tn + fn else 0.0
    nprec = tp / (tp + fp) if tp + fp else 0.0
    cf1 = 2 * cprec * crec / (cprec + crec) if cprec + crec else 0.0
    nf1 = 2 * nprec * nrec / (nprec + nrec) if nprec + nrec else 0.0
    return {"n": n, "accuracy": (tp + tn) / n, "balanced_accuracy": (crec + nrec) / 2,
            "clean_precision": round(cprec, 4), "clean_recall": round(crec, 4),
            "notclean_precision": round(nprec, 4), "notclean_recall": round(nrec, 4),
            "notclean_f1": round(nf1, 4), "macro_f1": round((cf1 + nf1) / 2, 4),
            "confusion_matrix": {"TN": tn, "FP": fp, "FN": fn, "TP": tp}}


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    d = np.load(EMB_V2, allow_pickle=False)
    E = d["embedding"].astype(np.float64)
    views = [str(v) for v in d["view_names"]]
    props = np.array([str(x) for x in d["property_group"]])
    y_all = (d["label"] == "NOT_CLEAN").astype(int)
    paths = [str(p) for p in d["image_path"]]

    with open(COMBINED, encoding="utf-8-sig", newline="") as f:
        comb = {r["image_path"]: r for r in csv.DictReader(f)}

    is_test = np.isin(props, list(TEST_PROPERTIES))
    is_dev = ~is_test

    print("=" * 104)
    print("ISOLATION VERIFICATION")
    print("=" * 104)
    print(f"  development images (training + threshold + selection): {int(is_dev.sum())}")
    print(f"  final-test images (A11/A12, untouched)               : {int(is_test.sum())}")
    dev_props = sorted(set(props[is_dev])); test_props = sorted(set(props[is_test]))
    print(f"  development properties : {dev_props}")
    print(f"  final-test properties  : {test_props}")
    assert int(is_dev.sum()) == 136 and int(is_test.sum()) == 27
    assert not (set(dev_props) & set(test_props)), "property leak"
    assert not (set(dev_props) & TEST_PROPERTIES), "A11/A12 present in development set"
    dev_versions = {comb[p]["dataset_version"] for p in np.array(paths)[is_dev]}
    assert dev_versions == {"v1_original"}, dev_versions
    print(f"  development dataset_version values: {dev_versions}  (no v2_new_a11_a12) OK")
    print("  VERIFIED: no A11/A12 image enters training, threshold selection or model selection")

    X_dev_idx = np.where(is_dev)[0]
    X_test_idx = np.where(is_test)[0]
    y_dev, g_dev = y_all[X_dev_idx], props[X_dev_idx]

    # ---------- step 2: would candidate C still be chosen from the 136 alone? ----------
    print()
    print("=" * 104)
    print("SELECTION INTEGRITY - re-select the candidate using ONLY the 136 development images")
    print("=" * 104)
    v1_belong16 = [i for i in X_dev_idx
                   if comb[paths[i]]["label"] == "NOT_CLEAN"
                   and subtype(comb[paths[i]]["sub_area"], comb[paths[i]]["human_reason"])
                   == "belongings_unorganized_floor_clean"]
    print(f"  belongings positives inside the 136: {len(v1_belong16)}")
    sel = {}
    for name, spec in ({"C_lower_f2": CANDIDATE_C} | OTHER_CANDIDATES).items():
        Xc = build(E, views, spec["views"])[X_dev_idx]
        oof_pred = np.full(len(y_dev), -1)
        for prop in np.unique(g_dev):
            te = g_dev == prop
            tr = ~te
            thr, _, _ = pick_threshold_f2(Xc[tr], y_dev[tr], g_dev[tr], spec["beta"])
            sc = StandardScaler().fit(Xc[tr])
            clf = fit_classifier(sc.transform(Xc[tr]), y_dev[tr], "class_weighted")
            oof_pred[te] = (clf.predict_proba(sc.transform(Xc[te]))[:, 1] >= thr).astype(int)
        m = prf(y_dev, oof_pred)
        pos_in_dev = {paths[i] for i in v1_belong16}
        det = sum(1 for k, i in enumerate(X_dev_idx) if paths[i] in pos_in_dev and oof_pred[k] == 1)
        sel[name] = dict(m, belongings_detected=det)
        print("  %-16s belongings=%2d/%d  NCrec=%.3f NCprec=%.3f bal=%.3f macroF1=%.3f FP=%d"
              % (name, det, len(v1_belong16), m["notclean_recall"], m["notclean_precision"],
                 m["balanced_accuracy"], m["macro_f1"], m["confusion_matrix"]["FP"]))
    winner = max(sel, key=lambda k: (sel[k]["belongings_detected"], sel[k]["balanced_accuracy"]))
    print(f"  candidate selected on the 136 alone: {winner}")
    print(f"  (candidate C was selected earlier with the 27 included - this re-check is the control)")

    # ---------- lock candidate C on the 136 ----------
    print()
    print("=" * 104)
    print("LOCKING CANDIDATE C ON THE 136 DEVELOPMENT IMAGES")
    print("=" * 104)
    Xc_all = build(E, views, CANDIDATE_C["views"])
    X_dev, X_test = Xc_all[X_dev_idx], Xc_all[X_test_idx]
    thr, f2, _ = pick_threshold_f2(X_dev, y_dev, g_dev, CANDIDATE_C["beta"])
    scaler = StandardScaler().fit(X_dev)
    clf = fit_classifier(scaler.transform(X_dev), y_dev, "class_weighted")
    print(f"  views={CANDIDATE_C['views']} dim={X_dev.shape[1]}")
    print(f"  trained on {X_dev.shape[0]} images ({int(y_dev.sum())} NOT_CLEAN)")
    print(f"  threshold={thr} (F2={f2:.4f}, inner 5-property folds of the 136 only)")

    bundle_path = os.path.join(OUT_DIR, "candidate_c_locked.joblib")
    joblib.dump({"scaler": scaler, "classifier": clf, "views": CANDIDATE_C["views"],
                 "threshold": thr, "trained_on": "136 original images (v1_original) only",
                 "positive_class": "NOT_CLEAN"}, bundle_path)

    # ---------- score once on the 27 ----------
    print()
    print("=" * 104)
    print("FINAL TEST - scoring the 27 A11/A12 images ONCE")
    print("=" * 104)
    y_test = y_all[X_test_idx]
    prob_c = clf.predict_proba(scaler.transform(X_test))[:, 1]
    pred_c = (prob_c >= thr).astype(int)

    # v1 demo artifact, unchanged, on the same 27 (tile2x2_concat, its own threshold)
    demo = joblib.load(DEMO_MODEL)
    X_tile = build(E, views, ["whole", "q1", "q2", "q3", "q4"])[X_test_idx]
    prob_v1 = demo["classifier"].predict_proba(demo["scaler"].transform(X_tile))[:, 1]
    thr_v1 = float(demo["default_threshold"])
    pred_v1 = (prob_v1 >= thr_v1).astype(int)
    print(f"  v1 demo artifact threshold: {thr_v1} (unchanged)")

    m_c, m_v1 = prf(y_test, pred_c), prf(y_test, pred_v1)

    rows = []
    for k, i in enumerate(X_test_idx):
        r = comb[paths[i]]
        st = subtype(r["sub_area"], r["human_reason"]) if r["label"] == "NOT_CLEAN" else "clean"
        rows.append({
            "room_id": r["room_id"], "property_group": r["property_group"],
            "image_file": r["image_file"], "image_path": paths[i],
            "human_label": r["label"], "subtype": st,
            "candidateC_prediction": "NOT_CLEAN" if pred_c[k] else "CLEAN",
            "candidateC_p_not_clean": round(float(prob_c[k]), 6), "candidateC_threshold": thr,
            "v1demo_prediction": "NOT_CLEAN" if pred_v1[k] else "CLEAN",
            "v1demo_p_not_clean": round(float(prob_v1[k]), 6), "v1demo_threshold": thr_v1,
            "candidateC_outcome": ("correct" if pred_c[k] == y_test[k] else
                                   ("false_positive" if pred_c[k] else "false_negative")),
            "v1demo_outcome": ("correct" if pred_v1[k] == y_test[k] else
                               ("false_positive" if pred_v1[k] else "false_negative")),
            "human_reason": r["human_reason"], "split": "FINAL_TEST",
        })

    with open(os.path.join(OUT_DIR, "final_test_predictions.csv"), "w",
              newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    belong_idx = [k for k, i in enumerate(X_test_idx)
                  if comb[paths[i]]["label"] == "NOT_CLEAN"
                  and subtype(comb[paths[i]]["sub_area"], comb[paths[i]]["human_reason"])
                  == "belongings_unorganized_floor_clean"]
    bel_c = int(sum(pred_c[k] for k in belong_idx))
    bel_v1 = int(sum(pred_v1[k] for k in belong_idx))

    per_room = {}
    for room in sorted({r["room_id"] for r in rows}):
        idx = [k for k, r in enumerate(rows) if r["room_id"] == room]
        per_room[room] = {
            "n": len(idx),
            "human_clean": sum(1 for k in idx if y_test[k] == 0),
            "human_not_clean": sum(1 for k in idx if y_test[k] == 1),
            "candidateC": prf(y_test[idx], pred_c[idx]),
            "v1demo": prf(y_test[idx], pred_v1[idx]),
        }

    report = {
        "title": "Final test on 27 untouched A11/A12 images",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split_marker": "FINAL_TEST - do not reuse for training without explicit approval",
        "isolation": {
            "training_images": 136, "training_properties": dev_props,
            "test_images": 27, "test_properties": test_props,
            "test_used_in_training": False,
            "test_used_in_threshold_selection": False,
            "test_used_in_candidate_selection_this_run": False,
            "caveat": "candidate C's view set and F2 rule were originally chosen in an "
                      "earlier comparison whose metrics included these 27 images; the "
                      "selection-integrity re-check in this run repeats that choice using "
                      "the 136 alone",
        },
        "candidate_c_config": {
            "backbone": "frozen resnet50 IMAGENET1K_V2, 384 letterbox, EXIF corrected",
            "views": CANDIDATE_C["views"], "dim": int(X_dev.shape[1]),
            "scaler": "StandardScaler", "classifier": "LogisticRegression(C=1.0, "
            "class_weight=balanced, lbfgs, max_iter=5000)",
            "threshold": thr, "threshold_rule": "F2, inner 5-property folds of the 136",
        },
        "selection_integrity_on_136_only": sel,
        "selected_on_136_alone": winner,
        "final_test": {"candidate_c": m_c, "v1_demo": m_v1,
                       "belongings_positives": len(belong_idx),
                       "belongings_detected_candidate_c": bel_c,
                       "belongings_detected_v1_demo": bel_v1},
        "per_room": per_room,
    }
    with open(os.path.join(OUT_DIR, "final_test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print()
    hdr = "  %-14s %8s %9s %8s %8s %9s %8s %8s  %s"
    print(hdr % ("model", "acc", "bal_acc", "C_prec", "C_rec", "NC_prec", "NC_rec", "NC_F1", "TN/FP/FN/TP"))
    for name, m in (("candidate C", m_c), ("v1 demo", m_v1)):
        cm = m["confusion_matrix"]
        print(hdr % (name, f"{m['accuracy']:.4f}", f"{m['balanced_accuracy']:.4f}",
                     f"{m['clean_precision']:.3f}", f"{m['clean_recall']:.3f}",
                     f"{m['notclean_precision']:.3f}", f"{m['notclean_recall']:.3f}",
                     f"{m['notclean_f1']:.3f}",
                     f"{cm['TN']}/{cm['FP']}/{cm['FN']}/{cm['TP']}"))
    print(f"\n  belongings-type positives in the test set: {len(belong_idx)}")
    print(f"    candidate C detected: {bel_c}/{len(belong_idx)}")
    print(f"    v1 demo detected    : {bel_v1}/{len(belong_idx)}")

    print("\n  per room:")
    for room, d2 in per_room.items():
        print("    %-24s n=%2d (C:%d/NC:%d)  C acc=%.3f NCrec=%s | v1 acc=%.3f NCrec=%s"
              % (room, d2["n"], d2["human_clean"], d2["human_not_clean"],
                 d2["candidateC"]["accuracy"],
                 f"{d2['candidateC']['confusion_matrix']['TP']}/{d2['human_not_clean']}",
                 d2["v1demo"]["accuracy"],
                 f"{d2['v1demo']['confusion_matrix']['TP']}/{d2['human_not_clean']}"))

    print("\n  per image:")
    print("    %-22s %-28s %-10s %-22s %-22s" % ("room", "image", "human", "candidate C", "v1 demo"))
    for r in rows:
        print("    %-22s %-28s %-10s %-10s %-11s %-10s %-11s"
              % (r["room_id"][:22], r["image_file"][:28], r["human_label"],
                 r["candidateC_prediction"], f"p={r['candidateC_p_not_clean']:.3f}",
                 r["v1demo_prediction"], f"p={r['v1demo_p_not_clean']:.3f}"))

    print(f"\nwrote {OUT_DIR}")


if __name__ == "__main__":
    main()
