# -*- coding: utf-8 -*-
r"""STAGE A - the EXISTING frozen-embedding pipeline, retrained on 163 images.

Identical method to the previous best experiment (tile2x2_concat + StandardScaler
+ class-weighted LogisticRegression, C=1.0, threshold chosen by inner LOPO on the
training properties only). The only change is the data: 163 images / 7 properties
instead of 136 / 5.

This is the apples-to-apples "more data, same method" comparison. It does NOT
touch the v1 model, embeddings, manifest or experiment results.
"""
from __future__ import annotations

import csv
import json
import os

import numpy as np
from sklearn.preprocessing import StandardScaler

from dataset_prep import PROJECT_ROOT
from run_lopo_experiment import fit_classifier, metrics, pick_threshold

EMB = os.path.join(PROJECT_ROOT, "artifacts", "embeddings",
                   "embeddings_resnet50_384_spatial_v2.npz")
COMBINED = os.path.join(PROJECT_ROOT, "artifacts", "combined", "combined_manifest_v2.csv")
OUT_DIR = os.path.join(PROJECT_ROOT, "artifacts", "experiments_v2")
POS = "NOT_CLEAN"


def build_tile2x2_concat(E, view_names):
    idx = {v: i for i, v in enumerate(view_names)}
    return np.concatenate([E[:, idx["whole"], :]] + [E[:, idx[q], :] for q in ("q1", "q2", "q3", "q4")],
                          axis=1)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    d = np.load(EMB, allow_pickle=False)
    E = d["embedding"].astype(np.float64)
    views = [str(v) for v in d["view_names"]]
    X = build_tile2x2_concat(E, views)
    y = (d["label"] == POS).astype(int)
    groups = d["property_group"]
    with open(COMBINED, encoding="utf-8-sig", newline="") as f:
        manifest = {r["image_path"]: r for r in csv.DictReader(f)}

    print(f"X={X.shape}  positives={int(y.sum())}/{len(y)}  properties={sorted(set(groups))}")

    rows, fold_results = [], []
    oof_pred = np.full(len(y), -1, dtype=int)
    oof_prob = np.full(len(y), np.nan)

    for fold_i, held in enumerate(sorted(np.unique(groups)), start=1):
        te = groups == held
        tr = ~te
        thr, inner = pick_threshold(X[tr], y[tr], groups[tr], "class_weighted")
        scaler = StandardScaler().fit(X[tr])
        clf = fit_classifier(scaler.transform(X[tr]), y[tr], "class_weighted")
        prob = clf.predict_proba(scaler.transform(X[te]))[:, 1]
        pred = (prob >= thr).astype(int)
        oof_pred[te], oof_prob[te] = pred, prob

        m = metrics(y[te], pred)
        m.update({"fold": fold_i, "held_out_property": held, "threshold": thr,
                  "inner_macro_f1": inner, "n_train": int(tr.sum())})
        fold_results.append(m)
        print("  fold %d %-4s test=%3d pos=%2d thr=%.2f acc=%.3f bal=%.3f NCrec=%.3f"
              % (fold_i, held, m["n"], m["n_pos"], thr, m["accuracy"],
                 m["balanced_accuracy"], m["notclean_recall"]))

        for i in np.where(te)[0]:
            mr = manifest[d["image_path"][i]]
            rows.append({
                "model": "frozen_resnet50_tile2x2_logreg_v2",
                "room_id": d["room_id"][i], "property_group": d["property_group"][i],
                "sub_area": d["sub_area"][i], "dataset_version": d["dataset_version"][i],
                "image_file": d["image_file"][i], "image_path": d["image_path"][i],
                "human_label": d["label"][i],
                "model_prediction": POS if oof_pred[i] == 1 else "CLEAN",
                "p_not_clean": round(float(oof_prob[i]), 6),
                "threshold": thr, "fold": fold_i,
                "correct_or_error": ("correct" if oof_pred[i] == y[i] else
                                     ("false_positive" if oof_pred[i] == 1 else "false_negative")),
                "human_reason": mr["human_reason"],
            })

    agg = metrics(y, oof_pred)
    agg.update({"model": "frozen_resnet50_tile2x2_logreg_v2",
                "n_images": len(y), "n_properties": len(set(groups)),
                "thresholds_per_fold": {r["held_out_property"]: r["threshold"] for r in fold_results}})

    with open(os.path.join(OUT_DIR, "lopo_v2_frozen_predictions.csv"), "w",
              newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open(os.path.join(OUT_DIR, "lopo_v2_frozen_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"aggregate": agg, "per_fold": fold_results}, f, indent=2)

    print("\nAGGREGATE (7-property LOPO, 163 images)")
    for k in ("accuracy", "balanced_accuracy", "clean_precision", "clean_recall",
              "notclean_precision", "notclean_recall", "notclean_f1", "macro_f1"):
        print(f"  {k:<22} {agg[k]:.4f}")
    print(f"  confusion TN={agg['tn']} FP={agg['fp']} FN={agg['fn']} TP={agg['tp']}")
    print(f"\nwrote {OUT_DIR}")


if __name__ == "__main__":
    main()
