# -*- coding: utf-8 -*-
"""Spatial/tiled LOPO experiment vs the existing 384 whole-image baseline.

Same folds, same classifier family, same threshold protocol as run_lopo_experiment.py.
Everything learned (scaler, classifier, threshold) is fitted inside the training
properties of each fold. Metadata is grouping only, never a feature.

Representations compared (all from the SAME frozen ResNet-50, 384 letterbox):
  whole            2048   - existing baseline (whole image only)
  spatial3_concat  6144   - [whole | upper | lower]
  spatial3_mean    2048   - mean(whole, upper, lower)
  tile2x2_concat  10240   - [whole | q1 | q2 | q3 | q4]
  tile2x2_mean     2048   - mean(whole, q1..q4)
"""
from __future__ import annotations

import csv
import itertools
import json
import os

import numpy as np
from sklearn.preprocessing import StandardScaler

from dataset_prep import PROJECT_ROOT, resolve_image_path, to_repo_relative
from run_lopo_experiment import (POS, fit_classifier, metrics, pick_threshold)

ART = os.path.join(PROJECT_ROOT, "artifacts")
EMB = os.path.join(ART, "embeddings", "embeddings_resnet50_384_spatial.npz")
OUT_DIR = os.path.join(ART, "experiments")
MANIFEST = os.path.join(PROJECT_ROOT, "cleanliness_manifest.csv")

VARIANTS = ("unweighted", "class_weighted")


def build_representations(E: np.ndarray, view_names: list[str]) -> dict[str, np.ndarray]:
    idx = {v: i for i, v in enumerate(view_names)}
    whole = E[:, idx["whole"], :]
    upper = E[:, idx["upper"], :]
    lower = E[:, idx["lower"], :]
    quads = [E[:, idx[q], :] for q in ("q1", "q2", "q3", "q4")]
    return {
        "whole":           whole,
        "spatial3_concat": np.concatenate([whole, upper, lower], axis=1),
        "spatial3_mean":   np.mean(np.stack([whole, upper, lower]), axis=0),
        "tile2x2_concat":  np.concatenate([whole] + quads, axis=1),
        "tile2x2_mean":    np.mean(np.stack([whole] + quads), axis=0),
    }


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    d = np.load(EMB, allow_pickle=False)
    E = d["embedding"].astype(np.float64)
    views = [str(v) for v in d["view_names"]]
    y = (d["final_label"] == POS).astype(int)
    groups = d["property_group"]
    # keyed by resolved absolute path so it matches the paths stored in the .npz
    manifest = {resolve_image_path(r["image_path"]): r
                for r in csv.DictReader(open(MANIFEST, encoding="utf-8-sig"))}

    reps = build_representations(E, views)
    props = sorted(np.unique(groups))
    rows, fold_results, agg_results = [], [], []

    for rep_name, X in reps.items():
        for variant in VARIANTS:
            oof_pred = np.full(len(y), -1, dtype=int)
            oof_prob = np.full(len(y), np.nan)

            for fold_i, held in enumerate(props, start=1):
                te = groups == held
                tr = ~te
                thr, inner_f1 = pick_threshold(X[tr], y[tr], groups[tr], variant)
                scaler = StandardScaler().fit(X[tr])
                clf = fit_classifier(scaler.transform(X[tr]), y[tr], variant)
                prob = clf.predict_proba(scaler.transform(X[te]))[:, 1]
                pred = (prob >= thr).astype(int)
                oof_pred[te], oof_prob[te] = pred, prob

                m = metrics(y[te], pred)
                m.update({"fold": fold_i, "held_out_property": held, "threshold": thr,
                          "representation": rep_name, "dim": int(X.shape[1]),
                          "classifier_variant": variant, "inner_macro_f1": inner_f1})
                fold_results.append(m)

                for i in np.where(te)[0]:
                    mr = manifest[d["image_path"][i]]
                    rows.append({
                        "image_path": to_repo_relative(d["image_path"][i]), "image_file": d["image_file"][i],
                        "room_id": d["room_id"][i], "property_group": d["property_group"][i],
                        "sub_area": d["sub_area"][i], "actual_label": d["final_label"][i],
                        "predicted_label": POS if oof_pred[i] == 1 else "CLEAN",
                        "predicted_probability": round(float(oof_prob[i]), 6),
                        "fold": fold_i, "embedding_config": "384",
                        "representation": rep_name, "dim": int(X.shape[1]),
                        "classifier_variant": variant, "threshold_used": thr,
                        "correct_or_error": ("correct" if oof_pred[i] == y[i] else
                                             ("false_positive" if oof_pred[i] == 1 else "false_negative")),
                        "floor_tile_condition": mr["floor_tile_condition"],
                        "overall_neatness": mr["overall_neatness"],
                        "evidence_reason": mr["evidence_reason"],
                        "maintenance_note": mr["maintenance_note"],
                    })

            a = metrics(y, oof_pred)
            a.update({"representation": rep_name, "dim": int(X.shape[1]),
                      "classifier_variant": variant,
                      "thresholds_per_fold": {r["held_out_property"]: r["threshold"]
                                              for r in fold_results
                                              if r["representation"] == rep_name
                                              and r["classifier_variant"] == variant}})
            agg_results.append(a)

    pred_csv = os.path.join(OUT_DIR, "spatial_lopo_predictions.csv")
    with open(pred_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open(os.path.join(OUT_DIR, "spatial_lopo_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"aggregate": agg_results, "per_fold": fold_results}, f, indent=2)

    print("=" * 112)
    print("AGGREGATE OUT-OF-FOLD  (136 images, each predicted while its property was held out)")
    print("=" * 112)
    h = "%-16s %6s %-15s %6s %7s %7s %7s %8s %7s %7s %8s  %s"
    print(h % ("representation", "dim", "variant", "acc", "bal_acc", "C_prec", "C_rec",
               "NC_prec", "NC_rec", "NC_F1", "macroF1", "TN FP FN TP"))
    for a in agg_results:
        print(h % (a["representation"], a["dim"], a["classifier_variant"],
                   "%.3f" % a["accuracy"], "%.3f" % a["balanced_accuracy"],
                   "%.3f" % a["clean_precision"], "%.3f" % a["clean_recall"],
                   "%.3f" % a["notclean_precision"], "%.3f" % a["notclean_recall"],
                   "%.3f" % a["notclean_f1"], "%.3f" % a["macro_f1"],
                   "%d %d %d %d" % (a["tn"], a["fp"], a["fn"], a["tp"])))

    print()
    print("=" * 112)
    print("PER-FOLD")
    print("=" * 112)
    for rep_name, variant in itertools.product(reps, VARIANTS):
        print(f"\n--- {rep_name} / {variant} ---")
        print("%-5s %-6s %5s %5s %6s %7s %7s %8s %7s %7s %9s  %s"
              % ("fold", "held", "n", "n_NC", "thr", "acc", "bal_acc", "NC_prec",
                 "NC_rec", "NC_F1", "macro_F1", "TN FP FN TP"))
        for r in fold_results:
            if r["representation"] == rep_name and r["classifier_variant"] == variant:
                print("%-5d %-6s %5d %5d %6.2f %7.3f %7.3f %8.3f %7.3f %7.3f %9.3f  %d %d %d %d"
                      % (r["fold"], r["held_out_property"], r["n"], r["n_pos"], r["threshold"],
                         r["accuracy"], r["balanced_accuracy"], r["notclean_precision"],
                         r["notclean_recall"], r["notclean_f1"], r["macro_f1"],
                         r["tn"], r["fp"], r["fn"], r["tp"]))

    print()
    print("predictions ->", pred_csv)
    print("metrics ->", os.path.join(OUT_DIR, "spatial_lopo_metrics.json"))


if __name__ == "__main__":
    main()
