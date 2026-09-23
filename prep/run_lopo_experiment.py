# -*- coding: utf-8 -*-
"""LOPO evaluation experiment: frozen ResNet-50 embeddings + logistic regression.

Strict discipline:
  * every learned transform (scaler, classifier, threshold) is fitted ONLY on the
    training properties of that fold;
  * the held-out property's labels are never inspected before prediction;
  * no model is trained on all 136 images;
  * metadata (room_id / property_group / sub_area) is used for GROUPING only,
    never as a feature.

Threshold selection: inner leave-one-property-out over the 4 training properties,
maximising macro-F1 on inner out-of-fold probabilities. Deterministic, group aware.
"""
from __future__ import annotations

import csv
import itertools
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.preprocessing import StandardScaler

from dataset_prep import PROJECT_ROOT, resolve_image_path, to_repo_relative

ART = os.path.join(PROJECT_ROOT, "artifacts")
EMB_DIR = os.path.join(ART, "embeddings")
OUT_DIR = os.path.join(ART, "experiments")
MANIFEST = os.path.join(PROJECT_ROOT, "cleanliness_manifest.csv")

POS = "NOT_CLEAN"          # positive class
NEG = "CLEAN"
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.01), 2)
CONFIGS = ("224", "384")
VARIANTS = ("unweighted", "class_weighted")


def load_embeddings(cfg: str):
    d = np.load(os.path.join(EMB_DIR, f"embeddings_resnet50_{cfg}.npz"), allow_pickle=False)
    return {
        "X": d["embedding"].astype(np.float64),
        "y": (d["final_label"] == POS).astype(int),
        "label": d["final_label"],
        "path": d["image_path"],
        "file": d["image_file"],
        "room": d["room_id"],
        "prop": d["property_group"],
        "sub": d["sub_area"],
    }


def fit_classifier(X, y, variant: str):
    """Fresh model each call. No hyperparameter search: fixed, conservative settings."""
    cw = "balanced" if variant == "class_weighted" else None
    clf = LogisticRegression(C=1.0, max_iter=5000, class_weight=cw, solver="lbfgs")
    clf.fit(X, y)
    return clf


def pick_threshold(X_tr, y_tr, groups_tr, variant: str):
    """Inner LOPO on the TRAINING properties only -> threshold maximising macro-F1."""
    oof = np.full(len(y_tr), np.nan)
    for g in np.unique(groups_tr):
        inner_tr = groups_tr != g
        inner_va = ~inner_tr
        if len(np.unique(y_tr[inner_tr])) < 2:
            continue
        sc = StandardScaler().fit(X_tr[inner_tr])
        clf = fit_classifier(sc.transform(X_tr[inner_tr]), y_tr[inner_tr], variant)
        oof[inner_va] = clf.predict_proba(sc.transform(X_tr[inner_va]))[:, 1]
    mask = ~np.isnan(oof)
    if mask.sum() == 0 or len(np.unique(y_tr[mask])) < 2:
        return 0.5, None
    best_t, best_f1 = 0.5, -1.0
    for t in THRESHOLDS:
        f1 = f1_score(y_tr[mask], (oof[mask] >= t).astype(int), average="macro", zero_division=0)
        # tie-break towards 0.5 for stability
        if f1 > best_f1 + 1e-12 or (abs(f1 - best_f1) <= 1e-12 and abs(t - 0.5) < abs(best_t - 0.5)):
            best_t, best_f1 = float(t), f1
    return best_t, float(best_f1)


def metrics(y_true, y_pred) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "n": int(len(y_true)), "n_pos": int(y_true.sum()), "n_neg": int((1 - y_true).sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "clean_precision": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "clean_recall": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "notclean_precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "notclean_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "notclean_f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    # keyed by resolved absolute path so it matches the paths stored in the .npz,
    # whether the manifest holds relative (published) or absolute (legacy) paths
    manifest = {resolve_image_path(r["image_path"]): r
                for r in csv.DictReader(open(MANIFEST, encoding="utf-8-sig"))}

    all_rows, fold_results, agg_results = [], [], []

    for cfg, variant in itertools.product(CONFIGS, VARIANTS):
        D = load_embeddings(cfg)
        X, y, groups = D["X"], D["y"], D["prop"]
        props = sorted(np.unique(groups))
        oof_pred = np.full(len(y), -1, dtype=int)
        oof_prob = np.full(len(y), np.nan)

        for fold_i, held in enumerate(props, start=1):
            te = groups == held
            tr = ~te

            thr, inner_f1 = pick_threshold(X[tr], y[tr], groups[tr], variant)

            scaler = StandardScaler().fit(X[tr])              # fitted on TRAIN only
            clf = fit_classifier(scaler.transform(X[tr]), y[tr], variant)
            prob = clf.predict_proba(scaler.transform(X[te]))[:, 1]
            pred = (prob >= thr).astype(int)

            oof_pred[te], oof_prob[te] = pred, prob

            m = metrics(y[te], pred)
            m.update({"fold": fold_i, "held_out_property": held, "threshold": thr,
                      "embedding_config": cfg, "classifier_variant": variant,
                      "inner_macro_f1": inner_f1,
                      "class_weights": ("balanced (computed on train fold)"
                                        if variant == "class_weighted" else "none")})
            fold_results.append(m)

            for idx in np.where(te)[0]:
                mr = manifest[D["path"][idx]]
                all_rows.append({
                    "image_path": to_repo_relative(D["path"][idx]), "image_file": D["file"][idx],
                    "room_id": D["room"][idx], "property_group": D["prop"][idx],
                    "sub_area": D["sub"][idx],
                    "actual_label": D["label"][idx],
                    "predicted_label": POS if oof_pred[idx] == 1 else NEG,
                    "predicted_probability": round(float(oof_prob[idx]), 6),
                    "fold": fold_i, "embedding_config": cfg,
                    "classifier_variant": variant, "threshold_used": thr,
                    "correct_or_error": ("correct" if oof_pred[idx] == y[idx] else
                                         ("false_positive" if oof_pred[idx] == 1 else "false_negative")),
                    "floor_tile_condition": mr["floor_tile_condition"],
                    "overall_neatness": mr["overall_neatness"],
                    "evidence_reason": mr["evidence_reason"],
                    "maintenance_note": mr["maintenance_note"],
                })

        a = metrics(y, oof_pred)
        a.update({"embedding_config": cfg, "classifier_variant": variant,
                  "thresholds_per_fold": {r["held_out_property"]: r["threshold"]
                                          for r in fold_results
                                          if r["embedding_config"] == cfg
                                          and r["classifier_variant"] == variant}})
        agg_results.append(a)

    # ---- outputs ----
    pred_csv = os.path.join(OUT_DIR, "lopo_predictions_error_analysis.csv")
    with open(pred_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader(); w.writerows(all_rows)
    with open(os.path.join(OUT_DIR, "lopo_fold_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(fold_results, f, indent=2)
    with open(os.path.join(OUT_DIR, "lopo_aggregate_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(agg_results, f, indent=2)

    # ---- console report ----
    print("=" * 100)
    print("AGGREGATE OUT-OF-FOLD RESULTS (each image predicted only when its property was held out)")
    print("=" * 100)
    hdr = "%-8s %-15s %5s %5s %6s %7s %7s %7s %7s %7s %7s  %s"
    print(hdr % ("config", "variant", "n", "pos", "acc", "bal_acc", "C_prec", "C_rec",
                 "NC_prec", "NC_rec", "NC_F1", "macroF1 | TN FP FN TP"))
    for a in agg_results:
        print(hdr % (a["embedding_config"], a["classifier_variant"], a["n"], a["n_pos"],
                     "%.3f" % a["accuracy"], "%.3f" % a["balanced_accuracy"],
                     "%.3f" % a["clean_precision"], "%.3f" % a["clean_recall"],
                     "%.3f" % a["notclean_precision"], "%.3f" % a["notclean_recall"],
                     "%.3f" % a["notclean_f1"],
                     "%.3f  | %d %d %d %d" % (a["macro_f1"], a["tn"], a["fp"], a["fn"], a["tp"])))

    print()
    print("=" * 100)
    print("PER-FOLD RESULTS")
    print("=" * 100)
    for cfg, variant in itertools.product(CONFIGS, VARIANTS):
        print(f"\n--- {cfg} / {variant} ---")
        print("%-5s %-6s %5s %5s %6s %7s %7s %7s %7s %7s %9s %s"
              % ("fold", "held", "n", "n_NC", "thr", "acc", "bal_acc", "NC_prec", "NC_rec",
                 "NC_F1", "macro_F1", "TN FP FN TP"))
        for r in fold_results:
            if r["embedding_config"] == cfg and r["classifier_variant"] == variant:
                print("%-5d %-6s %5d %5d %6.2f %7.3f %7.3f %7.3f %7.3f %7.3f %9.3f  %d %d %d %d"
                      % (r["fold"], r["held_out_property"], r["n"], r["n_pos"], r["threshold"],
                         r["accuracy"], r["balanced_accuracy"], r["notclean_precision"],
                         r["notclean_recall"], r["notclean_f1"], r["macro_f1"],
                         r["tn"], r["fp"], r["fn"], r["tp"]))

    print()
    print("predictions/error analysis ->", pred_csv)
    print("fold metrics ->", os.path.join(OUT_DIR, "lopo_fold_metrics.json"))
    print("aggregate metrics ->", os.path.join(OUT_DIR, "lopo_aggregate_metrics.json"))


if __name__ == "__main__":
    main()
