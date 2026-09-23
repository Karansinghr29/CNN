# -*- coding: utf-8 -*-
r"""Fit the FINAL DEMO estimator from the already-validated configuration.

This is packaging, not a new experiment. It reuses the exact functions from the
completed spatial experiment:

    build_representations()  <- run_spatial_experiment.py   (tile2x2_concat, 10240-d)
    fit_classifier()         <- run_lopo_experiment.py      (LogisticRegression,
                                                             C=1.0, max_iter=5000,
                                                             class_weight="balanced",
                                                             solver="lbfgs")
    StandardScaler()         <- sklearn, same as every fold

Difference from the experiment, stated plainly: the experiment fitted 5 fold models
on 4 properties each and never persisted them. This fits ONE estimator on all 136
labeled images so the demo has something to load. That fitted object therefore has
NO unbiased performance estimate of its own - the LOPO numbers describe the
procedure, not this object.

Writes to artifacts\model\. Touches no image, no manifest, no experiment result.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler

from dataset_prep import PROJECT_ROOT
from run_lopo_experiment import POS, fit_classifier
from run_spatial_experiment import build_representations

PROJECT = PROJECT_ROOT
EMB = os.path.join(PROJECT, "artifacts", "embeddings", "embeddings_resnet50_384_spatial.npz")
BASELINE_EMB = os.path.join(PROJECT, "artifacts", "embeddings", "embeddings_resnet50_384.npz")
METRICS = os.path.join(PROJECT, "artifacts", "experiments", "spatial_lopo_metrics.json")
OUT_DIR = os.path.join(PROJECT, "artifacts", "model")

REPRESENTATION = "tile2x2_concat"
VARIANT = "class_weighted"


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    print("=" * 84)
    print("PRE-FIT VERIFICATION - final pipeline vs completed experiment")
    print("=" * 84)

    d = np.load(EMB, allow_pickle=False)
    E = d["embedding"].astype(np.float64)
    views = [str(v) for v in d["view_names"]]
    labels = d["final_label"]
    y = (labels == POS).astype(int)

    print(f"  spatial embeddings      : {E.shape}  views={views}")
    assert E.shape == (136, 7, 2048), E.shape
    assert views == ["whole", "upper", "lower", "q1", "q2", "q3", "q4"], views

    # identical feature construction (imported, not reimplemented)
    reps = build_representations(E, views)
    X = reps[REPRESENTATION]
    print(f"  representation          : {REPRESENTATION}  dim={X.shape[1]}")
    assert X.shape == (136, 10240), X.shape

    # the whole-image slice must equal the original 384 baseline embeddings
    base = np.load(BASELINE_EMB, allow_pickle=False)["embedding"]
    dmax = float(np.abs(base - E[:, 0, :]).max())
    print(f"  whole-view vs 384 baseline max abs diff: {dmax:.8f}")
    assert dmax < 1e-3

    # labels untouched
    counts = {"CLEAN": int((labels == "CLEAN").sum()), "NOT_CLEAN": int((labels == POS).sum())}
    print(f"  label counts            : {counts}")
    assert counts == {"CLEAN": 105, "NOT_CLEAN": 31}, counts

    # thresholds actually selected per fold in the completed experiment
    m = json.load(open(METRICS, encoding="utf-8"))
    fold_thr = {r["held_out_property"]: r["threshold"] for r in m["per_fold"]
                if r["representation"] == REPRESENTATION and r["classifier_variant"] == VARIANT}
    agg = [a for a in m["aggregate"]
           if a["representation"] == REPRESENTATION and a["classifier_variant"] == VARIANT][0]
    thr_median = float(np.median(list(fold_thr.values())))
    print(f"  LOPO fold thresholds    : {fold_thr}")
    print(f"  demo default threshold  : {thr_median} (median of the five fold thresholds)")

    # hyperparameters: taken from the imported fit_classifier, not restated here
    probe = fit_classifier(np.zeros((4, 3)), np.array([0, 1, 0, 1]), VARIANT)
    hp = {k: probe.get_params()[k] for k in ("C", "max_iter", "class_weight", "solver")}
    print(f"  classifier hyperparams  : {hp}")
    assert hp == {"C": 1.0, "max_iter": 5000, "class_weight": "balanced", "solver": "lbfgs"}, hp

    print()
    print("=" * 84)
    print("FITTING DEMO ESTIMATOR ON ALL 136 LABELED IMAGES")
    print("=" * 84)
    scaler = StandardScaler().fit(X)
    clf = fit_classifier(scaler.transform(X), y, VARIANT)
    print(f"  scaler   : StandardScaler on {X.shape}")
    print(f"  classifier: {clf}")
    print(f"  class weights applied: balanced -> "
          f"{dict(zip(*np.unique(y, return_counts=True)))} (0=CLEAN, 1=NOT_CLEAN)")

    os.makedirs(OUT_DIR, exist_ok=True)
    model_path = os.path.join(OUT_DIR, "demo_cleanliness_model.joblib")
    joblib.dump({"scaler": scaler, "classifier": clf,
                 "representation": REPRESENTATION, "positive_class": POS,
                 "default_threshold": thr_median}, model_path)

    meta = {
        "artifact": "demo_cleanliness_model.joblib",
        "purpose": "PROTOTYPE DEMO ONLY - not production validated",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fitted_on": {"n_images": int(len(y)), "CLEAN": counts["CLEAN"],
                      "NOT_CLEAN": counts["NOT_CLEAN"],
                      "source": "the 136 labeled images of cleanliness_manifest.csv "
                                "(5 UNCERTAIN images excluded, labels unchanged)"},
        "pipeline": {
            "backbone": "torchvision resnet50, weights=IMAGENET1K_V2, frozen (eval, no grad)",
            "preprocessing": "EXIF transpose -> crop view -> letterbox 384x384 bicubic -> "
                             "ImageNet mean/std normalisation (prep/preprocess_config_384.json)",
            "views": "whole + q1..q4 (2x2 quadrants, 10% overlap) as defined in "
                     "extract_spatial_embeddings.VIEWS",
            "representation": f"{REPRESENTATION} = concat[whole, q1, q2, q3, q4] -> 10240-d",
            "scaler": "sklearn StandardScaler",
            "classifier": f"sklearn LogisticRegression {hp}",
        },
        "threshold": {
            "default": thr_median,
            "lopo_fold_thresholds": fold_thr,
            "note": "Default is the median of the five LOPO fold thresholds. Each fold "
                    "threshold was selected on that fold's training properties only.",
        },
        "validation_evidence": {
            "protocol": "5-fold Leave-One-Property-Out, see artifacts/experiments/spatial_lopo_metrics.json",
            "lopo_accuracy": round(agg["accuracy"], 4),
            "lopo_balanced_accuracy": round(agg["balanced_accuracy"], 4),
            "lopo_notclean_recall": round(agg["notclean_recall"], 4),
            "lopo_notclean_precision": round(agg["notclean_precision"], 4),
            "lopo_macro_f1": round(agg["macro_f1"], 4),
            "IMPORTANT": "These figures describe the LOPO PROCEDURE on 136 images from 5 "
                         "properties. They are NOT the accuracy of this fitted artifact, "
                         "which was trained on all 136 images and therefore has no "
                         "unbiased performance estimate.",
        },
        "limitations": [
            "31 NOT_CLEAN images originate from only 6 rooms.",
            "Data comes from 5 property groups, one building, one day, one camera.",
            "Small floor debris and discarded waste were poorly detected in evaluation.",
            "Not production validated. Prototype only.",
        ],
        "inputs_that_are_not_features": ["room_id", "property_group", "sub_area", "image_file"],
        "source_embeddings_sha256": sha256_file(EMB),
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
    }
    meta_path = os.path.join(OUT_DIR, "demo_cleanliness_model_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print()
    print(f"  saved: {model_path}")
    print(f"  saved: {meta_path}")

    # smoke check on the training data (NOT a performance claim)
    prob = clf.predict_proba(scaler.transform(X))[:, 1]
    pred = (prob >= thr_median).astype(int)
    print(f"\n  in-sample agreement with training labels: {(pred == y).mean():.3f} "
          f"(EXPECTED to be optimistic - fitted on these same images; not a validation metric)")


if __name__ == "__main__":
    main()
