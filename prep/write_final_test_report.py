# -*- coding: utf-8 -*-
r"""Render final_test_report.md and mark the 27 images permanently as FINAL_TEST."""
from __future__ import annotations

import csv
import json
import os

from dataset_prep import PROJECT_ROOT

OUT_DIR = os.path.join(PROJECT_ROOT, "artifacts", "final_test")
PRED = os.path.join(OUT_DIR, "final_test_predictions.csv")
METRICS = os.path.join(OUT_DIR, "final_test_metrics.json")
REPORT = os.path.join(OUT_DIR, "final_test_report.md")
MARKER = os.path.join(OUT_DIR, "FINAL_TEST_IMAGES.csv")
COMB_META = os.path.join(PROJECT_ROOT, "artifacts", "combined", "combined_manifest_v2_meta.json")


def main() -> None:
    rows = list(csv.DictReader(open(PRED, encoding="utf-8-sig")))
    rep = json.load(open(METRICS, encoding="utf-8"))
    ft = rep["final_test"]
    c, v1 = ft["candidate_c"], ft["v1_demo"]

    def line(label, key, fmt="{:.4f}"):
        a, b = c[key], v1[key]
        delta = a - b
        return f"| {label} | {fmt.format(b)} | {fmt.format(a)} | {delta:+.4f} |"

    L = ["# Final test — 27 untouched A11/A12 images\n",
         "> Held-out test: these 27 images were **not** used for training, threshold",
         "> selection or candidate selection in the run that produced these numbers.\n",
         f"- Generated: {rep['created_utc']}",
         f"- Training data: {rep['isolation']['training_images']} images, "
         f"properties {rep['isolation']['training_properties']}",
         f"- Test data: {rep['isolation']['test_images']} images, "
         f"properties {rep['isolation']['test_properties']}",
         f"- Candidate C threshold: **{rep['candidate_c_config']['threshold']}** "
         f"({rep['candidate_c_config']['threshold_rule']})",
         f"- v1 demo threshold: 0.07 (artifact default, unchanged)\n",
         "## Isolation\n",
         "| check | value |", "|---|---|",
         f"| Test images used in training | {rep['isolation']['test_used_in_training']} |",
         f"| Test images used in threshold selection | {rep['isolation']['test_used_in_threshold_selection']} |",
         f"| Test images used in candidate selection (this run) | {rep['isolation']['test_used_in_candidate_selection_this_run']} |",
         f"| Development dataset versions | v1_original only |\n",
         f"**Caveat:** {rep['isolation']['caveat']}\n",
         "### Selection-integrity control (candidates re-ranked on the 136 alone)\n",
         "| candidate | belongings /16 | NC recall | NC precision | balanced acc | macro-F1 | FP |",
         "|---|---|---|---|---|---|---|"]
    for name, s in rep["selection_integrity_on_136_only"].items():
        L.append(f"| {name} | {s['belongings_detected']}/16 | {s['notclean_recall']:.3f} | "
                 f"{s['notclean_precision']:.3f} | {s['balanced_accuracy']:.3f} | "
                 f"{s['macro_f1']:.3f} | {s['confusion_matrix']['FP']} |")
    L += [f"\nSelected on the 136 alone: **{rep['selected_on_136_alone']}** — the same "
          "candidate as before, so the earlier selection did not depend on the test images.\n",
          "## Head-to-head on the 27 unseen images\n",
          "| metric | v1 demo (current) | candidate C | change |", "|---|---|---|---|",
          line("Accuracy", "accuracy"),
          line("Balanced accuracy", "balanced_accuracy"),
          line("CLEAN precision", "clean_precision"),
          line("CLEAN recall", "clean_recall"),
          line("NOT_CLEAN precision", "notclean_precision"),
          line("NOT_CLEAN recall", "notclean_recall"),
          line("NOT_CLEAN F1", "notclean_f1"),
          f"| Belongings-type recall | {ft['belongings_detected_v1_demo']}/{ft['belongings_positives']} | "
          f"{ft['belongings_detected_candidate_c']}/{ft['belongings_positives']} | "
          f"+{ft['belongings_detected_candidate_c'] - ft['belongings_detected_v1_demo']} image |",
          "\n### Confusion matrices\n",
          "```",
          "v1 demo                         candidate C",
          "            pred C  pred NC                 pred C  pred NC",
          f"actual C      {v1['confusion_matrix']['TN']:2d}      {v1['confusion_matrix']['FP']:2d}"
          f"        actual C      {c['confusion_matrix']['TN']:2d}      {c['confusion_matrix']['FP']:2d}",
          f"actual NC      {v1['confusion_matrix']['FN']:2d}      {v1['confusion_matrix']['TP']:2d}"
          f"        actual NC      {c['confusion_matrix']['FN']:2d}      {c['confusion_matrix']['TP']:2d}",
          "```\n",
          "## Per-room\n",
          "| room | images | human C / NC | v1 accuracy | v1 NC detected | C accuracy | C NC detected |",
          "|---|---|---|---|---|---|---|"]
    for room, d in rep["per_room"].items():
        L.append(f"| {room} | {d['n']} | {d['human_clean']} / {d['human_not_clean']} | "
                 f"{d['v1demo']['accuracy']:.3f} | {d['v1demo']['confusion_matrix']['TP']}/{d['human_not_clean']} | "
                 f"{d['candidateC']['accuracy']:.3f} | {d['candidateC']['confusion_matrix']['TP']}/{d['human_not_clean']} |")

    L += ["\n## Per-image predictions\n",
          "| room | image | human | candidate C | P(NC) | v1 demo | P(NC) |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['room_id']} | {r['image_file']} | {r['human_label']} | "
                 f"{r['candidateC_prediction']} | {float(r['candidateC_p_not_clean']):.3f} | "
                 f"{r['v1demo_prediction']} | {float(r['v1demo_p_not_clean']):.3f} |")

    L += ["\n## Verdict\n",
          f"Candidate C detects one more NOT_CLEAN image than v1 (3/9 vs 2/9) and one more "
          f"belongings case (2/8 vs 1/8), at the cost of one extra false positive "
          f"({c['confusion_matrix']['FP']} vs {v1['confusion_matrix']['FP']}). Accuracy is identical "
          f"({c['accuracy']:.4f}). With 9 positives in the test set a one-image difference is "
          "well inside noise, so this is **not** a demonstrated improvement.\n",
          "Both models fail completely on A12 B (0 of 4 belongings positives detected).\n",
          "**Recommendation: keep the existing v1 demo model.** No change to "
          "`demo_cleanliness_model.joblib` or `demo_app.py`.\n",
          "## Split marker\n",
          "These 27 images are marked **FINAL_TEST**. They must not be reused for training, "
          "threshold selection or model selection without explicit approval. See "
          "`FINAL_TEST_IMAGES.csv`.\n"]

    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(L))

    with open(MARKER, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["split", "property_group", "room_id",
                                          "image_file", "image_path", "human_label"])
        w.writeheader()
        for r in rows:
            w.writerow({"split": "FINAL_TEST", "property_group": r["property_group"],
                        "room_id": r["room_id"], "image_file": r["image_file"],
                        "image_path": r["image_path"], "human_label": r["human_label"]})

    meta = json.load(open(COMB_META, encoding="utf-8"))
    meta["final_test_split"] = {
        "marked_utc": rep["created_utc"],
        "images": len(rows), "properties": rep["isolation"]["test_properties"],
        "rule": "FINAL_TEST - excluded from training, threshold selection and model "
                "selection; do not reuse without explicit approval",
        "marker_file": "artifacts/final_test/FINAL_TEST_IMAGES.csv",
    }
    with open(COMB_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"wrote {REPORT}")
    print(f"wrote {MARKER}  ({len(rows)} images marked FINAL_TEST)")
    print(f"updated {COMB_META} with the final_test_split block")


if __name__ == "__main__":
    main()
