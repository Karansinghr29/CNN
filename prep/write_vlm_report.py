# -*- coding: utf-8 -*-
r"""Turn a scored VLM run into the named deliverables for the dev baseline.

Read-only on the dataset. Copies the runner output to the requested filenames and
writes a markdown report from the metrics JSON produced by evaluate_vlm.py.
Nothing existing outside artifacts/vlm_baseline/ is touched.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil

from dataset_prep import PROJECT_ROOT

OUT_DIR = os.path.join(PROJECT_ROOT, "artifacts", "vlm_baseline")


def cm(d):
    return (f"|            | pred CLEAN | pred NOT_CLEAN |\n"
            f"|---|---|---|\n"
            f"| **true CLEAN** | {d['TN']} | {d['FP']} |\n"
            f"| **true NOT_CLEAN** | {d['FN']} | {d['TP']} |\n")


def table(rows, headers):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default="vlm_predictions_dev_qwen2_5vl3b",
                    help="stem of the runner output files")
    ap.add_argument("--out-prefix", default="vlm_qwen2_5vl3b")
    args = ap.parse_args()

    src_csv = os.path.join(OUT_DIR, args.stem + ".csv")
    src_jsonl = os.path.join(OUT_DIR, args.stem + ".jsonl")
    src_cfg = os.path.join(OUT_DIR, args.stem + "_runconfig.json")
    src_metrics = os.path.join(OUT_DIR, args.stem + "_metrics.json")
    src_errors = os.path.join(OUT_DIR, args.stem + "_errors.csv")
    for p in (src_csv, src_jsonl, src_cfg, src_metrics):
        if not os.path.exists(p):
            raise SystemExit(
                f"missing {os.path.basename(p)} - run the runner and evaluate_vlm first")

    p = args.out_prefix
    shutil.copyfile(src_csv, os.path.join(OUT_DIR, f"{p}_dev_predictions.csv"))
    shutil.copyfile(src_jsonl, os.path.join(OUT_DIR, f"{p}_dev_predictions.jsonl"))
    shutil.copyfile(src_cfg, os.path.join(OUT_DIR, f"{p}_run_config.json"))
    shutil.copyfile(src_metrics, os.path.join(OUT_DIR, f"{p}_dev_metrics.json"))

    m = json.load(open(src_metrics, encoding="utf-8"))
    cfg = json.load(open(src_cfg, encoding="utf-8"))
    rows = list(csv.DictReader(open(src_csv, encoding="utf-8-sig")))

    # error-analysis table: every disagreement, in the requested column order
    err_cols = ["image_file", "room_id", "property_group", "human_label",
                "vlm_prediction", "predicted_reason", "evidence_reason",
                "error_type", "human_reason"]
    err_rows = []
    for r in rows:
        pl, hl = r["predicted_label"], r["human_label"]
        if pl == "ERROR":
            etype = "call_failed"
        elif pl == "UNCERTAIN":
            etype = "abstained_UNCERTAIN"
        elif pl == hl:
            continue
        elif hl == "NOT_CLEAN":
            etype = "false_negative_missed_NOT_CLEAN"
        else:
            etype = "false_positive_flagged_CLEAN_room"
        err_rows.append({"image_file": r["image_file"], "room_id": r["room_id"],
                         "property_group": r["property_group"], "human_label": hl,
                         "vlm_prediction": pl, "predicted_reason": r["predicted_reason"],
                         "evidence_reason": r["evidence_reason"], "error_type": etype,
                         "human_reason": r["human_reason"]})
    err_path = os.path.join(OUT_DIR, f"{p}_error_analysis.csv")
    with open(err_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=err_cols)
        w.writeheader()
        w.writerows(err_rows)

    o, s = m["metrics_excluding_uncertain"], m["metrics_uncertain_as_clean_miss"]
    secs = [float(r["seconds"]) for r in rows if r["seconds"]]
    md = [f"# VLM baseline - {m['model_name']} - development set\n",
          "Zero-shot local VLM, run once with a fixed prompt. **Development set only.** The "
          "27 A11/A12 images remain locked and were not touched.\n",
          "## Run configuration\n",
          table([["model", m["model_name"]],
                 ["backend", f"{m['backend']} (local, no data left the machine)"],
                 ["prompt version", m["prompt_version"]],
                 ["temperature", cfg["temperature"]],
                 ["max image edge", f"{cfg['max_image_edge_px']} px"],
                 ["images", m["images"]], ["call failures", m["errors"]],
                 ["wall clock", f"{cfg['wall_seconds'] / 3600:.2f} h"],
                 ["mean per image", f"{sum(secs) / len(secs):.1f} s" if secs else "n/a"],
                 ["run date (UTC)", cfg["created_utc"]]], ["field", "value"]),
          "\n## Headline metrics\n",
          f"Of {m['images']} images the model returned a usable CLEAN/NOT_CLEAN label on "
          f"{m['scored_images']}, abstained with UNCERTAIN on {m['uncertain']}, and failed "
          f"on {m['errors']}.\n",
          "Primary view - the images it actually decided:\n",
          table([["accuracy", o["accuracy"]], ["balanced accuracy", o["balanced_accuracy"]],
                 ["macro-F1", o["macro_f1"]], ["NOT_CLEAN recall", o["notclean_recall"]],
                 ["NOT_CLEAN precision", o["notclean_precision"]],
                 ["NOT_CLEAN F1", o["notclean_f1"]], ["CLEAN recall", o["clean_recall"]],
                 ["CLEAN precision", o["clean_precision"]], ["n", o["n"]]],
                ["metric", "value"]),
          "\nStrict view - every UNCERTAIN counted as a wrong answer:\n",
          table([["accuracy", s["accuracy"]], ["balanced accuracy", s["balanced_accuracy"]],
                 ["macro-F1", s["macro_f1"]], ["NOT_CLEAN recall", s["notclean_recall"]],
                 ["n", s["n"]]], ["metric", "value"]),
          "\n## Confusion matrix (excluding UNCERTAIN)\n", cm(o["confusion_matrix"]),
          "\n## Recall by NOT_CLEAN subtype\n",
          "Subtypes are derived from the human evidence text, so they are an approximate "
          "grouping, not a separately labelled field.\n",
          table([[k, v["n"], v["detected"], v["recall"]]
                 for k, v in m["subtype_recall"].items()],
                ["subtype", "images", "detected", "recall"]),
          "\n## Reason code chosen on correctly detected NOT_CLEAN images\n",
          table(sorted(m["reason_code_mix_on_true_positives"].items(), key=lambda x: -x[1]),
                ["predicted_reason", "count"]),
          "\n## Per property\n",
          table([[k, v["n"], v["uncertain"], v["accuracy"], v["balanced_accuracy"],
                  v["notclean_recall"]] for k, v in m["per_property"].items()],
                ["property", "scored", "uncertain", "accuracy", "balanced accuracy",
                 "NC recall"]),
          "\n## Per room\n",
          table([[k, v["n"], v["uncertain"], v["accuracy"], v["notclean_recall"]]
                 for k, v in m["per_room"].items()],
                ["room", "scored", "uncertain", "accuracy", "NC recall"]),
          "\n## Schema violations\n",
          ("The runner validates every field and records violations instead of hiding them.\n"
           + table(sorted(m["schema_problem_counts"].items(), key=lambda x: -x[1]),
                   ["violation", "count"])
           if m["schema_problem_counts"] else "None.\n"),
          "\n## Honesty notes\n",
          "- The `evidence_reason` text is a model-generated observation, not verified fact. It "
          "is stored for comparison with the human reason and must never be shown as confirmed.\n"
          "- These are development-set numbers on images whose labels were authored by the owner. "
          "They are not a held-out estimate and are not comparable to a production figure.\n"
          "- No threshold, prompt or configuration was tuned on these results.\n"
          f"- Error analysis for every disagreement: `{os.path.basename(err_path)}` "
          f"({len(err_rows)} rows).\n"]
    rep = os.path.join(OUT_DIR, f"{p}_dev_report.md")
    with open(rep, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print(f"wrote {p}_dev_predictions.csv / .jsonl")
    print(f"wrote {p}_dev_metrics.json")
    print(f"wrote {p}_run_config.json")
    print(f"wrote {p}_error_analysis.csv  ({len(err_rows)} rows)")
    print(f"wrote {p}_dev_report.md")
    if os.path.exists(src_errors):
        print(f"(evaluate_vlm also left {os.path.basename(src_errors)})")


if __name__ == "__main__":
    main()
