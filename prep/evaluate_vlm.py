# -*- coding: utf-8 -*-
r"""Score a VLM prediction file against the authoritative human labels.

Read-only. Writes metrics/report/error-analysis into artifacts/vlm_baseline/.
Refuses to score a dry-run file, which contains no predictions.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import re
from datetime import datetime, timezone

from dataset_prep import PROJECT_ROOT

OUT_DIR = os.path.join(PROJECT_ROOT, "artifacts", "vlm_baseline")

GARBAGE = re.compile(r"debris|waste|discarded|packing cover|wrapper|tissue|litter|garbage", re.I)
DIRT = re.compile(r"spill|residue|grim|stain|unwashed|greasy|soiled|crumbs|not been (?:cleaned|wiped|swept)", re.I)
KITCHEN = re.compile(r"kitchen|counter|slab|sink|vessels|fridge", re.I)


def human_subtype(sub_area, reason):
    if sub_area == "Kitchen" or KITCHEN.search(reason):
        if DIRT.search(reason) or GARBAGE.search(reason):
            return "kitchen_dirt"
    if GARBAGE.search(reason):
        return "garbage_debris"
    if DIRT.search(reason):
        return "dirty_floor_or_surface"
    return "belongings_unorganized"


def prf(pairs):
    tp = sum(1 for y, p in pairs if y == 1 and p == 1)
    fn = sum(1 for y, p in pairs if y == 1 and p == 0)
    fp = sum(1 for y, p in pairs if y == 0 and p == 1)
    tn = sum(1 for y, p in pairs if y == 0 and p == 0)
    n = len(pairs)
    crec = tn / (tn + fp) if tn + fp else 0.0
    nrec = tp / (tp + fn) if tp + fn else 0.0
    cprec = tn / (tn + fn) if tn + fn else 0.0
    nprec = tp / (tp + fp) if tp + fp else 0.0
    cf1 = 2 * cprec * crec / (cprec + crec) if cprec + crec else 0.0
    nf1 = 2 * nprec * nrec / (nprec + nrec) if nprec + nrec else 0.0
    return {"n": n, "accuracy": round((tp + tn) / n, 4) if n else None,
            "balanced_accuracy": round((crec + nrec) / 2, 4),
            "clean_precision": round(cprec, 4), "clean_recall": round(crec, 4),
            "notclean_precision": round(nprec, 4), "notclean_recall": round(nrec, 4),
            "notclean_f1": round(nf1, 4), "macro_f1": round((cf1 + nf1) / 2, 4),
            "confusion_matrix": {"TN": tn, "FP": fp, "FN": fn, "TP": tp}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.predictions, encoding="utf-8-sig")))
    if all(r["model_name"] == "dryrun" or r["backend"] == "dryrun" for r in rows):
        raise SystemExit("REFUSED: this is a dry-run file and contains no predictions.")

    errors = [r for r in rows if r["predicted_label"] == "ERROR"]
    unc = [r for r in rows if r["predicted_label"] == "UNCERTAIN"]
    scored = [r for r in rows if r["predicted_label"] in ("CLEAN", "NOT_CLEAN")]

    pairs = [(1 if r["human_label"] == "NOT_CLEAN" else 0,
              1 if r["predicted_label"] == "NOT_CLEAN" else 0) for r in scored]
    overall = prf(pairs)

    # UNCERTAIN counted as a miss (strict view)
    strict_pairs = [(1 if r["human_label"] == "NOT_CLEAN" else 0,
                     1 if r["predicted_label"] == "NOT_CLEAN" else 0)
                    for r in rows if r["predicted_label"] != "ERROR"]
    strict = prf(strict_pairs)

    for r in rows:
        r["human_subtype"] = (human_subtype(r["sub_area"], r["human_reason"])
                              if r["human_label"] == "NOT_CLEAN" else "clean")

    sub = {}
    for s in sorted({r["human_subtype"] for r in rows if r["human_label"] == "NOT_CLEAN"}):
        g = [r for r in rows if r["human_subtype"] == s]
        hit = sum(1 for r in g if r["predicted_label"] == "NOT_CLEAN")
        sub[s] = {"n": len(g), "detected": hit, "recall": round(hit / len(g), 4)}

    # reason-code agreement on the true positives
    reason_mix = collections.Counter(
        r["predicted_reason"] for r in rows
        if r["human_label"] == "NOT_CLEAN" and r["predicted_label"] == "NOT_CLEAN")

    per_room, per_prop = {}, {}
    for key, store in (("room_id", per_room), ("property_group", per_prop)):
        for v in sorted({r[key] for r in rows}):
            g = [r for r in rows if r[key] == v and r["predicted_label"] in ("CLEAN", "NOT_CLEAN")]
            if g:
                store[v] = prf([(1 if r["human_label"] == "NOT_CLEAN" else 0,
                                 1 if r["predicted_label"] == "NOT_CLEAN" else 0) for r in g])
                store[v]["uncertain"] = sum(1 for r in rows if r[key] == v
                                            and r["predicted_label"] == "UNCERTAIN")

    schema_issues = collections.Counter()
    for r in rows:
        for p in filter(None, r["schema_problems"].split("|")):
            schema_issues[p.split()[0]] += 1

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "predictions_file": os.path.basename(args.predictions),
        "model_name": rows[0]["model_name"], "backend": rows[0]["backend"],
        "prompt_version": rows[0]["prompt_version"], "split": rows[0]["split"],
        "images": len(rows), "errors": len(errors), "uncertain": len(unc),
        "scored_images": len(scored),
        "metrics_excluding_uncertain": overall,
        "metrics_uncertain_as_clean_miss": strict,
        "subtype_recall": sub,
        "reason_code_mix_on_true_positives": dict(reason_mix),
        "per_room": per_room, "per_property": per_prop,
        "schema_problem_counts": dict(schema_issues),
        "caveat": "VLM evidence text is a model-generated observation, not verified truth.",
    }
    stem = os.path.splitext(os.path.basename(args.predictions))[0]
    with open(os.path.join(OUT_DIR, stem + "_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    err_rows = [r for r in rows if r["predicted_label"] in ("CLEAN", "NOT_CLEAN")
                and r["predicted_label"] != r["human_label"]]
    if err_rows:
        with open(os.path.join(OUT_DIR, stem + "_errors.csv"), "w", newline="",
                  encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(err_rows[0].keys()))
            w.writeheader(); w.writerows(err_rows)

    print(json.dumps({k: report[k] for k in
                      ("model_name", "backend", "prompt_version", "images", "errors",
                       "uncertain", "metrics_excluding_uncertain", "subtype_recall")},
                     indent=2))
    print(f"\nwrote {stem}_metrics.json" + (f" and {stem}_errors.csv" if err_rows else ""))


if __name__ == "__main__":
    main()
