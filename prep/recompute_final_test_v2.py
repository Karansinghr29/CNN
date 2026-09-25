# -*- coding: utf-8 -*-
r"""Recompute the final-test metrics against the CORRECTED A12 B labels.

No model is run: the per-image probabilities and thresholds already stored in
final_test_predictions.csv are model outputs and do not depend on labels. Only
the ground truth changed, so only the metrics change.

Writes NEW artifacts (…_corrected_v2.*). The previous results are left in place.
"""
from __future__ import annotations

import csv
import json
import os
import re
from datetime import datetime, timezone

from dataset_prep import PROJECT_ROOT

ART = os.path.join(PROJECT_ROOT, "artifacts")
OLD_PRED = os.path.join(ART, "final_test", "final_test_predictions.csv")
OLD_METRICS = os.path.join(ART, "final_test", "final_test_metrics.json")
LABELS = os.path.join(ART, "validation", "external_validation_inventory.csv")   # authoritative
OUT_PRED = os.path.join(ART, "final_test", "final_test_predictions_corrected_v2.csv")
OUT_METRICS = os.path.join(ART, "final_test", "final_test_metrics_corrected_v2.json")
OUT_MD = os.path.join(ART, "final_test", "final_test_report_corrected_v2.md")

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
    old_rows = list(csv.DictReader(open(OLD_PRED, encoding="utf-8-sig")))
    truth = {r["image_file"]: r for r in csv.DictReader(open(LABELS, encoding="utf-8-sig"))}
    old_metrics = json.load(open(OLD_METRICS, encoding="utf-8"))

    rows, relabelled = [], []
    for r in old_rows:
        t = truth[r["image_file"]]
        new_label, new_reason = t["human_label"], t["human_reason"]
        if new_label != r["human_label"]:
            relabelled.append((r["room_id"], r["image_file"], r["human_label"], new_label))
        st = subtype("Room", new_reason) if new_label == "NOT_CLEAN" else "clean"
        y = 1 if new_label == "NOT_CLEAN" else 0
        pc = 1 if r["candidateC_prediction"] == "NOT_CLEAN" else 0
        pv = 1 if r["v1demo_prediction"] == "NOT_CLEAN" else 0
        rows.append(dict(r, human_label=new_label, human_reason=new_reason, subtype=st,
                         y=y, pc=pc, pv=pv,
                         candidateC_outcome=("correct" if pc == y else
                                             ("false_positive" if pc else "false_negative")),
                         v1demo_outcome=("correct" if pv == y else
                                         ("false_positive" if pv else "false_negative")),
                         label_source="corrected_after_A12B_relabel"))

    print("relabelled since the previous final test:")
    for room, f, a, b in relabelled:
        print(f"  {room:<24} {f:<28} {a} -> {b}")
    n_clean = sum(1 for r in rows if r["y"] == 0)
    print(f"\ncorrected test set: {n_clean} CLEAN / {len(rows) - n_clean} NOT_CLEAN "
          f"(was 18 / 9)")

    m_c = prf([(r["y"], r["pc"]) for r in rows])
    m_v1 = prf([(r["y"], r["pv"]) for r in rows])

    bel = [r for r in rows if r["subtype"] == "belongings_unorganized_floor_clean"]
    bel_c = sum(r["pc"] for r in bel)
    bel_v1 = sum(r["pv"] for r in bel)

    per_room = {}
    for room in sorted({r["room_id"] for r in rows}):
        rr = [r for r in rows if r["room_id"] == room]
        per_room[room] = {
            "n": len(rr),
            "human_clean": sum(1 for r in rr if r["y"] == 0),
            "human_not_clean": sum(1 for r in rr if r["y"] == 1),
            "candidateC": prf([(r["y"], r["pc"]) for r in rr]),
            "v1demo": prf([(r["y"], r["pv"]) for r in rr]),
        }

    # before -> after deltas
    old_c, old_v1 = old_metrics["final_test"]["candidate_c"], old_metrics["final_test"]["v1_demo"]
    deltas = {}
    for key in ("accuracy", "balanced_accuracy", "clean_precision", "clean_recall",
                "notclean_precision", "notclean_recall", "notclean_f1", "macro_f1"):
        deltas[key] = {
            "candidate_c": {"before": old_c[key], "after": m_c[key],
                            "change": round(m_c[key] - old_c[key], 4)},
            "v1_demo": {"before": old_v1[key], "after": m_v1[key],
                        "change": round(m_v1[key] - old_v1[key], 4)},
        }

    out_cols = ["room_id", "property_group", "image_file", "image_path", "human_label",
                "subtype", "candidateC_prediction", "candidateC_p_not_clean",
                "candidateC_threshold", "candidateC_outcome", "v1demo_prediction",
                "v1demo_p_not_clean", "v1demo_threshold", "v1demo_outcome",
                "human_reason", "split", "label_source"]
    with open(OUT_PRED, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=out_cols, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

    report = {
        "title": "Final test RECOMPUTED against corrected A12 B labels",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "supersedes": "artifacts/final_test/final_test_metrics.json (and final_test_report.md)",
        "models_rerun": False,
        "note": "Per-image probabilities and thresholds are unchanged; they are model "
                "outputs and do not depend on labels. Only the ground truth changed.",
        "label_correction": {
            "room": "A12 B Double attached",
            "images": [f for _, f, _, _ in relabelled],
            "change": "NOT_CLEAN -> CLEAN",
            "test_set_before": {"CLEAN": 18, "NOT_CLEAN": 9},
            "test_set_after": {"CLEAN": n_clean, "NOT_CLEAN": len(rows) - n_clean},
        },
        "thresholds": {"candidate_c": rows[0]["candidateC_threshold"],
                       "v1_demo": rows[0]["v1demo_threshold"],
                       "tuned_on_this_set": False},
        "final_test_corrected": {
            "candidate_c": m_c, "v1_demo": m_v1,
            "belongings_positives": len(bel),
            "belongings_detected_candidate_c": int(bel_c),
            "belongings_detected_v1_demo": int(bel_v1),
        },
        "before_after_deltas": deltas,
        "per_room": per_room,
        "model_selection_changed": False,
    }
    with open(OUT_METRICS, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # markdown
    L = ["# Final test — recomputed against corrected A12 B labels\n",
         f"- Generated: {report['created_utc']}",
         "- **Supersedes** `final_test_metrics.json` / `final_test_report.md`",
         "- Models were **not** re-run; thresholds unchanged "
         f"(candidate C {rows[0]['candidateC_threshold']}, v1 demo {rows[0]['v1demo_threshold']})",
         f"- Test set: **{n_clean} CLEAN / {len(rows) - n_clean} NOT_CLEAN** (was 18 / 9)\n",
         "## Head-to-head (corrected labels)\n",
         "| metric | v1 demo | candidate C |", "|---|---|---|"]
    for k in ("accuracy", "balanced_accuracy", "clean_precision", "clean_recall",
              "notclean_precision", "notclean_recall", "notclean_f1", "macro_f1"):
        L.append(f"| {k.replace('_', ' ')} | {m_v1[k]:.4f} | {m_c[k]:.4f} |")
    L.append(f"| belongings recall | {bel_v1}/{len(bel)} | {bel_c}/{len(bel)} |")
    cm_v, cm_c = m_v1["confusion_matrix"], m_c["confusion_matrix"]
    L += ["\n```",
          "v1 demo                          candidate C",
          "            pred C  pred NC                  pred C  pred NC",
          f"actual C      {cm_v['TN']:2d}      {cm_v['FP']:2d}         actual C      {cm_c['TN']:2d}      {cm_c['FP']:2d}",
          f"actual NC      {cm_v['FN']:2d}      {cm_v['TP']:2d}         actual NC      {cm_c['FN']:2d}      {cm_c['TP']:2d}",
          "```\n", "## Before → after (effect of the A12 B relabel)\n",
          "| metric | v1 before | v1 after | Δ | C before | C after | Δ |", "|---|---|---|---|---|---|---|"]
    for k, d in deltas.items():
        L.append(f"| {k.replace('_', ' ')} | {d['v1_demo']['before']:.4f} | {d['v1_demo']['after']:.4f} | "
                 f"{d['v1_demo']['change']:+.4f} | {d['candidate_c']['before']:.4f} | "
                 f"{d['candidate_c']['after']:.4f} | {d['candidate_c']['change']:+.4f} |")
    L += ["\n## Per-room\n",
          "| room | images | human C / NC | v1 acc | v1 NC detected | C acc | C NC detected |",
          "|---|---|---|---|---|---|---|"]
    for room, d in per_room.items():
        L.append(f"| {room} | {d['n']} | {d['human_clean']} / {d['human_not_clean']} | "
                 f"{d['v1demo']['accuracy']:.3f} | {d['v1demo']['confusion_matrix']['TP']}/{d['human_not_clean']} | "
                 f"{d['candidateC']['accuracy']:.3f} | {d['candidateC']['confusion_matrix']['TP']}/{d['human_not_clean']} |")
    L += ["\n## Status\n",
          "- No model was trained, re-run or replaced.",
          "- No threshold was tuned on this set.",
          "- Model selection is unchanged; the v1 demo model remains in place.\n"]
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(L))

    print("\n%-14s %8s %9s %8s %8s %9s %8s %8s %9s  %s"
          % ("model", "acc", "bal_acc", "C_prec", "C_rec", "NC_prec", "NC_rec", "NC_F1", "macroF1", "TN/FP/FN/TP"))
    for name, m in (("v1 demo", m_v1), ("candidate C", m_c)):
        cm = m["confusion_matrix"]
        print("%-14s %8.4f %9.4f %8.3f %8.3f %9.3f %8.3f %8.3f %9.3f  %d/%d/%d/%d"
              % (name, m["accuracy"], m["balanced_accuracy"], m["clean_precision"],
                 m["clean_recall"], m["notclean_precision"], m["notclean_recall"],
                 m["notclean_f1"], m["macro_f1"], cm["TN"], cm["FP"], cm["FN"], cm["TP"]))
    print(f"\nbelongings positives (corrected): {len(bel)}")
    print(f"  v1 demo detected    : {bel_v1}/{len(bel)}")
    print(f"  candidate C detected: {bel_c}/{len(bel)}")
    print("\nper room:")
    for room, d in per_room.items():
        print("  %-24s n=%2d (C:%d/NC:%d)  v1 acc=%.3f NC=%d/%d | C acc=%.3f NC=%d/%d"
              % (room, d["n"], d["human_clean"], d["human_not_clean"],
                 d["v1demo"]["accuracy"], d["v1demo"]["confusion_matrix"]["TP"], d["human_not_clean"],
                 d["candidateC"]["accuracy"], d["candidateC"]["confusion_matrix"]["TP"], d["human_not_clean"]))
    print(f"\nwrote {OUT_PRED}\nwrote {OUT_METRICS}\nwrote {OUT_MD}")


if __name__ == "__main__":
    main()
