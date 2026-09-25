# -*- coding: utf-8 -*-
r"""External validation on NEW rooms that were never part of the 136-image set.

Read-only with respect to everything that exists:
  * does NOT touch cleanliness_manifest.csv/.jsonl
  * does NOT touch artifacts/model, artifacts/embeddings, artifacts/experiments
  * does NOT retrain, refit or tune anything - it loads the existing fitted
    demo artifact and calls the existing demo_inference.predict()
  * the decision threshold is the artifact's own default; it is NOT tuned here

Subcommands:
    inventory  - enumerate the new rooms, write the inventory CSV, build contact
                 sheets for human review
    predict    - run the existing model, join with human labels, write outputs
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from datetime import datetime, timezone

from PIL import Image, ImageOps

from dataset_prep import MANIFEST, PROJECT_ROOT, to_repo_relative

OUT_DIR = os.path.join(PROJECT_ROOT, "artifacts", "validation")
SHEET_DIR = os.path.join(OUT_DIR, "review_sheets")
INVENTORY = os.path.join(OUT_DIR, "external_validation_inventory.csv")
PREDICTIONS = os.path.join(OUT_DIR, "external_validation_predictions.csv")
REPORT_JSON = os.path.join(OUT_DIR, "external_validation_report.json")
REPORT_MD = os.path.join(OUT_DIR, "external_validation_report.md")

IMAGE_EXT = (".jpg", ".jpeg", ".png")


def manifest_paths() -> set[str]:
    with open(MANIFEST, encoding="utf-8-sig", newline="") as f:
        return {r["image_path"].replace("/", os.sep) for r in csv.DictReader(f)}


def discover_new_rooms() -> dict[str, list[str]]:
    """Room folders at the repo root holding images absent from the manifest."""
    known = manifest_paths()
    rooms: dict[str, list[str]] = {}
    for entry in sorted(os.listdir(PROJECT_ROOT)):
        folder = os.path.join(PROJECT_ROOT, entry)
        if not os.path.isdir(folder) or entry in ("prep", "artifacts", ".git"):
            continue
        imgs = sorted(p for p in glob.glob(os.path.join(folder, "*"))
                      if p.lower().endswith(IMAGE_EXT))
        fresh = [p for p in imgs if os.path.relpath(p, PROJECT_ROOT) not in known]
        if fresh:
            rooms[entry] = fresh
    return rooms


def property_group(room_id: str) -> str:
    return room_id.split()[0].upper()


def cmd_inventory() -> None:
    os.makedirs(SHEET_DIR, exist_ok=True)
    rooms = discover_new_rooms()
    rows = []
    for room, imgs in rooms.items():
        for idx, path in enumerate(imgs, 1):
            with Image.open(path) as im:
                stored = im.size
                exif_o = im.getexif().get(274)
                cw, ch = ImageOps.exif_transpose(im).size
            rows.append({
                "room_id": room,
                "property_group": property_group(room),
                "image_index": idx,
                "image_file": os.path.basename(path),
                "image_path": to_repo_relative(path),
                "stored_w": stored[0], "stored_h": stored[1],
                "exif_orientation": exif_o,
                "corrected_w": cw, "corrected_h": ch,
                "orientation": "portrait" if ch > cw else "landscape",
                "in_training_manifest": "no",
                "human_label": "", "human_reason": "",
            })
    with open(INVENTORY, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print(f"new rooms: {len(rooms)}   new images: {len(rows)}")
    for room, imgs in rooms.items():
        print(f"  {room:<26} {len(imgs)} images  (property {property_group(room)})")
    print(f"\ninventory -> {INVENTORY}")

    # contact sheets for human review (written to artifacts/validation, not the dataset)
    from PIL import ImageDraw
    W = 950
    for room, imgs in rooms.items():
        tiles = []
        for i, p in enumerate(imgs, 1):
            im = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
            im.thumbnail((W, W))
            t = Image.new("RGB", (W, W), "white")
            t.paste(im, ((W - im.width) // 2, (W - im.height) // 2))
            d = ImageDraw.Draw(t)
            d.rectangle([0, 0, 70, 46], fill="yellow")
            d.text((10, 8), str(i), fill="black", font_size=32)
            tiles.append(t)
        for s in range(0, len(tiles), 4):
            chunk = tiles[s:s + 4]
            nrows = (len(chunk) + 1) // 2
            sheet = Image.new("RGB", (2 * W, nrows * W), "gray")
            for k, t in enumerate(chunk):
                sheet.paste(t, ((k % 2) * W, (k // 2) * W))
            sheet.save(os.path.join(SHEET_DIR, f"{room.replace(' ', '_')}_{s // 4 + 1}.jpg"),
                       quality=88)
    print(f"review sheets -> {SHEET_DIR}")


def cmd_predict(labels_path: str) -> None:
    """Run the EXISTING model on the new images and compare with human labels."""
    from demo_inference import load_artifacts, load_backbone, predict

    human = {}
    with open(labels_path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            human[r["image_path"]] = (r["human_label"], r["human_reason"])

    with open(INVENTORY, encoding="utf-8-sig", newline="") as f:
        inv = list(csv.DictReader(f))

    bundle, meta = load_artifacts()
    backbone = load_backbone()
    thr = float(bundle["default_threshold"])          # artifact default - NOT tuned here
    print(f"model: {os.path.basename(bundle['representation'])} | threshold {thr} (artifact default)")

    rows = []
    for r in inv:
        lbl, reason = human.get(r["image_path"], ("", ""))
        abs_path = os.path.join(PROJECT_ROOT, r["image_path"].replace("/", os.sep))
        with Image.open(abs_path) as im:
            out = predict(im, backbone, bundle, threshold=thr)
        match = (out["label"] == lbl)
        rows.append({
            "room_id": r["room_id"], "property_group": r["property_group"],
            "image_file": r["image_file"], "image_path": r["image_path"],
            "human_label": lbl, "human_reason": reason,
            "model_prediction": out["label"],
            "p_not_clean": round(out["probability_not_clean"], 6),
            "p_clean": round(out["probability_clean"], 6),
            "threshold": thr,
            "confidence": round(out["confidence"], 6),
            "match": "yes" if match else "no",
            "error_type": ("" if match else
                           ("false_positive" if out["label"] == "NOT_CLEAN" else "false_negative")),
        })
        print(f"  {r['room_id']:<24} {r['image_file']:<30} human={lbl:<10} "
              f"model={out['label']:<10} p={out['probability_not_clean']:.3f} "
              f"{'OK' if match else 'MISS'}")

    with open(PREDICTIONS, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    report = build_report(rows, thr, meta)
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    write_markdown(report, rows)
    print(f"\npredictions -> {PREDICTIONS}\nreport -> {REPORT_JSON}\nreport -> {REPORT_MD}")


def _metrics(rows) -> dict:
    tp = sum(1 for r in rows if r["human_label"] == "NOT_CLEAN" and r["model_prediction"] == "NOT_CLEAN")
    fn = sum(1 for r in rows if r["human_label"] == "NOT_CLEAN" and r["model_prediction"] == "CLEAN")
    fp = sum(1 for r in rows if r["human_label"] == "CLEAN" and r["model_prediction"] == "NOT_CLEAN")
    tn = sum(1 for r in rows if r["human_label"] == "CLEAN" and r["model_prediction"] == "CLEAN")
    n = len(rows)
    return {
        "images": n,
        "human_clean": tn + fp, "human_not_clean": tp + fn,
        "correct": tp + tn, "incorrect": fp + fn,
        "false_positives_clean_to_notclean": fp,
        "false_negatives_notclean_to_clean": fn,
        "accuracy": round((tp + tn) / n, 4) if n else None,
        "notclean_precision": round(tp / (tp + fp), 4) if (tp + fp) else None,
        "notclean_recall": round(tp / (tp + fn), 4) if (tp + fn) else None,
        "clean_recall": round(tn / (tn + fp), 4) if (tn + fp) else None,
        "confusion_matrix": {"TN": tn, "FP": fp, "FN": fn, "TP": tp},
    }


def build_report(rows, thr, meta) -> dict:
    overall = _metrics(rows)
    per_room = {}
    for room in sorted({r["room_id"] for r in rows}):
        rr = [r for r in rows if r["room_id"] == room]
        per_room[room] = _metrics(rr) | {"property_group": rr[0]["property_group"]}
    fns = [{k: r[k] for k in ("room_id", "image_file", "human_reason", "p_not_clean")}
           for r in rows if r["error_type"] == "false_negative"]
    fps = [{k: r[k] for k in ("room_id", "image_file", "human_reason", "p_not_clean")}
           for r in rows if r["error_type"] == "false_positive"]
    return {
        "title": "External validation on 4 new rooms",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "PROTOTYPE - not production validated",
        "model": {
            "artifact": "artifacts/model/demo_cleanliness_model.joblib",
            "pipeline": meta["pipeline"]["representation"],
            "backbone": meta["pipeline"]["backbone"],
            "threshold_used": thr,
            "threshold_source": "artifact default (median of the five LOPO fold thresholds); NOT tuned on this validation set",
            "retrained_or_refitted": False,
        },
        "data": {
            "source": "4 room folders added after the original dataset was frozen",
            "rooms": sorted({r["room_id"] for r in rows}),
            "properties": sorted({r["property_group"] for r in rows}),
            "in_training_manifest": False,
            "used_for_training": False,
        },
        "overall": overall,
        "per_room": per_room,
        "false_negatives": fns,
        "false_positives": fps,
        "caveats": [
            "4 rooms from 2 properties - far too small to establish production performance.",
            "Human reference labels are single-annotator and should be owner-reviewed.",
            "The threshold was not tuned on this set; doing so would invalidate it as held-out data.",
            "These images were NOT added to the training manifest and the model was NOT refitted.",
        ],
    }


def write_markdown(rep: dict, rows) -> None:
    o = rep["overall"]
    cm = o["confusion_matrix"]
    L = []
    L.append("# External Validation Report — 4 new rooms\n")
    L.append("> **Prototype — not yet production validated.** "
             "4 rooms from 2 properties cannot establish production performance.\n")
    L.append(f"- Generated: {rep['created_utc']}")
    L.append(f"- Model artifact: `{rep['model']['artifact']}` (not retrained, not refitted)")
    L.append(f"- Representation: {rep['model']['pipeline']}")
    L.append(f"- Threshold: **{rep['model']['threshold_used']}** — {rep['model']['threshold_source']}")
    L.append(f"- Rooms: {', '.join(rep['data']['rooms'])}")
    L.append(f"- Properties: {', '.join(rep['data']['properties'])}\n")

    L.append("## Overall\n")
    L.append("| metric | value |")
    L.append("|---|---|")
    L.append(f"| Total images | {o['images']} |")
    L.append(f"| Total rooms | {len(rep['per_room'])} |")
    L.append(f"| Human CLEAN | {o['human_clean']} |")
    L.append(f"| Human NOT_CLEAN | {o['human_not_clean']} |")
    L.append(f"| Correct predictions | {o['correct']} |")
    L.append(f"| Incorrect predictions | {o['incorrect']} |")
    L.append(f"| False positives (CLEAN → NOT_CLEAN) | {o['false_positives_clean_to_notclean']} |")
    L.append(f"| **False negatives (NOT_CLEAN → CLEAN)** | **{o['false_negatives_notclean_to_clean']}** |")
    L.append(f"| Accuracy | {o['accuracy']} |")
    L.append(f"| NOT_CLEAN precision | {o['notclean_precision']} |")
    L.append(f"| NOT_CLEAN recall | {o['notclean_recall']} |")
    L.append(f"| CLEAN recall | {o['clean_recall']} |\n")

    L.append("### Confusion matrix\n")
    L.append("| | pred CLEAN | pred NOT_CLEAN |")
    L.append("|---|---|---|")
    L.append(f"| **actual CLEAN** | {cm['TN']} | {cm['FP']} |")
    L.append(f"| **actual NOT_CLEAN** | {cm['FN']} | {cm['TP']} |\n")

    L.append("## Per-room results\n")
    L.append("| room | property | images | human C / NC | correct | FP | FN | accuracy | NC recall |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for room, m in rep["per_room"].items():
        L.append(f"| {room} | {m['property_group']} | {m['images']} | "
                 f"{m['human_clean']} / {m['human_not_clean']} | {m['correct']} | "
                 f"{m['false_positives_clean_to_notclean']} | "
                 f"**{m['false_negatives_notclean_to_clean']}** | {m['accuracy']} | "
                 f"{m['notclean_recall'] if m['notclean_recall'] is not None else 'n/a'} |")
    L.append("")

    L.append("## False negatives — missed dirty rooms (key failure mode)\n")
    if rep["false_negatives"]:
        L.append("| room | image | P(NOT_CLEAN) | human reason |")
        L.append("|---|---|---|---|")
        for f in rep["false_negatives"]:
            L.append(f"| {f['room_id']} | {f['image_file']} | {f['p_not_clean']} | {f['human_reason']} |")
    else:
        L.append("None.")
    L.append("")

    L.append("## False positives — clean rooms flagged dirty\n")
    if rep["false_positives"]:
        L.append("| room | image | P(NOT_CLEAN) | human reason |")
        L.append("|---|---|---|---|")
        for f in rep["false_positives"]:
            L.append(f"| {f['room_id']} | {f['image_file']} | {f['p_not_clean']} | {f['human_reason']} |")
    else:
        L.append("None.")
    L.append("")

    L.append("## Per-image detail\n")
    L.append("| room | image | human | model | P(NOT_CLEAN) | match | human reason |")
    L.append("|---|---|---|---|---|---|---|")
    for r in rows:
        L.append(f"| {r['room_id']} | {r['image_file']} | {r['human_label']} | "
                 f"{r['model_prediction']} | {r['p_not_clean']:.3f} | {r['match']} | {r['human_reason']} |")
    L.append("")

    L.append("## Caveats\n")
    for c in rep["caveats"]:
        L.append(f"- {c}")
    L.append("")

    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["inventory", "predict"])
    ap.add_argument("--labels", default=INVENTORY,
                    help="CSV carrying human_label / human_reason (defaults to the inventory)")
    a = ap.parse_args()
    if a.command == "inventory":
        cmd_inventory()
    else:
        cmd_predict(a.labels)
