# -*- coding: utf-8 -*-
"""Step 2 - characteristics audit of the 136 labeled images. Read-only.

Reports dimensions, aspect ratios, EXIF orientation, load failures,
exact-duplicate hashes, near-duplicate candidates (dHash), and investigates
the per-folder 'cover' images. Deletes nothing.
"""
from __future__ import annotations

import collections
import csv
import json
import os

from PIL import Image, ImageOps

from dataset_prep import (MANIFEST, load_manifest, excluded_rows, verify_all,
                          file_sha256, property_group)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def dhash(path: str, size: int = 8) -> int:
    """Difference hash on the EXIF-corrected grayscale image. Read-only."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("L").resize((size + 1, size), Image.LANCZOS)
    px = list(im.getdata())
    bits = 0
    for row in range(size):
        base = row * (size + 1)
        for col in range(size):
            bits = (bits << 1) | int(px[base + col] > px[base + col + 1])
    return bits


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def main() -> None:
    rows = load_manifest()          # 136 supervised rows
    all_rows = load_manifest(MANIFEST, supervised_only=False)  # 141
    unc = excluded_rows()

    print("=" * 78)
    print("1. LOAD STATUS")
    res = verify_all(rows)
    print(f"  labeled rows: {len(rows)}   loaded ok: {res['loaded_ok']}   failed: {len(res['failed'])}")
    for p, e in res["failed"]:
        print("  FAILED:", p, e)
    recs = res["records"]

    print()
    print("2. DIMENSIONS / ASPECT / EXIF")
    print("  raw (stored) sizes      :", dict(collections.Counter((r["raw_w"], r["raw_h"]) for r in recs)))
    print("  corrected sizes         :", dict(collections.Counter((r["corrected_w"], r["corrected_h"]) for r in recs)))
    print("  exif orientation tag    :", dict(collections.Counter(r["exif_orientation"] for r in recs)))
    print("  orientation after fix   :", dict(collections.Counter(r["orientation_after"] for r in recs)))
    print("  aspect ratio (corrected):", dict(collections.Counter(r["aspect_ratio_corrected"] for r in recs)))

    print()
    print("3. EXACT DUPLICATES (sha256 of file bytes)")
    by_hash = collections.defaultdict(list)
    for r in all_rows:
        by_hash[file_sha256(r["image_path"])].append(r["image_path"])
    dups = {h: v for h, v in by_hash.items() if len(v) > 1}
    print(f"  duplicate groups across all 141 images: {len(dups)}")
    for h, v in dups.items():
        print("   ", h[:12], [os.path.relpath(p, r"D:\data science\CNN") for p in v])

    print()
    print("4. NEAR-DUPLICATE CANDIDATES (dHash, hamming <= 6, labeled set)")
    hashes = {r["image_path"]: dhash(r["image_path"]) for r in rows}
    meta = {r["image_path"]: r for r in rows}
    paths = list(hashes)
    near = []
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            d = hamming(hashes[paths[i]], hashes[paths[j]])
            if d <= 6:
                a, b = meta[paths[i]], meta[paths[j]]
                near.append({
                    "distance": d,
                    "a_file": a["image_file"], "a_room": a["room_id"], "a_label": a["final_label"],
                    "b_file": b["image_file"], "b_room": b["room_id"], "b_label": b["final_label"],
                    "same_room": a["room_id"] == b["room_id"],
                    "same_property": a["property_group"] == b["property_group"],
                    "same_label": a["final_label"] == b["final_label"],
                })
    near.sort(key=lambda x: x["distance"])
    print(f"  candidate pairs: {len(near)}")
    print(f"    same room     : {sum(1 for p in near if p['same_room'])}")
    print(f"    same property, different room: {sum(1 for p in near if p['same_property'] and not p['same_room'])}")
    print(f"    across properties            : {sum(1 for p in near if not p['same_property'])}")
    print(f"    conflicting labels           : {sum(1 for p in near if not p['same_label'])}")
    for p in near[:15]:
        print("    d=%d %-28s [%s] <-> %-28s [%s] same_room=%s same_label=%s"
              % (p["distance"], p["a_file"], p["a_label"], p["b_file"], p["b_label"],
                 p["same_room"], p["same_label"]))

    print()
    print("5. COVER-IMAGE INVESTIGATION (one file per folder named after the room)")
    covers = [r for r in all_rows if not r["image_file"].upper().startswith("IMG_")]
    print(f"  cover-style files: {len(covers)}")
    cover_report = []
    for c in covers:
        same_folder = [r for r in all_rows
                       if r["room_id"] == c["room_id"] and r["sub_area"] == c["sub_area"]
                       and r["image_path"] != c["image_path"]]
        ch = dhash(c["image_path"])
        best = None
        for o in same_folder:
            d = hamming(ch, dhash(o["image_path"]))
            if best is None or d < best[0]:
                best = (d, o["image_file"])
        cover_report.append({
            "room_id": c["room_id"], "cover_file": c["image_file"], "label": c["final_label"],
            "nearest_in_folder": best[1] if best else None,
            "dhash_distance": best[0] if best else None,
            "is_exact_duplicate": any(c["image_path"] in v for v in dups.values()),
        })
        print("   %-24s %-28s label=%-9s nearest=%-28s d=%s exact_dup=%s"
              % (c["room_id"], c["image_file"], c["final_label"],
                 best[1] if best else "-", best[0] if best else "-",
                 cover_report[-1]["is_exact_duplicate"]))

    print()
    print("6. EXCLUDED UNCERTAIN IMAGES (retained, not audited as supervised)")
    for r in unc:
        print("   ", r["room_id"], "|", r["image_file"])

    # persist
    with open(os.path.join(OUT_DIR, "image_audit_records.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        w.writeheader(); w.writerows(recs)
    with open(os.path.join(OUT_DIR, "image_audit_summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "labeled_images": len(rows), "loaded_ok": res["loaded_ok"],
            "failures": res["failed"],
            "raw_sizes": {str(k): v for k, v in collections.Counter((r["raw_w"], r["raw_h"]) for r in recs).items()},
            "corrected_sizes": {str(k): v for k, v in collections.Counter((r["corrected_w"], r["corrected_h"]) for r in recs).items()},
            "exif_orientation": {str(k): v for k, v in collections.Counter(r["exif_orientation"] for r in recs).items()},
            "exact_duplicate_groups": len(dups),
            "near_duplicate_pairs": near,
            "cover_images": cover_report,
        }, f, indent=2)
    print()
    print("written: image_audit_records.csv, image_audit_summary.json")


if __name__ == "__main__":
    main()
