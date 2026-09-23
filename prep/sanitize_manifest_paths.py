# -*- coding: utf-8 -*-
r"""Rewrite the manifest's absolute local paths as repo-relative paths.

Only the image_path column changes:
    D:\data science\CNN\A33 A single attach\IMG_x.jpg   ->   A33 A single attach/IMG_x.jpg

Everything else - labels, evidence text, room ids, sub_area, ordering, row count -
is preserved byte for byte. Paths are resolved back to absolute at load time by
dataset_prep.load_manifest(), so behaviour on this machine is unchanged while the
published file carries no machine-specific paths.

Run:  python prep\sanitize_manifest_paths.py            (writes in place, after checks)
      python prep\sanitize_manifest_paths.py --dry-run  (report only)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(PROJECT_ROOT, "cleanliness_manifest.csv")
JSONL_PATH = os.path.join(PROJECT_ROOT, "cleanliness_manifest.jsonl")


def to_relative(path: str) -> str:
    """Absolute -> repo-relative, forward slashes. Already-relative paths pass through."""
    if not os.path.isabs(path):
        return path.replace("\\", "/")
    return os.path.relpath(path, PROJECT_ROOT).replace("\\", "/")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    with open(CSV_PATH, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        cols = list(reader.fieldnames or [])
        rows = list(reader)
    jl = [json.loads(line) for line in open(JSONL_PATH, encoding="utf-8")]

    assert len(rows) == 141 and len(jl) == 141, (len(rows), len(jl))

    before_labels = [r["final_label"] for r in rows]
    n_abs = sum(1 for r in rows if os.path.isabs(r["image_path"]))
    print(f"rows: {len(rows)}   absolute image_path values: {n_abs}")
    if n_abs == 0:
        print("nothing to sanitize - manifest already uses relative paths")
        return

    for r in rows:
        rel = to_relative(r["image_path"])
        resolved = os.path.join(PROJECT_ROOT, rel.replace("/", os.sep))
        assert os.path.exists(resolved), f"path would not resolve: {rel}"
        r["image_path"] = rel
    for r in jl:
        r["image_path"] = to_relative(r["image_path"])

    # invariants
    assert [r["final_label"] for r in rows] == before_labels, "labels changed"
    assert [r["image_path"] for r in rows] == [r["image_path"] for r in jl], "csv/jsonl diverged"
    for a, b in zip(rows, jl):
        for k in a:
            assert str(a[k]) == str(b[k]), (k, a[k], b[k])

    print("sample:", rows[0]["image_path"])
    print("labels unchanged, csv/jsonl identical, all 141 paths resolve")

    if args.dry_run:
        print("--dry-run: nothing written")
        return

    with open(CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    with open(JSONL_PATH, "w", encoding="utf-8") as f:
        for r in jl:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"written: {CSV_PATH}")
    print(f"written: {JSONL_PATH}")


if __name__ == "__main__":
    sys.exit(main())
