r"""Rewrite absolute image paths as repo-relative in the tracked CSV artifacts.

ONLY the image_path column is touched. Every other column - labels, metrics,
predictions, probabilities, thresholds, fold assignments, filenames, evidence
text - is written back byte for byte, in the original row and column order.

Run:  python prep\sanitize_csv_paths.py [--dry-run]
"""
from __future__ import annotations

import argparse
import csv
import os

from dataset_prep import PROJECT_ROOT, to_repo_relative

TARGETS = [
    os.path.join(PROJECT_ROOT, "artifacts", "experiments", "spatial_lopo_predictions.csv"),
    os.path.join(PROJECT_ROOT, "artifacts", "experiments", "lopo_predictions_error_analysis.csv"),
    os.path.join(PROJECT_ROOT, "prep", "lopo_folds.csv"),
    os.path.join(PROJECT_ROOT, "prep", "image_audit_records.csv"),
]
PATH_COLUMNS = ("image_path",)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    for path in TARGETS:
        name = os.path.relpath(path, PROJECT_ROOT)
        with open(path, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            cols = list(reader.fieldnames or [])
            rows = list(reader)

        before = [dict(r) for r in rows]
        n_abs = 0
        for r in rows:
            for c in PATH_COLUMNS:
                if c in r and os.path.isabs(r[c]):
                    resolved = r[c]
                    r[c] = to_repo_relative(r[c])
                    back = os.path.normpath(os.path.join(PROJECT_ROOT, r[c].replace("/", os.sep)))
                    assert back.lower() == os.path.normpath(resolved).lower(), (resolved, r[c])
                    n_abs += 1

        # nothing but the path columns may differ
        for a, b in zip(before, rows):
            for k in a:
                if k not in PATH_COLUMNS:
                    assert a[k] == b[k], f"{name}: column {k} changed"

        print(f"{name:<52} rows={len(rows):>5}  paths rewritten={n_abs:>5}")

        if args.dry_run or n_abs == 0:
            continue
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)

    print("--dry-run: nothing written" if args.dry_run else "done")


if __name__ == "__main__":
    main()
