# -*- coding: utf-8 -*-
"""Step 4 - programmatic leakage verification for the LOPO folds. Read-only."""
from __future__ import annotations

import collections
import csv
import os

from dataset_prep import MANIFEST, load_manifest, excluded_rows, resolve_image_path

FOLDS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lopo_folds.csv")

ok = True


def check(name: str, cond: bool, detail="") -> None:
    global ok
    print(("PASS " if cond else "FAIL ") + name + (f" :: {detail}" if detail else ""))
    if not cond:
        ok = False


def main() -> None:
    with open(FOLDS, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:      # fold file stores repo-relative paths; compare on absolute
        r["image_path"] = resolve_image_path(r["image_path"])
    folds = sorted({int(r["fold"]) for r in rows})
    labeled = load_manifest()
    unc_paths = {r["image_path"] for r in excluded_rows()}

    check("5 folds present", folds == [1, 2, 3, 4, 5], folds)

    for fd in folds:
        fr = [r for r in rows if int(r["fold"]) == fd]
        tr = [r for r in fr if r["split"] == "train"]
        te = [r for r in fr if r["split"] == "test"]
        held = sorted({r["property_group"] for r in te})

        check(f"fold {fd}: every labeled image assigned exactly once",
              len(fr) == 136 and len({r['image_path'] for r in fr}) == 136, len(fr))
        check(f"fold {fd}: no image in both train and test",
              not ({r["image_path"] for r in tr} & {r["image_path"] for r in te}))
        check(f"fold {fd}: no room in both train and test",
              not ({r["room_id"] for r in tr} & {r["room_id"] for r in te}),
              sorted({r["room_id"] for r in tr} & {r["room_id"] for r in te}))
        check(f"fold {fd}: no property in both train and test",
              not ({r["property_group"] for r in tr} & {r["property_group"] for r in te}),
              sorted({r["property_group"] for r in tr} & {r["property_group"] for r in te}))
        check(f"fold {fd}: exactly one property held out", len(held) == 1, held)
        check(f"fold {fd}: test set has both-class presence check",
              sum(1 for r in te if r["final_label"] == "NOT_CLEAN") > 0,
              "NOT_CLEAN in test = %d" % sum(1 for r in te if r["final_label"] == "NOT_CLEAN"))

    # every labeled image is tested exactly once across the 5 folds
    test_counts = collections.Counter(r["image_path"] for r in rows if r["split"] == "test")
    check("every labeled image appears exactly once as TEST across folds",
          len(test_counts) == 136 and set(test_counts.values()) == {1},
          f"unique={len(test_counts)} counts={sorted(set(test_counts.values()))}")

    # coverage matches the manifest's supervised set
    check("fold image set == manifest supervised set",
          {r["image_path"] for r in rows} == {r["image_path"] for r in labeled})

    # UNCERTAIN images never appear
    check("no UNCERTAIN image appears in any fold",
          not ({r["image_path"] for r in rows} & unc_paths),
          sorted({r["image_path"] for r in rows} & unc_paths))

    # held-out property is distinct per fold
    held_per_fold = {fd: sorted({r["property_group"] for r in rows
                                 if int(r["fold"]) == fd and r["split"] == "test"})[0]
                     for fd in folds}
    check("each property held out exactly once",
          sorted(held_per_fold.values()) == sorted(set(held_per_fold.values())),
          held_per_fold)

    # labels unchanged vs manifest
    man = {r["image_path"]: r["final_label"] for r in labeled}
    check("labels in folds match manifest",
          all(man[r["image_path"]] == r["final_label"] for r in rows))

    print()
    print("Fold composition:")
    for fd in folds:
        te = [r for r in rows if int(r["fold"]) == fd and r["split"] == "test"]
        tr = [r for r in rows if int(r["fold"]) == fd and r["split"] == "train"]
        print("  fold %d | held out %-4s | test %3d (%2d NC/%2d C, %d rooms) | train %3d (%2d NC/%2d C, %d rooms)"
              % (fd, held_per_fold[fd], len(te),
                 sum(1 for r in te if r["final_label"] == "NOT_CLEAN"),
                 sum(1 for r in te if r["final_label"] == "CLEAN"),
                 len({r["room_id"] for r in te}), len(tr),
                 sum(1 for r in tr if r["final_label"] == "NOT_CLEAN"),
                 sum(1 for r in tr if r["final_label"] == "CLEAN"),
                 len({r["room_id"] for r in tr})))

    print()
    print("OVERALL:", "ALL LEAKAGE CHECKS PASSED" if ok else "FAILURES PRESENT")


if __name__ == "__main__":
    main()
