# -*- coding: utf-8 -*-
"""Verify the extracted embeddings. Read-only; fits nothing, trains nothing."""
from __future__ import annotations

import collections
import json
import os

import numpy as np

from dataset_prep import PROJECT_ROOT, load_manifest, excluded_rows

ART = os.path.join(PROJECT_ROOT, "artifacts", "embeddings")
ok = True


def check(name, cond, detail=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name + (f" :: {detail}" if detail else ""))
    if not cond:
        ok = False


def main() -> None:
    rows = load_manifest()
    unc = {r["image_path"] for r in excluded_rows()}
    man_order = [r["image_path"] for r in rows]

    for cfg in ("224", "384"):
        path = os.path.join(ART, f"embeddings_resnet50_{cfg}.npz")
        print(f"\n--- {os.path.basename(path)} ---")
        d = np.load(path, allow_pickle=False)
        emb = d["embedding"]
        check(f"{cfg}: 136 rows", emb.shape[0] == 136, emb.shape)
        check(f"{cfg}: dim 2048", emb.shape[1] == 2048, emb.shape)
        check(f"{cfg}: dtype float32", emb.dtype == np.float32, emb.dtype)
        check(f"{cfg}: all finite", bool(np.isfinite(emb).all()))
        check(f"{cfg}: no all-zero rows", int((np.abs(emb).sum(1) == 0).sum()) == 0)
        check(f"{cfg}: row order matches manifest",
              list(d["image_path"]) == man_order)
        check(f"{cfg}: no UNCERTAIN image present",
              not (set(d["image_path"].tolist()) & unc))
        check(f"{cfg}: labels match manifest",
              list(d["final_label"]) == [r["final_label"] for r in rows])
        check(f"{cfg}: label counts 105/31",
              dict(collections.Counter(d["final_label"].tolist())) == {"CLEAN": 105, "NOT_CLEAN": 31},
              dict(collections.Counter(d["final_label"].tolist())))
        check(f"{cfg}: metadata arrays present",
              all(k in d for k in ("room_id", "property_group", "sub_area", "image_file")))
        # duplicate embedding vectors would signal a loader bug
        uniq = len({e.tobytes() for e in emb})
        check(f"{cfg}: 136 distinct embedding vectors", uniq == 136, uniq)
        print(f"     stats: min={emb.min():.4f} max={emb.max():.4f} mean={emb.mean():.4f} "
              f"L2-norm mean={np.linalg.norm(emb, axis=1).mean():.2f}")
        print(f"     property groups: {dict(collections.Counter(d['property_group'].tolist()))}")
        print(f"     sub_area: {dict(collections.Counter(d['sub_area'].tolist()))}")
        meta = json.load(open(os.path.join(ART, f"embeddings_resnet50_{cfg}_meta.json"), encoding="utf-8"))
        check(f"{cfg}: meta records frozen backbone, no training",
              meta["frozen"] and not meta["fine_tuned"] and not meta["classifier_trained"])
        check(f"{cfg}: meta records no fitted transforms",
              meta["fitted_transforms"].startswith("none"))

    # the two configs must differ - otherwise resize did nothing
    a = np.load(os.path.join(ART, "embeddings_resnet50_224.npz"))["embedding"]
    b = np.load(os.path.join(ART, "embeddings_resnet50_384.npz"))["embedding"]
    diff = float(np.abs(a - b).mean())
    check("224 and 384 embeddings differ", diff > 1e-4, f"mean abs diff = {diff:.4f}")

    print()
    print("OVERALL:", "ALL EMBEDDING CHECKS PASSED" if ok else "FAILURES PRESENT")


if __name__ == "__main__":
    main()
