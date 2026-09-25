# -*- coding: utf-8 -*-
r"""STAGE B - ResNet-50 transfer learning with LOPO grouped evaluation.

Architecture
    torchvision ResNet-50, IMAGENET1K_V2 weights
    fc replaced by Linear(2048 -> 2)   [CLEAN, NOT_CLEAN]
    Phase 1: backbone frozen, train fc only
    Phase 2: unfreeze layer4 + fc, fine-tune at a lower LR

Evaluation
    7-property Leave-One-Property-Out. Inside each fold one TRAINING property is
    held out as inner validation, used only for early stopping and for choosing
    the decision threshold. The test property is never seen during training,
    early stopping or threshold selection.

Inputs are pixels only. room_id / property_group / sub_area / filenames are used
for grouping and reporting, never as features.

Nothing existing is modified: writes to artifacts/models_v2/ and
artifacts/experiments_v2/.
"""
from __future__ import annotations

import csv
import json
import os
import random
import time
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvm
from PIL import Image, ImageEnhance, ImageOps

from dataset_prep import PROJECT_ROOT

COMBINED = os.path.join(PROJECT_ROOT, "artifacts", "combined", "combined_manifest_v2.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "artifacts", "models_v2")
EXP_DIR = os.path.join(PROJECT_ROOT, "artifacts", "experiments_v2")

CONFIG = {
    "architecture": "torchvision resnet50, weights=IMAGENET1K_V2, fc -> Linear(2048,2)",
    "input_size": 224,
    "resize_mode": "letterbox (EXIF corrected, full frame preserved)",
    "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    "phase1": {"trainable": "fc only", "epochs": 6, "lr": 1e-3},
    "phase2": {"trainable": "layer4 + fc", "epochs": 8, "lr": 1e-4},
    "optimizer": "AdamW, weight_decay=1e-4",
    "loss": "CrossEntropyLoss with class weights computed from the training fold",
    "batch_size": 8,
    "augmentation_train_only": ["horizontal flip p=0.5", "rotation +/-7 deg",
                                "brightness 0.85-1.15", "contrast 0.85-1.15",
                                "scale jitter 0.88-1.00 (centre-safe, no aggressive crop)"],
    "augmentation_excluded": ["vertical flip", "heavy rotation", "cutout/erasing",
                              "strong colour jitter", "mixup/mosaic"],
    "early_stopping": "best inner-validation macro-F1, patience 4 (phase 2)",
    "threshold": "chosen on inner-validation probabilities (max macro-F1), applied to the test fold",
    "seed": 1337,
    "grouping": "leave-one-property-out over 7 properties; inner validation is a "
                "different training property (cyclic, deterministic)",
    "metadata_not_used_as_features": ["room_id", "property_group", "sub_area",
                                      "image_file", "image_path", "dataset_version"],
}

MEAN = np.array(CONFIG["normalization"]["mean"], dtype=np.float32)
STD = np.array(CONFIG["normalization"]["std"], dtype=np.float32)
SIZE = CONFIG["input_size"]


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def letterbox(im: Image.Image, size: int = SIZE) -> Image.Image:
    w, h = im.size
    s = size / max(w, h)
    im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
    return canvas


def load_base(path: str) -> Image.Image:
    """EXIF-corrected image, downscaled once for speed (augmentation happens after)."""
    with Image.open(path) as im:
        im.load()
        out = ImageOps.exif_transpose(im).convert("RGB")
    out.thumbnail((SIZE * 3, SIZE * 3), Image.BICUBIC)
    return out


def augment(im: Image.Image, rng: random.Random) -> Image.Image:
    if rng.random() < 0.5:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    ang = rng.uniform(-7, 7)
    if abs(ang) > 0.5:
        im = im.rotate(ang, resample=Image.BICUBIC, expand=False, fillcolor=(0, 0, 0))
    s = rng.uniform(0.88, 1.0)
    if s < 0.999:
        w, h = im.size
        nw, nh = int(w * s), int(h * s)
        left, top = (w - nw) // 2, (h - nh) // 2
        im = im.crop((left, top, left + nw, top + nh))
    im = ImageEnhance.Brightness(im).enhance(rng.uniform(0.85, 1.15))
    im = ImageEnhance.Contrast(im).enhance(rng.uniform(0.85, 1.15))
    return im


def to_tensor(im: Image.Image) -> torch.Tensor:
    arr = np.asarray(letterbox(im), dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return torch.from_numpy(arr.transpose(2, 0, 1))


def build_model() -> nn.Module:
    m = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
    m.fc = nn.Linear(2048, 2)
    return m


def set_trainable(model: nn.Module, phase: int) -> list:
    for p in model.parameters():
        p.requires_grad_(False)
    params = list(model.fc.parameters())
    for p in model.fc.parameters():
        p.requires_grad_(True)
    if phase == 2:
        for p in model.layer4.parameters():
            p.requires_grad_(True)
        params += list(model.layer4.parameters())
    return params


@torch.no_grad()
def predict_probs(model, images, batch=8) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(images), batch):
        x = torch.stack([to_tensor(im) for im in images[i:i + batch]])
        out.append(torch.softmax(model(x), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(out) if out else np.array([])


def macro_f1(y, p) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(y, p, average="macro", zero_division=0))


def fold_metrics(y_true, y_pred) -> dict:
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 confusion_matrix, f1_score, precision_score, recall_score)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "n": int(len(y_true)), "n_pos": int(y_true.sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "clean_precision": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "clean_recall": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "notclean_precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "notclean_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "notclean_f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def train_fold(rows, idx_tr, idx_val, idx_te, images, y, fold_tag, log):
    seed_all(CONFIG["seed"])
    rng = random.Random(CONFIG["seed"])
    model = build_model()

    counts = np.bincount(y[idx_tr], minlength=2)
    w = torch.tensor([len(idx_tr) / (2 * max(1, counts[0])),
                      len(idx_tr) / (2 * max(1, counts[1]))], dtype=torch.float32)
    crit = nn.CrossEntropyLoss(weight=w)
    log(f"    class counts train: CLEAN={counts[0]} NOT_CLEAN={counts[1]} -> weights {w.tolist()}")

    best = {"macro_f1": -1.0, "state": None, "epoch": -1, "phase": 0, "thr": 0.5}
    bs = CONFIG["batch_size"]

    for phase in (1, 2):
        params = set_trainable(model, phase)
        lr = CONFIG[f"phase{phase}"]["lr"]
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
        epochs = CONFIG[f"phase{phase}"]["epochs"]
        patience, since_best = 4, 0

        for ep in range(1, epochs + 1):
            model.train()
            order = list(idx_tr)
            rng.shuffle(order)
            tot = 0.0
            for i in range(0, len(order), bs):
                chunk = order[i:i + bs]
                x = torch.stack([to_tensor(augment(images[j], rng)) for j in chunk])
                t = torch.tensor(y[chunk])
                opt.zero_grad()
                loss = crit(model(x), t)
                loss.backward()
                opt.step()
                tot += float(loss) * len(chunk)

            vp = predict_probs(model, [images[j] for j in idx_val])
            yv = y[idx_val]
            thrs = np.round(np.arange(0.05, 0.96, 0.01), 2)
            scores = [(macro_f1(yv, (vp >= t).astype(int)), t) for t in thrs]
            vf1, vthr = max(scores, key=lambda s: (s[0], -abs(s[1] - 0.5)))
            log(f"    phase{phase} ep{ep:02d} loss={tot/len(order):.4f} "
                f"inner_macroF1={vf1:.3f} thr={vthr:.2f}")

            if vf1 > best["macro_f1"] + 1e-9:
                best = {"macro_f1": vf1, "epoch": ep, "phase": phase, "thr": float(vthr),
                        "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
                since_best = 0
            else:
                since_best += 1
                if phase == 2 and since_best >= patience:
                    log(f"    early stop (patience {patience})")
                    break

    model.load_state_dict(best["state"])
    tp_prob = predict_probs(model, [images[j] for j in idx_te])
    pred = (tp_prob >= best["thr"]).astype(int)
    torch.save({"state_dict": best["state"], "threshold": best["thr"],
                "config": CONFIG, "fold": fold_tag},
               os.path.join(MODEL_DIR, f"cnn_resnet50_fold_{fold_tag}.pt"))
    return tp_prob, pred, best


def main() -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(EXP_DIR, exist_ok=True)
    logf = open(os.path.join(EXP_DIR, "cnn_training_log.txt"), "w", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    with open(COMBINED, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    y = np.array([1 if r["label"] == "NOT_CLEAN" else 0 for r in rows])
    groups = np.array([r["property_group"] for r in rows])
    props = sorted(set(groups))

    log(f"loading {len(rows)} images into memory (EXIF corrected)...")
    t0 = time.time()
    images = [load_base(os.path.join(PROJECT_ROOT, r["image_path"].replace("/", os.sep)))
              for r in rows]
    log(f"  loaded in {time.time()-t0:.0f}s")
    log(f"properties: {props}  positives={int(y.sum())}/{len(y)}")

    oof_prob = np.full(len(y), np.nan)
    oof_pred = np.full(len(y), -1, dtype=int)
    fold_results, pred_rows = [], []

    for fi, held in enumerate(props):
        inner_val = props[(fi + 1) % len(props)]          # deterministic, cyclic
        idx_te = np.where(groups == held)[0]
        idx_val = np.where(groups == inner_val)[0]
        idx_tr = np.where((groups != held) & (groups != inner_val))[0]
        log(f"\nfold {fi+1}/{len(props)}  test={held}  inner_val={inner_val}  "
            f"train={len(idx_tr)} val={len(idx_val)} test={len(idx_te)}")
        t1 = time.time()
        prob, pred, best = train_fold(rows, idx_tr, idx_val, idx_te, images, y,
                                      held.replace(" ", "_"), log)
        oof_prob[idx_te], oof_pred[idx_te] = prob, pred
        m = fold_metrics(y[idx_te], pred)
        m.update({"fold": fi + 1, "held_out_property": held, "inner_val_property": inner_val,
                  "threshold": best["thr"], "best_epoch": best["epoch"],
                  "best_phase": best["phase"], "inner_macro_f1": best["macro_f1"],
                  "train_images": int(len(idx_tr)), "minutes": round((time.time()-t1)/60, 1)})
        fold_results.append(m)
        log(f"  -> test acc={m['accuracy']:.3f} bal={m['balanced_accuracy']:.3f} "
            f"NCrec={m['notclean_recall']:.3f} NCprec={m['notclean_precision']:.3f} "
            f"thr={best['thr']:.2f} ({m['minutes']} min)")

        for j, i in enumerate(idx_te):
            r = rows[i]
            pred_rows.append({
                "model": "cnn_resnet50_finetuned_v2",
                "room_id": r["room_id"], "property_group": r["property_group"],
                "sub_area": r["sub_area"], "dataset_version": r["dataset_version"],
                "image_file": r["image_file"], "image_path": r["image_path"],
                "human_label": r["label"],
                "model_prediction": "NOT_CLEAN" if pred[j] == 1 else "CLEAN",
                "p_not_clean": round(float(prob[j]), 6),
                "threshold": best["thr"], "fold": fi + 1,
                "correct_or_error": ("correct" if pred[j] == y[i] else
                                     ("false_positive" if pred[j] == 1 else "false_negative")),
                "human_reason": r["human_reason"],
            })

    agg = fold_metrics(y, oof_pred)
    agg.update({"model": "cnn_resnet50_finetuned_v2", "n_images": len(y),
                "n_properties": len(props),
                "thresholds_per_fold": {m["held_out_property"]: m["threshold"] for m in fold_results}})

    with open(os.path.join(EXP_DIR, "lopo_v2_cnn_predictions.csv"), "w",
              newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(pred_rows[0].keys()))
        w.writeheader(); w.writerows(pred_rows)
    with open(os.path.join(EXP_DIR, "lopo_v2_cnn_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"aggregate": agg, "per_fold": fold_results,
                   "config": CONFIG,
                   "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")},
                  f, indent=2)
    with open(os.path.join(MODEL_DIR, "training_config.json"), "w", encoding="utf-8") as f:
        json.dump(CONFIG, f, indent=2)

    log("\nAGGREGATE (7-property LOPO, 163 images, CNN fine-tuned)")
    for k in ("accuracy", "balanced_accuracy", "clean_precision", "clean_recall",
              "notclean_precision", "notclean_recall", "notclean_f1", "macro_f1"):
        log(f"  {k:<22} {agg[k]:.4f}")
    log(f"  confusion TN={agg['tn']} FP={agg['fp']} FN={agg['fn']} TP={agg['tp']}")
    logf.close()


if __name__ == "__main__":
    main()
