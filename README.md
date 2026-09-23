# Room Cleanliness Classifier — prototype

> **Prototype — not production validated.** Every figure below describes an
> evaluation procedure on 136 labelled images from a single property. Do not use
> this to make automated decisions affecting tenants.

Classifies a PG/hostel room photo as **CLEAN** or **NOT_CLEAN** using a frozen
ImageNet ResNet-50 as a feature extractor plus a small linear classifier, and
explains the prediction by showing the five image views the model actually sees.

## What the labels mean

**CLEAN** — floor/tiles reasonably clean, no meaningful garbage, waste or residue,
belongings present but reasonably arranged, room reasonably neat.

**NOT_CLEAN** — visible garbage, waste or debris; dirty floor or surfaces; kitchen
dirt; **or** belongings scattered enough that the room looks untidy, even when the
tiles themselves are clean.

Damp patches, peeling paint, wall holes and loose wires are **maintenance** issues,
recorded separately and never used as cleanliness evidence. The full rule set is in
[`cleanliness_labeling_spec.md`](cleanliness_labeling_spec.md).

## Pipeline

```
photo → EXIF orientation correction
      → 5 deterministic views: whole image + 2×2 quadrants (10% overlap)
      → each view letterboxed to 384×384, ImageNet normalisation
      → frozen ResNet-50 (IMAGENET1K_V2), 2048-d per view
      → concatenate → 10,240-d feature  (tile2x2_concat)
      → StandardScaler → class-weighted LogisticRegression
      → probability → threshold → CLEAN / NOT_CLEAN
```

The backbone is never fine-tuned: `eval()` mode, `requires_grad_(False)`,
`torch.no_grad()`.

## Quick start

```bash
pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

streamlit run demo_app.py
```

Opens at `http://localhost:8501`. Take a photo with the device camera or upload a
JPG/PNG. Held-upright (portrait) photos are required — landscape input is blocked
with a retake message rather than being silently rotated, because the model is
measurably less reliable on sideways frames.

On first run, torchvision downloads the ResNet-50 IMAGENET1K_V2 weights (~98 MB)
to your local torch hub cache.

## Repository layout

```
demo_app.py                     Streamlit prototype (camera + upload + orientation guard)
requirements.txt
cleanliness_manifest.csv/.jsonl Verified labels + per-image evidence for 141 images
cleanliness_labeling_spec.md    Labelling rules and the explanation format

prep/
  dataset_prep.py               Manifest loading, EXIF correction, letterbox resize, configs
  preprocess_config_224.json    Preprocessing configurations
  preprocess_config_384.json
  image_audit.py                Dimension / EXIF / duplicate / near-duplicate audit
  make_lopo_folds.py            Deterministic leave-one-property-out folds
  verify_folds.py               Leakage checks (image / room / property)
  extract_embeddings.py         Frozen ResNet-50 whole-image embeddings
  extract_spatial_embeddings.py Frozen ResNet-50 7-view spatial embeddings
  verify_embeddings.py          Embedding integrity checks
  run_lopo_experiment.py        Whole-image LOPO experiment (224 vs 384)
  run_spatial_experiment.py     Spatial/tiled LOPO experiment
  error_analysis.py             Error analysis by evidence subtype
  spatial_error_analysis.py
  fit_demo_model.py             Fits the demo estimator from the validated config
  demo_inference.py             Inference helper used by the Streamlit app
  sanitize_manifest_paths.py    Rewrites manifest paths as repo-relative
  lopo_folds.csv                The five folds
  image_audit_records.csv       Per-image audit output

artifacts/
  model/                        Fitted scaler + classifier (joblib) and provenance metadata
  embeddings/                   ResNet-50 features: 224, 384, and 384 spatial (136 × 7 × 2048)
  experiments/                  LOPO metrics and per-image out-of-fold predictions
```

## Dataset

141 photographs of 14 rooms across 5 properties, labelled **105 CLEAN /
31 NOT_CLEAN / 5 UNCERTAIN**. The 5 UNCERTAIN images (wall-only or window-only
frames with no assessable evidence) are kept in the manifest and excluded from
supervised use, leaving **136** labelled images.

**The original room photographs are not in this repository** — they are ~458 MB and
they show occupied tenant rooms. `image_path` in the manifest is repo-relative, so
placing the room folders back at the repository root restores every script. The
derived ResNet-50 features are committed, so the experiments can be re-run without
the raw photos.

## Evaluation

Images from one room are near-duplicate viewpoints, and rooms in one property share
flooring, furniture and lighting, so a random split would leak badly. Evaluation is
**leave-one-property-out**: five folds, each holding out one property entirely.
Scaler, classifier and decision threshold are fitted **inside each fold's training
properties only**.

Out-of-fold results, best configuration (`tile2x2_concat`, class-weighted):

| metric | value |
|---|---|
| Accuracy | 79.4% |
| Balanced accuracy | 75.3% |
| NOT_CLEAN recall | 67.7% |
| NOT_CLEAN precision | 53.8% |
| Macro F1 | 73.1% |

For reference, always predicting CLEAN scores 77.2% accuracy and 50% balanced
accuracy — which is why accuracy alone is not a meaningful headline here.

Spatial tiling clearly helped arrangement-based evidence (scattered belongings
39% → 72% recall, kitchen surface dirt 60% → 100%) but **not** small floor debris
(still 1 of 4) or discarded packing material.

## Known limitations

- All 31 NOT_CLEAN images originate from only **6 rooms**, in **5 properties**, one
  building, one day, one camera. A different property or phone is out of
  distribution.
- Small floor debris, crumbs and discarded waste are largely invisible to the frozen
  backbone — global average pooling dilutes objects of a few dozen pixels.
- Only 5 UNCERTAIN images exist, too few to learn as a third class.
- The demo estimator in `artifacts/model/` was fitted on **all 136** images so the
  app has something to load. It therefore has **no unbiased accuracy estimate of its
  own** — the figures above belong to the LOPO procedure, not to that file.
- Labels are single-annotator and owner-reviewed; several calls were borderline.

## Reproducing

With the room photos restored at the repository root:

```bash
python prep/image_audit.py              # image characteristics
python prep/make_lopo_folds.py          # fold definitions
python prep/verify_folds.py             # leakage checks
python prep/extract_embeddings.py --config 384 --backbone resnet50
python prep/extract_spatial_embeddings.py
python prep/run_spatial_experiment.py   # LOPO evaluation
python prep/fit_demo_model.py           # repackage the demo estimator
```

Without the photos, the committed embeddings under `artifacts/embeddings/` are
enough to re-run `run_lopo_experiment.py`, `run_spatial_experiment.py` and
`fit_demo_model.py`.

## Next steps

The binding constraint is data, not architecture: more NOT_CLEAN rooms from
**additional properties** would do more than any modelling change. After that, the
open questions are a tighter floor-region view at native resolution for the debris
failure, and whether fine-tuning or a reasoning-capable model is needed for
"discarded versus stored".
