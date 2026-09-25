# Final test — 27 untouched A11/A12 images

> Held-out test: these 27 images were **not** used for training, threshold
> selection or candidate selection in the run that produced these numbers.

- Generated: 2026-09-24T11:53:33+00:00
- Training data: 136 images, properties ['A33', 'A34', 'B41', 'C21', 'C31']
- Test data: 27 images, properties ['A11', 'A12']
- Candidate C threshold: **0.03** (F2, inner 5-property folds of the 136)
- v1 demo threshold: 0.07 (artifact default, unchanged)

## Isolation

| check | value |
|---|---|
| Test images used in training | False |
| Test images used in threshold selection | False |
| Test images used in candidate selection (this run) | False |
| Development dataset versions | v1_original only |

**Caveat:** candidate C's view set and F2 rule were originally chosen in an earlier comparison whose metrics included these 27 images; the selection-integrity re-check in this run repeats that choice using the 136 alone

### Selection-integrity control (candidates re-ranked on the 136 alone)

| candidate | belongings /16 | NC recall | NC precision | balanced acc | macro-F1 | FP |
|---|---|---|---|---|---|---|
| C_lower_f2 | 13/16 | 0.742 | 0.511 | 0.766 | 0.726 | 22 |
| A_tile2x2_f2 | 12/16 | 0.710 | 0.489 | 0.745 | 0.708 | 23 |
| B_allviews_f2 | 12/16 | 0.742 | 0.489 | 0.757 | 0.712 | 24 |
| A0_tile2x2_f1 | 12/16 | 0.710 | 0.512 | 0.755 | 0.722 | 21 |

Selected on the 136 alone: **C_lower_f2** — the same candidate as before, so the earlier selection did not depend on the test images.

## Head-to-head on the 27 unseen images

| metric | v1 demo (current) | candidate C | change |
|---|---|---|---|
| Accuracy | 0.7037 | 0.7037 | +0.0000 |
| Balanced accuracy | 0.5833 | 0.6111 | +0.0278 |
| CLEAN precision | 0.7083 | 0.7273 | +0.0190 |
| CLEAN recall | 0.9444 | 0.8889 | -0.0555 |
| NOT_CLEAN precision | 0.6667 | 0.6000 | -0.0667 |
| NOT_CLEAN recall | 0.2222 | 0.3333 | +0.1111 |
| NOT_CLEAN F1 | 0.3333 | 0.4286 | +0.0953 |
| Belongings-type recall | 1/8 | 2/8 | +1 image |

### Confusion matrices

```
v1 demo                         candidate C
            pred C  pred NC                 pred C  pred NC
actual C      17       1        actual C      16       2
actual NC       7       2        actual NC       6       3
```

## Per-room

| room | images | human C / NC | v1 accuracy | v1 NC detected | C accuracy | C NC detected |
|---|---|---|---|---|---|---|
| A11 B Double Attached | 6 | 6 / 0 | 1.000 | 0/0 | 1.000 | 0/0 |
| A11 C Double common | 7 | 2 / 5 | 0.571 | 2/5 | 0.714 | 3/5 |
| A12 B Double attached | 8 | 4 / 4 | 0.500 | 0/4 | 0.500 | 0/4 |
| A12 C Double common | 6 | 6 / 0 | 0.833 | 0/0 | 0.667 | 0/0 |

## Per-image predictions

| room | image | human | candidate C | P(NC) | v1 demo | P(NC) |
|---|---|---|---|---|---|---|
| A11 B Double Attached | IMG-20260923-WA0016.jpg | CLEAN | CLEAN | 0.022 | CLEAN | 0.023 |
| A11 B Double Attached | IMG-20260923-WA0020.jpg | CLEAN | CLEAN | 0.000 | CLEAN | 0.002 |
| A11 B Double Attached | IMG-20260923-WA0022.jpg | CLEAN | CLEAN | 0.010 | CLEAN | 0.004 |
| A11 B Double Attached | IMG-20260923-WA0023.jpg | CLEAN | CLEAN | 0.001 | CLEAN | 0.001 |
| A11 B Double Attached | IMG-20260923-WA0024.jpg | CLEAN | CLEAN | 0.001 | CLEAN | 0.001 |
| A11 B Double Attached | IMG-20260923-WA0025.jpg | CLEAN | CLEAN | 0.000 | CLEAN | 0.000 |
| A11 C Double common | IMG-20260923-WA0053.jpg | NOT_CLEAN | CLEAN | 0.014 | CLEAN | 0.005 |
| A11 C Double common | IMG-20260923-WA0054.jpg | NOT_CLEAN | NOT_CLEAN | 0.272 | NOT_CLEAN | 0.738 |
| A11 C Double common | IMG-20260923-WA0055.jpg | CLEAN | CLEAN | 0.007 | CLEAN | 0.002 |
| A11 C Double common | IMG-20260923-WA0056.jpg | NOT_CLEAN | NOT_CLEAN | 0.447 | NOT_CLEAN | 0.506 |
| A11 C Double common | IMG-20260923-WA0057.jpg | NOT_CLEAN | NOT_CLEAN | 0.168 | CLEAN | 0.019 |
| A11 C Double common | IMG-20260923-WA0058.jpg | CLEAN | CLEAN | 0.000 | CLEAN | 0.001 |
| A11 C Double common | IMG-20260923-WA0059.jpg | NOT_CLEAN | CLEAN | 0.000 | CLEAN | 0.001 |
| A12 B Double attached | IMG-20260923-WA0073.jpg | CLEAN | CLEAN | 0.000 | CLEAN | 0.001 |
| A12 B Double attached | IMG-20260923-WA0075.jpg | CLEAN | CLEAN | 0.000 | CLEAN | 0.000 |
| A12 B Double attached | IMG-20260923-WA0078.jpg | NOT_CLEAN | CLEAN | 0.000 | CLEAN | 0.000 |
| A12 B Double attached | IMG-20260923-WA0079.jpg | NOT_CLEAN | CLEAN | 0.017 | CLEAN | 0.050 |
| A12 B Double attached | IMG-20260923-WA0080.jpg | CLEAN | CLEAN | 0.001 | CLEAN | 0.001 |
| A12 B Double attached | IMG-20260923-WA0081.jpg | NOT_CLEAN | CLEAN | 0.004 | CLEAN | 0.009 |
| A12 B Double attached | IMG-20260923-WA0082.jpg | CLEAN | CLEAN | 0.001 | CLEAN | 0.001 |
| A12 B Double attached | IMG-20260923-WA0083.jpg | NOT_CLEAN | CLEAN | 0.001 | CLEAN | 0.000 |
| A12 C Double common | IMG-20260923-WA0092.jpg | CLEAN | CLEAN | 0.001 | CLEAN | 0.001 |
| A12 C Double common | IMG-20260923-WA0093.jpg | CLEAN | CLEAN | 0.001 | CLEAN | 0.001 |
| A12 C Double common | IMG-20260923-WA0098.jpg | CLEAN | CLEAN | 0.001 | CLEAN | 0.003 |
| A12 C Double common | IMG-20260923-WA0099.jpg | CLEAN | CLEAN | 0.008 | CLEAN | 0.006 |
| A12 C Double common | IMG-20260923-WA0100.jpg | CLEAN | NOT_CLEAN | 0.061 | NOT_CLEAN | 0.166 |
| A12 C Double common | IMG-20260923-WA0101.jpg | CLEAN | NOT_CLEAN | 0.053 | CLEAN | 0.037 |

## Verdict

Candidate C detects one more NOT_CLEAN image than v1 (3/9 vs 2/9) and one more belongings case (2/8 vs 1/8), at the cost of one extra false positive (2 vs 1). Accuracy is identical (0.7037). With 9 positives in the test set a one-image difference is well inside noise, so this is **not** a demonstrated improvement.

Both models fail completely on A12 B (0 of 4 belongings positives detected).

**Recommendation: keep the existing v1 demo model.** No change to `demo_cleanliness_model.joblib` or `demo_app.py`.

## Split marker

These 27 images are marked **FINAL_TEST**. They must not be reused for training, threshold selection or model selection without explicit approval. See `FINAL_TEST_IMAGES.csv`.
