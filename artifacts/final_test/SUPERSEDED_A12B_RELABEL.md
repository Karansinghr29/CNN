# SUPERSEDED — A12 B label correction

Applied: 2026-09-24T12:36:19+00:00

Room **A12 B Double attached** was relabelled: 4 images moved NOT_CLEAN -> CLEAN (localized cot-level untidiness is not room-level NOT_CLEAN).

Every metric below was computed with the OLD A12 B labels and is therefore **superseded**. The numbers have deliberately NOT been recomputed - rerunning is a separate, explicitly approved step.

| artifact | what it contained |
|---|---|
| `artifacts/validation/external_validation_predictions.csv` | external validation of the v1 demo model |
| `artifacts/validation/external_validation_report.json` | external validation metrics |
| `artifacts/validation/external_validation_report.md` | external validation report |
| `artifacts/experiments_v2/lopo_v2_frozen_predictions.csv` | v2 frozen LOPO predictions |
| `artifacts/experiments_v2/lopo_v2_frozen_metrics.json` | v2 frozen LOPO metrics |
| `artifacts/experiments_v2/lopo_v2_cnn_predictions.csv` | v2 CNN LOPO predictions |
| `artifacts/experiments_v2/lopo_v2_cnn_metrics.json` | v2 CNN LOPO metrics |
| `artifacts/experiments_v2/model_comparison_v1_vs_v2.json` | v1 vs v2 comparison |
| `artifacts/experiments_v2/error_analysis_subtypes.csv` | subtype error analysis |
| `artifacts/experiments_v2/belongings_audit.csv` | belongings audit (24 positives) |
| `artifacts/experiments_v3/candidates_v3_predictions.csv` | candidate A/B/C predictions |
| `artifacts/experiments_v3/candidates_v3_metrics.json` | candidate A/B/C metrics |
| `artifacts/experiments_v3/candidate_comparison.json` | candidate regression gate |
| `artifacts/final_test/final_test_predictions.csv` | final test predictions |
| `artifacts/final_test/final_test_metrics.json` | final test metrics |
| `artifacts/final_test/final_test_report.md` | final test report |
| `artifacts/embeddings/embeddings_resnet50_384_spatial_v2.npz` | embedding file carries a copy of the labels in its 'label' array (features unaffected) |

## Which conclusions are affected

- The 27-image external validation of the v1 demo model (was 18 CLEAN / 9 NOT_CLEAN; now 22 CLEAN / 5 NOT_CLEAN). Its NOT_CLEAN recall of 2/9 no longer applies.
- All v2 / v3 LOPO runs that trained or evaluated on the combined 163-image set.
- The final test on the 27 images: both models scored 0/4 on A12 B belongings positives that are no longer positives.
- The belongings-subtype counts (were 24 positives; A12 B contributed 4).

## Not affected

- The original 136-image manifest, its labels and the v1 LOPO baseline (accuracy 0.794, NOT_CLEAN recall 0.677, belongings 11/16) - no A12 B image is in it.
- The deployed demo model and the Streamlit app, which were not changed.
- The embedding *features*; only the label array copied inside the v2 npz is stale.
