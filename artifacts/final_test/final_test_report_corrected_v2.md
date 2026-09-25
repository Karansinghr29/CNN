# Final test — recomputed against corrected A12 B labels

- Generated: 2026-09-25T06:56:11+00:00
- **Supersedes** `final_test_metrics.json` / `final_test_report.md`
- Models were **not** re-run; thresholds unchanged (candidate C 0.03, v1 demo 0.07)
- Test set: **22 CLEAN / 5 NOT_CLEAN** (was 18 / 9)

## Head-to-head (corrected labels)

| metric | v1 demo | candidate C |
|---|---|---|
| accuracy | 0.8519 | 0.8519 |
| balanced accuracy | 0.6773 | 0.7545 |
| clean precision | 0.8750 | 0.9091 |
| clean recall | 0.9545 | 0.9091 |
| notclean precision | 0.6667 | 0.6000 |
| notclean recall | 0.4000 | 0.6000 |
| notclean f1 | 0.5000 | 0.6000 |
| macro f1 | 0.7065 | 0.7545 |
| belongings recall | 1/4 | 2/4 |

```
v1 demo                          candidate C
            pred C  pred NC                  pred C  pred NC
actual C      21       1         actual C      20       2
actual NC       3       2         actual NC       2       3
```

## Before → after (effect of the A12 B relabel)

| metric | v1 before | v1 after | Δ | C before | C after | Δ |
|---|---|---|---|---|---|---|
| accuracy | 0.7037 | 0.8519 | +0.1482 | 0.7037 | 0.8519 | +0.1482 |
| balanced accuracy | 0.5833 | 0.6773 | +0.0940 | 0.6111 | 0.7545 | +0.1434 |
| clean precision | 0.7083 | 0.8750 | +0.1667 | 0.7273 | 0.9091 | +0.1818 |
| clean recall | 0.9444 | 0.9545 | +0.0101 | 0.8889 | 0.9091 | +0.0202 |
| notclean precision | 0.6667 | 0.6667 | +0.0000 | 0.6000 | 0.6000 | +0.0000 |
| notclean recall | 0.2222 | 0.4000 | +0.1778 | 0.3333 | 0.6000 | +0.2667 |
| notclean f1 | 0.3333 | 0.5000 | +0.1667 | 0.4286 | 0.6000 | +0.1714 |
| macro f1 | 0.5714 | 0.7065 | +0.1351 | 0.6143 | 0.7545 | +0.1402 |

## Per-room

| room | images | human C / NC | v1 acc | v1 NC detected | C acc | C NC detected |
|---|---|---|---|---|---|---|
| A11 B Double Attached | 6 | 6 / 0 | 1.000 | 0/0 | 1.000 | 0/0 |
| A11 C Double common | 7 | 2 / 5 | 0.571 | 2/5 | 0.714 | 3/5 |
| A12 B Double attached | 8 | 8 / 0 | 1.000 | 0/0 | 1.000 | 0/0 |
| A12 C Double common | 6 | 6 / 0 | 0.833 | 0/0 | 0.667 | 0/0 |

## Status

- No model was trained, re-run or replaced.
- No threshold was tuned on this set.
- Model selection is unchanged; the v1 demo model remains in place.
