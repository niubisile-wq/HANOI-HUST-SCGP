# HANOI HUST threshold and physical-unit audit

## Frozen source reference

The source prediction cache contains 57 acquisition records from 19 physical bearings, with three load records per bearing. Bearing-level predictions are obtained by averaging the three record probabilities before thresholding and exact-set scoring.

| Fixed component threshold | Record exact-set | Bearing exact-set |
|---:|---:|---:|
| 0.2 | 0.491 | 0.474 |
| 0.3 | 0.544 | 0.526 |
| 0.4 | 0.544 | 0.526 |
| 0.5 (pre-specified) | 0.544 | 0.579 |
| 0.6 | 0.561 | 0.526 |
| 0.7 | 0.561 | 0.579 |
| 0.8 | 0.561 | 0.579 |

Thresholds were not selected from the compound confirmation set. The table is an operating-boundary audit, not a replacement for the pre-specified 0.5 endpoint.

## Per-bearing exact-set audit

The frozen source reference has 19 bearing-level outcomes. Five bearings (B5, B6, I7, O4, and O5) have zero exact-set accuracy across their three records. The cluster-bootstrap 95% interval for source bearing-level exact-set accuracy is 0.333--0.754. The complete machine-readable audit is `results/analysis/hanoi_hust_statistics_hardening.json` and the companion report is `results/analysis/hanoi_hust_statistics_hardening.md`.

## Interpretation boundary

The threshold table does not establish an optimal operating threshold. It shows that exact-set accuracy changes with a fixed threshold and that record-level and bearing-level summaries are not interchangeable. No target-label calibration or threshold tuning was performed.
