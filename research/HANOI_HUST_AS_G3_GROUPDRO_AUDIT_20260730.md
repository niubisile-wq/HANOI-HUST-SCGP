# HANOI-HUST AS G3 source-only GroupDRO audit

Date: 2026-07-30  
Status: completed R2-style domain-robustness comparator.

## Locked protocol

- Same 100 bearing-grouped G2 manifests; manifest SHA-256 `d35c20d17cf8a2f34a08d350b3cf300c1c2e702e7257caa057fd56234fc181a3`.
- I0 source-only information budget. Domain weights use only training records' `load_w`; no test labels or test-domain statistics are used.
- Representation: source engineered envelope log-power features from the frozen record-level cache; physical-bearing aggregation before scoring.
- Six deterministic reweighting rounds. Each round fits a balanced logistic head (`C=10`), estimates source-domain log loss, and updates domain weights toward high-loss domains.

## Results (100 splits)

| Metric | Mean | SD |
|---|---:|---:|
| Macro AUROC | 0.8447 | 0.0993 |
| Macro AUPR | 0.8248 | 0.0948 |
| Balanced accuracy | 0.7208 | 0.1326 |
| Macro-F1 | 0.6798 | 0.1407 |
| Brier score | 0.1539 | 0.0548 |
| Exact-set accuracy | 0.4820 | 0.2091 |
| Hamming loss | 0.2187 | 0.0953 |

The AUROC is slightly below the fixed bearing-grouped logistic reference (0.8619), but the procedure provides a transparent source-only worst-domain robustness comparator. It must not be described as a performance improvement unless a prespecified paired uncertainty test supports that claim.

## Reproducibility artifacts

- Feature cache SHA-256: `95b1b9fef3e75e7049b1e4f5e6f04c61d549936e6840b9cebf0e50cf7fcf9f54`.
- Result JSON: `results/g3/hanoi_hust_groupdro_bearing_grouped.json`; SHA-256 `6ed07f36fdd631a7e01cc5cd7dd8a8b15a6036bb2ceb041d75563647aad8963d`.
- Unit predictions: `results/g3/unit_level_predictions/hanoi_hust_groupdro_bearing_grouped.npz`; SHA-256 `b920a28f8237fc55ed9f78fd5811983c0faf64e4636b781926af772605c6bdfa`.

## Decision

Include GroupDRO in the domain-robustness supplementary table. Do not tune its six rounds or domain definition using outer test results.
