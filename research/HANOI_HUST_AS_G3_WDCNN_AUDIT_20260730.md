# HANOI-HUST AS G3 WDCNN audit

Date: 2026-07-30  
Status: completed R2-style diagnostic baseline; not a primary model.

## Locked protocol

- Same 100 bearing-grouped G2 manifests; manifest SHA-256 `d35c20d17cf8a2f34a08d350b3cf300c1c2e702e7257caa057fd56234fc181a3`.
- I0 source-only information budget; no external data and no compound-label inspection.
- Raw one-channel source windows, per-window mean/standard-deviation normalization, then fixed downsampling by 64.
- Compact WDCNN-style network: wide first convolution followed by three 3-point convolution blocks, batch normalization, ReLU, pooling, global average pooling, and three sigmoid outputs.
- AdamW, batch size 256, 3 epochs, learning rate 1e-3, weight decay 1e-4, deterministic split seed.
- Probabilities are averaged at physical-bearing level before all metrics.

## Results (100 splits)

| Metric | Mean | SD |
|---|---:|---:|
| Macro AUROC | 0.6300 | 0.1443 |
| Macro AUPR | 0.6075 | 0.1197 |
| Balanced accuracy | 0.5297 | 0.0874 |
| Macro-F1 | 0.3857 | 0.1129 |
| Brier score | 0.2501 | 0.0095 |
| Exact-set accuracy | 0.1160 | 0.1426 |
| Hamming loss | 0.4853 | 0.1367 |

The model is substantially below the fixed bearing-grouped logistic reference (AUROC 0.8619) and also below MiniROCKET (AUROC 0.7513). The failure is retained as evidence that an un-tuned compact raw-waveform network is not automatically superior to the frozen engineered-feature baseline.

## Reproducibility artifacts

- Result JSON: `results/g3/hanoi_hust_wdcnn_bearing_grouped.json`; SHA-256 `35817557268e69a15ee6d4a901b5d88184aaab83b15e2c6f323c7927ef615f1c`.
- Unit predictions: `results/g3/unit_level_predictions/hanoi_hust_wdcnn_bearing_grouped.npz`; SHA-256 `72ad6f21edd604a52640076569394f2579ed02a43f03c9d9e68a0252cf9e877c`.

## Decision

Keep WDCNN in the supplementary robustness table as a negative result. Do not tune it against the frozen test bearings or replace the primary model based on this run.
