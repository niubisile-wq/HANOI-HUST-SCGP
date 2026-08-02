# HANOI-HUST AS G3 ResNet1D audit

Date: 2026-07-30  
Status: completed R2-style diagnostic baseline; not a primary model.

## Locked protocol

- 100 bearing-grouped G2 manifests; manifest SHA-256 `d35c20d17cf8a2f34a08d350b3cf300c1c2e702e7257caa057fd56234fc181a3`.
- I0 source-only information budget; no external data or compound-label inspection.
- Raw one-channel windows normalized per window and downsampled by 64.
- Compact residual network: 31-point stem, five residual blocks with 7-point convolutions, channel widths 8/16/32, global average pooling, and three sigmoid outputs.
- AdamW, batch size 256, 3 epochs, learning rate 1e-3, weight decay 1e-4, deterministic split seed.
- Window probabilities are averaged to physical-bearing probabilities before scoring.

## Results (100 splits)

| Metric | Mean | SD |
|---|---:|---:|
| Macro AUROC | 0.6094 | 0.1662 |
| Macro AUPR | 0.6070 | 0.1475 |
| Balanced accuracy | 0.5211 | 0.0906 |
| Macro-F1 | 0.3726 | 0.1225 |
| Brier score | 0.2598 | 0.0415 |
| Exact-set accuracy | 0.1200 | 0.1531 |
| Hamming loss | 0.4907 | 0.1662 |

The result is below the fixed bearing-grouped logistic reference (AUROC 0.8619), MiniROCKET (0.7513), and WDCNN (0.6300). It is retained as a negative raw-waveform robustness result and is not used to replace the primary model.

## Reproducibility artifacts

- Result JSON: `results/g3/hanoi_hust_resnet1d_bearing_grouped.json`; SHA-256 `9cb72fce34e0bc6de0c0c3ccd92914a23bfa183626a4b6ea45b095b57206f263`.
- Unit predictions: `results/g3/unit_level_predictions/hanoi_hust_resnet1d_bearing_grouped.npz`; SHA-256 `617553a304e4d5ae9ffbe6c4319a1bd0e00c4e1cceba9b7b48db543916de3119`.

## Decision

ResNet1D remains a supplementary negative comparator. Any future hyperparameter search must use training-only inner splits and must not tune on the 100 frozen outer test splits.
