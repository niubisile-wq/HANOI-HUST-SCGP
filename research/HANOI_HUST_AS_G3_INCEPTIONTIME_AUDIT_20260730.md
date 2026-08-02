# HANOI-HUST AS G3 InceptionTime audit

Date: 2026-07-30  
Status: completed R2-style diagnostic baseline; not a primary model.

## Locked protocol

- 100 bearing-grouped G2 manifests; manifest SHA-256 `d35c20d17cf8a2f34a08d350b3cf300c1c2e702e7257caa057fd56234fc181a3`.
- I0 source-only information budget; no external data or compound-label inspection.
- Raw one-channel windows, per-window normalization, fixed downsampling by 64.
- Three compact Inception blocks with 9/19/39-point parallel convolutions, bottleneck channels, residual-free diagnostic head, global average pooling, and three sigmoid outputs.
- AdamW, batch size 256, 3 epochs, learning rate 1e-3, weight decay 1e-4, deterministic split seed.
- Probabilities averaged at physical-bearing level before scoring.

## Results (100 splits)

| Metric | Mean | SD |
|---|---:|---:|
| Macro AUROC | 0.5692 | 0.1479 |
| Macro AUPR | 0.5647 | 0.1090 |
| Balanced accuracy | 0.5157 | 0.0875 |
| Macro-F1 | 0.3703 | 0.1094 |
| Brier score | 0.2493 | 0.0146 |
| Exact-set accuracy | 0.1200 | 0.1393 |
| Hamming loss | 0.4860 | 0.1457 |

The result is below the fixed bearing-grouped logistic reference (AUROC 0.8619), MiniROCKET (0.7513), WDCNN (0.6300), and ResNet1D (0.6094). It is retained as a negative multi-scale raw-waveform robustness result.

## Reproducibility artifacts

- Result JSON: `results/g3/hanoi_hust_inceptiontime_bearing_grouped.json`; SHA-256 `8b3d19ec0cf66a0fe3b9e3ba6122169d9bb722396f9588807e9c913c6e6f38b1`.
- Unit predictions: `results/g3/unit_level_predictions/hanoi_hust_inceptiontime_bearing_grouped.npz`; SHA-256 `0f8dfe97a93a4f6b0f60d66bf1f5ede536ad31a37c4f2ad03d29035a72201dd2`.

## Decision

InceptionTime is supplementary only. No hyperparameter tuning is permitted on frozen outer test bearings.
