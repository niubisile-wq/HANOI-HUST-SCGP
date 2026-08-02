# HANOI-HUST AS G3 MiniROCKET audit

Date: 2026-07-30  
Status: completed exploratory R2 baseline; not a claim of state-of-the-art performance.

## Locked protocol

- Split: the same 100 bearing-grouped manifests used by G2; manifest SHA-256 `d35c20d17cf8a2f34a08d350b3cf300c1c2e702e7257caa057fd56234fc181a3`.
- Information budget: `I0_source_only`; no external dataset, no test-bearing reuse, and no target-domain adaptation.
- Independent metric unit: physical bearing. Window probabilities are aggregated to the bearing before scoring.
- Model: `sktime 0.35.0` `RocketClassifier`, `rocket_transform="minirocket"`, 1,000 kernels, one-channel raw waveform, deterministic per-split seed.
- Computational representation: waveform downsampled by 64 (800 samples/window) solely to make the method reproducible on the available CPU. This is an explicit model-input choice, not a post-hoc performance filter.

## Results

| Metric | Mean | SD |
|---|---:|---:|
| Macro AUROC | 0.7513 | 0.1463 |
| Macro AUPR | 0.7079 | 0.1471 |
| Balanced accuracy | 0.6818 | 0.1150 |
| Macro-F1 | 0.6284 | 0.1178 |
| Brier score | 0.2048 | 0.0707 |
| Exact-set accuracy | 0.5560 | 0.1409 |
| Hamming loss | 0.2600 | 0.0834 |

The result is below the G2 record-grouped logistic reference (AUROC 0.9585) and below the G2 bearing-grouped fixed logistic reference (AUROC 0.8619). It is therefore a negative/diagnostic baseline, not the paper's primary model.

## Reproducibility artifacts

- Result JSON: `results/g3/hanoi_hust_minirocket_bearing_grouped.json`; SHA-256 `f8ff93a408246988d2085539ad4118498fed7a7fce8d55eeb641f5dc3959c888`.
- Unit predictions: `results/g3/unit_level_predictions/hanoi_hust_minirocket_bearing_grouped.npz`; SHA-256 `dcf3da14a68e4b915d6911d3ac2fd59d29bb769442616ff817a0b69a6f343120`.
- Waveform cache: `artifacts/hanoi_hust_window/source_window_waveforms.npy`; SHA-256 `32f9c0837d80fdfd9d89a9fcb229cf7f97bb459cb955e0461f28584e49b563b91`.

## Decision

MiniROCKET is retained as a transparent, source-only robustness comparator. It does not displace the prespecified logistic baseline and should not be presented as evidence of superiority. Any future full-resolution run must be registered as a separate computational variant because it changes the input budget.
