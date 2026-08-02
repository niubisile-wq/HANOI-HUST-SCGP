# HANOI-HUST to Paderborn one-shot external lockbox audit

Date: 2026-07-30  
Status: completed one-shot I0 boundary check; not a pooled benchmark result.

## Frozen boundary

- Official source: Paderborn University Bearing DataCenter.
- Lockbox archives: K001 (healthy), KA01 (outer-race damage), and KI01 (inner-race damage); 80 four-second records per bearing, 240 records total.
- Archive SHA-256: K001 `0f119ebdb28fb2f4d9fac1beb1319429f63f7ae1256c23c872f280f3560918e5`; KA01 `6a6be1e11132730cc6f560d51eacedcbfd5fd74b829e9d8d3728c6c8a7cd4c0e`; KI01 `b1dd6d99bb64d556f889fefaedb7e6e672900f5f015125615feaae776f055348`.
- Independent unit: four-digit physical bearing code. All repeated records from one code are aggregated before scoring.
- Model: the HANOI source-only `logistic_l2 + envelope_log_power`, `C=10`, with standardization fit only on the HANOI source cache. Threshold 0.5. No external tuning.
- Signals were resampled from 64 kHz to the frozen 51.2 kHz feature interface; the same three offset aggregation procedure was applied.

## Results

| Component | AUROC | AUPR | Balanced accuracy | Brier | Macro-F1 |
|---|---:|---:|---:|---:|---:|
| Inner race | 0.000 | 0.333 | 0.250 | 0.459 | 0.250 |
| Outer race | 0.000 | 0.333 | 0.500 | 0.344 | 0.400 |

Exact-set accuracy across the three bearing units was 0.333 and Hamming loss was 0.333. The healthy bearing was correctly identified; the two damaged bearings were not mapped to their correct component labels.

## Interpretation and red lines

This is a negative transport boundary, not evidence that the HANOI model is universally invalid. The external sample contains only three physical units, a different rig, bearing type, sampling interface, and damage construction. The result must be reported as a one-shot boundary check with unit count and preprocessing disclosed, not pooled with HANOI or used for post-hoc model selection.

## Artifacts

- Result JSON: `results/g3/hanoi_hust_paderborn_external_lockbox.json`.
- Derived feature cache: `results/g3/hanoi_hust_paderborn_external_features.npz`; SHA-256 `683b9682e8cb88199fb7c60d9651f5150c84c8b5dec7d68be7ec0837021d0ea1`.
- Execution script: `src/run_hanoi_hust_paderborn_external_lockbox.py`.
