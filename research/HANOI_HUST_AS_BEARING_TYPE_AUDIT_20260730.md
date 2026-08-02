# HANOI HUST bearing-type and label cross-distribution audit

The source cache contains five bearing-type codes (4--8) across 19 physical bearings and 57 records. The record distribution is:

| Bearing type | Normal | Inner | Outer | Ball | Total |
|---:|---:|---:|---:|---:|---:|
| 4 | 3 | 3 | 3 | 0 | 9 |
| 5 | 3 | 3 | 3 | 3 | 12 |
| 6 | 3 | 3 | 3 | 3 | 12 |
| 7 | 3 | 3 | 3 | 3 | 12 |
| 8 | 3 | 3 | 3 | 3 | 12 |

Each bearing type is represented across the three load values (0, 200, and 400). Type 4 has no ball-state source record, whereas types 5--8 each contain one normal, inner, outer, and ball record at each load. This imbalance is a potential bearing-type/label confound and is not removed merely by grouping records by bearing.

The source identity probe therefore supports a conservative interpretation: bearing-level grouping prevents same-bearing leakage, but it does not establish that the learned signal is independent of hardware type. A type-stratified or leave-type-out analysis would be needed to isolate that mechanism; the current manuscript reports the probe as a confounding diagnostic rather than a causal test.
