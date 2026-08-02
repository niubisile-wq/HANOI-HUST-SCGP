# SCGP factorial endpoint audit

## Scope

The requested 2 x 2 design crosses split hierarchy (record-grouped versus bearing-grouped) with endpoint aggregation (record-level versus bearing-aggregated).

## Audit result

The G2 runner was extended to retain pre-aggregation record predictions for every split. The resulting split-level means are:

| Split hierarchy | Record-level AUROC / exact-set | Bearing-aggregated AUROC / exact-set |
|---|---:|---:|
| Record-grouped, fixed | 0.951 / 0.776 | 0.959 / 0.791 |
| Bearing-grouped, fixed | 0.828 / 0.510 | 0.860 / 0.490 |
| Bearing-grouped, nested model selection | 0.777 / 0.420 | 0.803 / 0.406 |

The first two rows use the same frozen logistic--envelope reference. The nested row uses fold-specific inner selection; one selected candidate generates all three load predictions within an outer fold before aggregation. Selection frequencies were logistic--envelope 54/100 folds, extra-trees--envelope 29/100, and RBF-SVM--envelope 17/100. The aggregation contrast is an endpoint contrast within each split state; it does not remove the dependence structure of record-grouped splitting. Exact-set 19-bearing cluster-bootstrap intervals are: record-grouped record-level 0.626--0.899, record-grouped bearing-aggregated 0.635--0.919, fixed bearing-grouped record-level 0.417--0.747, fixed bearing-grouped bearing-aggregated 0.400--0.776, nested record-level 0.345--0.686, and nested bearing-aggregated 0.328--0.701.

## Reporting decision

The manuscript now reports the complete crossed table and states that the factorial result supports separate endpoint bookkeeping, not a causal claim that physical-unit separation alone caused the decline. Raw record predictions are retained in the three `*_record_predictions.npz` artifacts.
