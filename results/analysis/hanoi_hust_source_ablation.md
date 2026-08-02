# Hanoi HUST Source Ablation

- Selected model: `logistic_l2` / `envelope_log_power` / `{'C': 10.0}`
- Best view: `envelope_log_power`
- Best family: `logistic_l2`

## View Ranking

| View | Family | Hyperparameter | Mean AUROC | Exact-set | Mean CBA |
|---|---|---|---:|---:|---:|
| envelope_log_power | logistic_l2 | {'C': 10.0} | 0.813845 | 0.543860 | 0.742725 |
| all | logistic_l2 | {'C': 0.1} | 0.806702 | 0.596491 | 0.728968 |
| fixed_log_power | logistic_l2 | {'C': 0.01} | 0.752557 | 0.403509 | 0.677116 |
| statistics | logistic_l2 | {'C': 10.0} | 0.720635 | 0.543860 | 0.729365 |

## Family Ranking

| Family | View | Hyperparameter | Mean AUROC | Exact-set | Mean CBA |
|---|---|---|---:|---:|---:|
| logistic_l2 | envelope_log_power | {'C': 10.0} | 0.813845 | 0.543860 | 0.742725 |
| extra_trees | envelope_log_power | {'min_samples_leaf': 4} | 0.747178 | 0.596491 | 0.657804 |
| rbf_svm | all | {'C': 1.0, 'gamma': 'scale'} | 0.725044 | 0.473684 | 0.611508 |
| empirical_prior | statistics | {} | 0.000000 | 0.263158 | 0.500000 |

## Key Findings

- Best view gain over `fixed_log_power`: 0.061287
- Best view gain over `statistics`: 0.093210
- Exact-set gain over `fixed_log_power`: 0.140351
- Gap between selected model and best-view score: 0.000000
