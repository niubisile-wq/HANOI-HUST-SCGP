"""Source-only baselines for the HANOI HUST cached source records."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Literal

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import ClassifierMixin
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


ComponentName = Literal["inner", "outer", "ball"]
COMPONENT_NAMES: tuple[ComponentName, ...] = ("inner", "outer", "ball")


@dataclass(frozen=True)
class Candidate:
    """One frozen comparator family and its hyperparameter contract."""

    family: str
    representation: str
    hyperparameter: dict[str, Any]
    build: Callable[[], ClassifierMixin]


def component_targets(states: np.ndarray) -> np.ndarray:
    """Map the frozen source states to three binary component labels."""
    values = np.asarray(states, dtype=str)
    targets = np.zeros((len(values), len(COMPONENT_NAMES)), dtype=np.int8)
    targets[values == "I", 0] = 1
    targets[values == "O", 1] = 1
    targets[values == "B", 2] = 1
    unknown = set(values.tolist()) - {"N", "I", "O", "B"}
    if unknown:
        raise ValueError(f"Unknown HANOI HUST states: {sorted(unknown)}")
    return targets


def bearing_groups(states: np.ndarray, bearing_type: np.ndarray) -> np.ndarray:
    """Physical-bearing grouping across the three repeated loads."""
    state_values = np.asarray(states, dtype=str)
    type_values = np.asarray(bearing_type, dtype=np.int16)
    if state_values.shape != type_values.shape:
        raise ValueError("HANOI HUST grouping inputs must match")
    return np.asarray(
        [f"{state}{bearing}" for state, bearing in zip(state_values, type_values, strict=True)],
        dtype=str,
    )


def aggregate_group_predictions(
    targets: np.ndarray,
    probabilities: np.ndarray,
    groups: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, np.ndarray]:
    """Aggregate repeated observations to one independent physical unit.

    Labels must be invariant within a group. Probabilities are averaged before
    thresholding, so the returned predictions are defined at the same unit as
    the returned targets and probabilities.
    """
    labels = np.asarray(targets, dtype=np.int8)
    scores = np.asarray(probabilities, dtype=np.float64)
    group_values = np.asarray(groups, dtype=str)
    if labels.ndim != 2 or scores.shape != labels.shape:
        raise ValueError("targets and probabilities must have the same 2-D shape")
    if group_values.shape != (len(labels),):
        raise ValueError("group labels must match the number of observations")
    if not np.isfinite(scores).all():
        raise ValueError("probabilities contain non-finite values")
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("threshold must be between zero and one")

    unique_groups, inverse = np.unique(group_values, return_inverse=True)
    grouped_targets = np.zeros((len(unique_groups), labels.shape[1]), dtype=np.int8)
    grouped_probabilities = np.zeros(
        (len(unique_groups), labels.shape[1]), dtype=np.float64
    )
    for group_index, group_name in enumerate(unique_groups):
        members = inverse == group_index
        label_rows = labels[members]
        if not np.all(label_rows == label_rows[0]):
            raise ValueError(f"labels are not invariant within group {group_name}")
        grouped_targets[group_index] = label_rows[0]
        grouped_probabilities[group_index] = scores[members].mean(axis=0)
    grouped_predictions = (grouped_probabilities >= float(threshold)).astype(np.int8)
    return {
        "groups": unique_groups.astype(str),
        "targets": grouped_targets,
        "probabilities": grouped_probabilities,
        "predictions": grouped_predictions,
    }


def _safe_metric(metric: Callable[[np.ndarray, np.ndarray], float], y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    """Return None when a binary metric is undefined for a unit-level slice."""
    if np.unique(y_true).size < 2:
        return None
    return float(metric(y_true, y_score))


def compute_multilabel_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute common metrics at the already-aggregated independent-unit level."""
    labels = np.asarray(targets, dtype=np.int8)
    scores = np.asarray(probabilities, dtype=np.float64)
    if labels.ndim != 2 or scores.shape != labels.shape:
        raise ValueError("targets and probabilities must have the same 2-D shape")
    if predictions is None:
        hard = (scores >= 0.5).astype(np.int8)
    else:
        hard = np.asarray(predictions, dtype=np.int8)
        if hard.shape != labels.shape:
            raise ValueError("predictions must have the same shape as targets")

    balanced = [
        _safe_metric(balanced_accuracy_score, labels[:, head], hard[:, head])
        for head in range(labels.shape[1])
    ]
    auroc = [
        _safe_metric(roc_auc_score, labels[:, head], scores[:, head])
        for head in range(labels.shape[1])
    ]
    aupr = [
        _safe_metric(average_precision_score, labels[:, head], scores[:, head])
        for head in range(labels.shape[1])
    ]
    brier = [
        float(brier_score_loss(labels[:, head], scores[:, head]))
        for head in range(labels.shape[1])
    ]
    macro_f1 = [
        float(
            f1_score(
                labels[:, head],
                hard[:, head],
                labels=(0, 1),
                average="macro",
                zero_division=0,
            )
        )
        for head in range(labels.shape[1])
    ]

    def _mean(values: list[float | None]) -> float | None:
        numeric = [value for value in values if value is not None]
        return float(np.mean(numeric)) if numeric else None

    return {
        "unit_count": int(len(labels)),
        "balanced_accuracy_by_component": balanced,
        "mean_component_balanced_accuracy": _mean(balanced),
        "auroc_by_component": auroc,
        "mean_component_auroc": _mean(auroc),
        "aupr_by_component": aupr,
        "mean_component_aupr": _mean(aupr),
        "brier_by_component": brier,
        "mean_brier_score": float(np.mean(brier)),
        "macro_f1_by_component": macro_f1,
        "mean_component_macro_f1": float(np.mean(macro_f1)),
        "exact_set_accuracy": float(np.all(labels == hard, axis=1).mean()),
        "hamming_loss": float(np.mean(labels != hard)),
    }


def representation_views(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Build the frozen feature views used by the source-only baselines."""
    statistics = np.asarray(arrays["statistics"], dtype=np.float32)
    fixed = np.asarray(arrays["fixed_log_power"], dtype=np.float32)
    envelope = np.asarray(arrays["envelope_log_power"], dtype=np.float32)
    if (
        statistics.ndim != 3
        or fixed.ndim != 3
        or envelope.ndim != 3
        or statistics.shape[:2] != fixed.shape[:2]
        or statistics.shape[:2] != envelope.shape[:2]
    ):
        raise ValueError("HANOI HUST cache shape contract changed")
    views = {
        "statistics": statistics.reshape(len(statistics), -1),
        "fixed_log_power": fixed.reshape(len(fixed), -1),
        "envelope_log_power": envelope.reshape(len(envelope), -1),
    }
    views["all"] = np.concatenate(list(views.values()), axis=1)
    for name, values in views.items():
        if values.ndim != 2 or not np.isfinite(values).all():
            raise RuntimeError(f"HANOI HUST {name} view is invalid")
    return views


def build_candidates(*, random_state: int) -> tuple[Candidate, ...]:
    """Return the frozen comparator grid for source-only screening."""
    candidates: list[Candidate] = []
    for c_value in (0.01, 0.1, 1.0, 10.0):
        candidates.append(
            Candidate(
                family="logistic_l2",
                representation="all",
                hyperparameter={"C": c_value},
                build=lambda value=c_value: make_pipeline(
                    StandardScaler(),
                    LogisticRegression(
                        C=value,
                        class_weight="balanced",
                        max_iter=5_000,
                        random_state=random_state,
                    ),
                ),
            )
        )
    for c_value in (0.1, 1.0, 10.0):
        for gamma in ("scale", "auto"):
            candidates.append(
                Candidate(
                    family="rbf_svm",
                    representation="all",
                    hyperparameter={"C": c_value, "gamma": gamma},
                    build=lambda value=c_value, g=gamma: CalibratedClassifierCV(
                        make_pipeline(
                            StandardScaler(),
                            SVC(
                                C=value,
                                gamma=g,
                                kernel="rbf",
                                class_weight="balanced",
                                probability=False,
                                random_state=random_state,
                            ),
                        ),
                        ensemble=False,
                    ),
                )
            )
    for leaf in (1, 2, 4):
        candidates.append(
            Candidate(
                family="extra_trees",
                representation="all",
                hyperparameter={"min_samples_leaf": leaf},
                build=lambda value=leaf: ExtraTreesClassifier(
                    n_estimators=500,
                    min_samples_leaf=value,
                    max_features="sqrt",
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            )
        )
    candidates.append(
        Candidate(
            family="empirical_prior",
            representation="all",
            hyperparameter={},
            build=lambda: _EmpiricalPriorClassifier(),
        )
    )
    return tuple(candidates)


def build_g2_candidates(*, random_state: int) -> tuple[Candidate, ...]:
    """Return the G2 protocol pool with its explicitly frozen compute budget.

    The historical source baseline uses 500 ExtraTrees. G2 uses 100 trees in
    every protocol cell so that the matched comparison is computationally
    tractable and identical across split/selection factors.
    """
    candidates: list[Candidate] = []
    for candidate in build_candidates(random_state=random_state):
        if candidate.family == "rbf_svm":
            c_value = float(candidate.hyperparameter["C"])
            gamma = str(candidate.hyperparameter["gamma"])
            candidates.append(
                Candidate(
                    family="rbf_svm",
                    representation="all",
                    hyperparameter={"C": c_value, "gamma": gamma, "calibration_cv": 3},
                    build=lambda value=c_value, g=gamma: CalibratedClassifierCV(
                        make_pipeline(
                            StandardScaler(),
                            SVC(
                                C=value,
                                gamma=g,
                                kernel="rbf",
                                class_weight="balanced",
                                probability=False,
                                random_state=random_state,
                            ),
                        ),
                        cv=3,
                        ensemble=False,
                    ),
                )
            )
            continue
        if candidate.family != "extra_trees":
            candidates.append(candidate)
            continue
        leaf = int(candidate.hyperparameter["min_samples_leaf"])
        candidates.append(
            Candidate(
                family="extra_trees",
                representation="all",
                hyperparameter={"min_samples_leaf": leaf, "n_estimators": 100},
                build=lambda value=leaf: ExtraTreesClassifier(
                    n_estimators=100,
                    min_samples_leaf=value,
                    max_features="sqrt",
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            )
        )
    return tuple(candidates)


def build_g3_random_forest_candidate(*, random_state: int) -> Candidate:
    """Published-style Random Forest comparator for the G3 feature baseline."""
    return Candidate(
        family="random_forest_g3",
        representation="all",
        hyperparameter={
            "n_estimators": 500,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "class_weight": "balanced",
        },
        build=lambda: RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=1,
            max_features="sqrt",
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
    )


def build_deep_candidates(*, random_state: int) -> tuple[Candidate, ...]:
    """Return a compact deep comparator grid for NOACE-Deep."""
    candidates: list[Candidate] = []
    for hidden_layers, alpha, learning_rate in (
        ((128,), 1e-4, 1e-3),
        ((128, 64), 1e-4, 1e-3),
        ((256, 128), 1e-5, 5e-4),
    ):
        candidates.append(
            Candidate(
                family="mlp_deep",
                representation="all",
                hyperparameter={
                    "hidden_layer_sizes": hidden_layers,
                    "alpha": alpha,
                    "learning_rate_init": learning_rate,
                    "early_stopping": True,
                },
                build=lambda layers=hidden_layers, reg=alpha, lr=learning_rate: make_pipeline(
                    StandardScaler(),
                    MLPClassifier(
                        hidden_layer_sizes=layers,
                        activation="relu",
                        solver="adam",
                        alpha=reg,
                        batch_size="auto",
                        learning_rate_init=lr,
                        max_iter=800,
                        early_stopping=True,
                        n_iter_no_change=30,
                        validation_fraction=0.2,
                        shuffle=True,
                        random_state=random_state,
                    ),
                ),
            )
        )
    candidates.append(
        Candidate(
            family="mlp_deep",
            representation="all",
            hyperparameter={
                "hidden_layer_sizes": (64, 64, 32),
                "alpha": 1e-4,
                "learning_rate_init": 1e-3,
                "early_stopping": True,
            },
            build=lambda: make_pipeline(
                StandardScaler(),
                MLPClassifier(
                    hidden_layer_sizes=(64, 64, 32),
                    activation="relu",
                    solver="adam",
                    alpha=1e-4,
                    batch_size="auto",
                    learning_rate_init=1e-3,
                    max_iter=800,
                    early_stopping=True,
                    n_iter_no_change=30,
                    validation_fraction=0.2,
                    shuffle=True,
                    random_state=random_state,
                ),
            ),
        )
    )
    return tuple(candidates)


class _EmpiricalPriorClassifier:
    """A simple probability prior baseline for one binary head."""

    def fit(self, features: np.ndarray, target: np.ndarray) -> "_EmpiricalPriorClassifier":
        del features
        target = np.asarray(target, dtype=np.int8)
        self.positive_rate_ = float(target.mean())
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        positives = np.full(len(features), self.positive_rate_, dtype=np.float64)
        return np.column_stack([1.0 - positives, positives])


def _predict_positive_probability(
    model: ClassifierMixin,
    features: np.ndarray,
) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)
        if probabilities.ndim != 2 or probabilities.shape[1] < 2:
            raise RuntimeError("HANOI HUST comparator probability output changed")
        return np.asarray(probabilities[:, 1], dtype=np.float64)
    raise RuntimeError("Comparator lacks predict_proba")


def evaluate_candidate(
    features: np.ndarray,
    targets: np.ndarray,
    groups: np.ndarray,
    candidate: Candidate,
) -> dict[str, Any]:
    """Leave-one-bearing-out evaluation for one frozen comparator."""
    values = np.asarray(features, dtype=np.float32)
    labels = np.asarray(targets, dtype=np.int8)
    group_values = np.asarray(groups, dtype=str)
    if values.ndim != 2 or labels.shape != (len(values), len(COMPONENT_NAMES)):
        raise ValueError("HANOI HUST candidate inputs are invalid")
    splitter = LeaveOneGroupOut()
    probabilities = np.zeros_like(labels, dtype=np.float64)
    predictions = np.zeros_like(labels, dtype=np.int8)
    fold_count = 0
    for train_index, test_index in splitter.split(values, labels, group_values):
        fold_count += 1
        for head in range(len(COMPONENT_NAMES)):
            estimator = candidate.build()
            estimator.fit(values[train_index], labels[train_index, head])
            positive = _predict_positive_probability(
                estimator,
                values[test_index],
            )
            probabilities[test_index, head] = positive
            predictions[test_index, head] = (positive >= 0.5).astype(np.int8)
    balanced = [
        float(balanced_accuracy_score(labels[:, head], predictions[:, head]))
        for head in range(len(COMPONENT_NAMES))
    ]
    auroc = [
        float(roc_auc_score(labels[:, head], probabilities[:, head]))
        for head in range(len(COMPONENT_NAMES))
    ]
    aupr = [
        float(average_precision_score(labels[:, head], probabilities[:, head]))
        for head in range(len(COMPONENT_NAMES))
    ]
    brier = [
        float(brier_score_loss(labels[:, head], probabilities[:, head]))
        for head in range(len(COMPONENT_NAMES))
    ]
    macro_f1 = [
        float(
            f1_score(
                labels[:, head],
                predictions[:, head],
                labels=(0, 1),
                average="macro",
                zero_division=0,
            )
        )
        for head in range(len(COMPONENT_NAMES))
    ]
    exact = np.all(labels == predictions, axis=1)
    return {
        "family": candidate.family,
        "representation": candidate.representation,
        "hyperparameter": candidate.hyperparameter,
        "fold_count": fold_count,
        "record_count": int(len(values)),
        "feature_dimension": int(values.shape[1]),
        "balanced_accuracy_by_component": dict(
            zip(COMPONENT_NAMES, balanced, strict=True)
        ),
        "mean_component_balanced_accuracy": float(np.mean(balanced)),
        "auroc_by_component": dict(zip(COMPONENT_NAMES, auroc, strict=True)),
        "mean_component_auroc": float(np.mean(auroc)),
        "aupr_by_component": dict(zip(COMPONENT_NAMES, aupr, strict=True)),
        "mean_component_aupr": float(np.mean(aupr)),
        "brier_by_component": dict(zip(COMPONENT_NAMES, brier, strict=True)),
        "mean_brier_score": float(np.mean(brier)),
        "macro_f1_by_component": dict(
            zip(COMPONENT_NAMES, macro_f1, strict=True)
        ),
        "mean_component_macro_f1": float(np.mean(macro_f1)),
        "exact_set_accuracy": float(exact.mean()),
        "hamming_loss": float(np.mean(labels != predictions)),
        "probabilities": probabilities,
        "predictions": predictions,
        "targets": labels,
    }


def evaluate_candidates(
    features_by_view: dict[str, np.ndarray],
    targets: np.ndarray,
    groups: np.ndarray,
    *,
    random_state: int,
) -> list[dict[str, Any]]:
    """Evaluate the frozen comparator grid across every feature view."""
    results = []
    for view_name, features in features_by_view.items():
        for candidate in build_candidates(random_state=random_state):
            if candidate.representation != "all":
                continue
            screened = Candidate(
                family=candidate.family,
                representation=view_name,
                hyperparameter=candidate.hyperparameter,
                build=candidate.build,
            )
            results.append(
                evaluate_candidate(features, targets, groups, screened)
            )
    results.sort(
        key=lambda row: (
            -row["mean_component_auroc"],
            -row["exact_set_accuracy"],
            row["mean_brier_score"],
            row["feature_dimension"],
            row["family"],
            row["representation"],
            json.dumps(row["hyperparameter"], sort_keys=True),
        )
    )
    return results


def evaluate_deep_candidates(
    features_by_view: dict[str, np.ndarray],
    targets: np.ndarray,
    groups: np.ndarray,
    *,
    random_state: int,
) -> list[dict[str, Any]]:
    """Evaluate only the deep comparator grid across every feature view."""
    results = []
    for view_name, features in features_by_view.items():
        for candidate in build_deep_candidates(random_state=random_state):
            screened = Candidate(
                family=candidate.family,
                representation=view_name,
                hyperparameter=candidate.hyperparameter,
                build=candidate.build,
            )
            results.append(
                evaluate_candidate(features, targets, groups, screened)
            )
    results.sort(
        key=lambda row: (
            -row["mean_component_auroc"],
            -row["exact_set_accuracy"],
            row["mean_brier_score"],
            row["feature_dimension"],
            row["family"],
            row["representation"],
            json.dumps(row["hyperparameter"], sort_keys=True),
        )
    )
    return results
