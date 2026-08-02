"""NOACE-style source-only baseline for HANOI HUST.

This module provides a minimal but strict implementation of the NOACE contract:
1) remove nuisance from source-only training by bearing type and load,
2) synthesize all component subset prototypes from healthy/singleton residuals,
3) score unknown records by subset energy and marginalize to component scores,
4) aggregate evidence per physical bearing across the three loads.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut

from .hanoi_hust_baselines import COMPONENT_NAMES, bearing_groups, component_targets

ComponentName = str
SubsetKey = tuple[bool, bool, bool]


@dataclass(frozen=True)
class NOACEFit:
    """Frozen NOACE fit objects used at inference stage."""

    subset_signatures: Sequence[SubsetKey]
    subset_means: np.ndarray
    subset_inv_variance: np.ndarray
    temperature: float

    def subset_energy(
        self,
        residual: np.ndarray,
    ) -> np.ndarray:
        """Return matrix of per-record subset energies.

        Returns
        -------
        np.ndarray
            shape: (n_records, 8), one value per subset.
        """
        values = np.asarray(residual, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError("Residual features must be 2-D")
        centered = values[:, None, :] - self.subset_means[None, :, :]
        return 0.5 * np.sum(
            (centered * centered) * self.subset_inv_variance[None, :, :],
            axis=2,
        )

    def component_probabilities(self, residual: np.ndarray) -> np.ndarray:
        """Return per-record component probabilities from subset evidence."""
        energy = self.subset_energy(residual)
        log_weights = -(energy / self.temperature)
        log_weights = log_weights - log_weights.max(axis=1, keepdims=True)
        weights = np.exp(log_weights)
        weights = np.clip(weights, np.finfo(np.float64).tiny, np.inf)
        weights = weights / np.sum(weights, axis=1, keepdims=True)
        indicator = np.asarray(
            [
                [1.0 if bit else 0.0 for bit in signature]
                for signature in self.subset_signatures
            ],
            dtype=np.float64,
        )
        return weights @ indicator


def _prepare_view(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features)
    if values.ndim == 2:
        return values.astype(np.float64)
    if values.ndim == 3:
        return values.mean(axis=1).astype(np.float64)
    raise ValueError(
        "HANOI HUST NOACE expects record x feature or record x window x feature"
    )


def _select_view(
    arrays: Mapping[str, np.ndarray],
    representation: str,
) -> np.ndarray:
    """Return a view array for one NOACE representation."""
    if representation == "statistics":
        return np.asarray(arrays["statistics"], dtype=np.float32)
    if representation == "fixed_log_power":
        return np.asarray(arrays["fixed_log_power"], dtype=np.float32)
    if representation == "envelope_log_power":
        return np.asarray(arrays["envelope_log_power"], dtype=np.float32)
    if representation == "all":
        return np.concatenate(
            [
                np.asarray(arrays["statistics"], dtype=np.float32),
                np.asarray(arrays["fixed_log_power"], dtype=np.float32),
                np.asarray(arrays["envelope_log_power"], dtype=np.float32),
            ],
            axis=2,
        )
    raise ValueError(f"Unknown NOACE representation: {representation}")


def _build_nuisance_design(
    values: np.ndarray,
    *,
    categories: tuple[int, ...],
) -> tuple[np.ndarray, dict[int, int]]:
    """Build one-hot nuisance design block with fixed category order."""
    integer = np.asarray(values, dtype=np.int64).reshape(-1)
    category_list = list(categories)
    index = {value: position for position, value in enumerate(category_list)}
    design = np.zeros((len(integer), len(category_list)), dtype=np.float64)
    for row, item in enumerate(integer):
        key = int(item)
        if key not in index:
            raise ValueError(f"Unknown nuisance category encountered: {key}")
        design[row, index[key]] = 1.0
    return design, index


def _fit_nuisance(
    features: np.ndarray,
    bearing_type: np.ndarray,
    load_w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...], tuple[int, ...]]:
    """Fit linear nuisance model with bearing type and load one-hot design."""
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("Nuisance input must be record x feature")
    bearing_codes = np.asarray(bearing_type, dtype=np.int64).reshape(-1)
    load_codes = np.asarray(load_w, dtype=np.int64).reshape(-1)
    if len(bearing_codes) != len(load_codes) or len(bearing_codes) != len(x):
        raise ValueError("Nuisance inputs have different lengths")
    bearing_categories = tuple(sorted(set(int(v) for v in bearing_codes)))
    load_categories = tuple(sorted(set(int(v) for v in load_codes)))

    bearing_design, _bearing_map = _build_nuisance_design(
        bearing_codes,
        categories=bearing_categories,
    )
    load_design, _load_map = _build_nuisance_design(
        load_codes,
        categories=load_categories,
    )

    design = np.hstack(
        [
            np.ones((len(x), 1), dtype=np.float64),
            bearing_design,
            load_design,
        ]
    )
    coeff, _, _rank, _singular = np.linalg.lstsq(design, x, rcond=None)
    if design.shape[1] != coeff.shape[0]:
        raise RuntimeError("Nuisance fit matrix shape is invalid")
    if np.any(np.isnan(coeff)):
        raise RuntimeError("Nuisance coefficients are invalid")
    predicted = design @ coeff
    residual = x - predicted
    return residual, coeff, bearing_categories, load_categories


def _subset_states(
    states: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(states, dtype=str)
    if not np.any(values == "N"):
        raise ValueError("NOACE requires healthy (N) records in training partition")
    state_symbols = {"inner": "I", "outer": "O", "ball": "B"}
    for name in COMPONENT_NAMES:
        symbol = state_symbols[name]
        if not np.any(values == symbol):
            raise ValueError(f"NOACE requires singleton records for {name}")
    return values == "N", values == "I", values == "O", values == "B"


def _component_metric(
    truth: np.ndarray,
    score: np.ndarray,
    pred: np.ndarray,
) -> dict[str, float]:
    balanced = [
        float(balanced_accuracy_score(truth[:, idx], pred[:, idx]))
        for idx in range(len(COMPONENT_NAMES))
    ]
    auroc = [
        float(roc_auc_score(truth[:, idx], score[:, idx]))
        for idx in range(len(COMPONENT_NAMES))
    ]
    aupr = [
        float(average_precision_score(truth[:, idx], score[:, idx]))
        for idx in range(len(COMPONENT_NAMES))
    ]
    brier = [
        float(brier_score_loss(truth[:, idx], score[:, idx]))
        for idx in range(len(COMPONENT_NAMES))
    ]
    macro_f1 = [
        float(
            f1_score(
                truth[:, idx],
                pred[:, idx],
                average="macro",
                labels=(0, 1),
                zero_division=0,
            )
        )
        for idx in range(len(COMPONENT_NAMES))
    ]
    exact = np.all(truth == pred, axis=1)
    return {
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
        "exact_set_accuracy": float(np.mean(exact)),
        "hamming_loss": float(np.mean(truth != pred)),
    }


def _aggregate_bearing_predictions(
    groups: np.ndarray,
    truth: np.ndarray,
    probs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate load-level scores to physical bearings and rebuild arrays."""
    group_ids = np.asarray(groups)
    unique_groups, inverse = np.unique(group_ids, return_inverse=True)
    agg_probs = []
    agg_truth = []
    for gi in range(len(unique_groups)):
        mask = inverse == gi
        agg_truth.append(truth[mask][0])
        agg_probs.append(np.mean(probs[mask], axis=0))
    agg_probs_arr = np.asarray(agg_probs, dtype=np.float64)
    agg_truth_arr = np.asarray(agg_truth, dtype=np.int8)
    pred = (agg_probs_arr >= 0.5).astype(np.int8)
    return (
        unique_groups,
        agg_truth_arr,
        agg_probs_arr,
        pred,
    )


def evaluate_noace_candidate(
    *,
    features: np.ndarray,
    states: np.ndarray,
    bearing_type: np.ndarray,
    load_w: np.ndarray,
    groups: np.ndarray,
    random_seed: int,
    representation: str = "all",
) -> dict[str, Any]:
    """Run LOBO-style source validation and return NOACE summary."""
    values = _prepare_view(features)
    target = component_targets(np.asarray(states))
    groups = np.asarray(groups, dtype=str)
    splitter = LeaveOneGroupOut()
    fold_records = np.zeros((len(values), len(COMPONENT_NAMES)), dtype=np.float64)
    fold_pred = np.zeros((len(values), len(COMPONENT_NAMES)), dtype=np.int8)

    fold_count = 0
    for train_index, test_index in splitter.split(values, target, groups):
        fold_count += 1
        x_train = values[train_index]
        x_test = values[test_index]
        state_train = np.asarray(states)[train_index]
        bearing_train = np.asarray(bearing_type)[train_index]
        bearing_test = np.asarray(bearing_type)[test_index]
        load_train = np.asarray(load_w)[train_index]
        load_test = np.asarray(load_w)[test_index]
        residual_train, _coeff, bearing_categories, load_categories = _fit_nuisance(
            x_train,
            bearing_type=bearing_train,
            load_w=load_train,
        )
        residual_test = x_test - _predict_nuisance(
            _coeff,
            bearing_test=bearing_test,
            load_test=load_test,
            bearing_categories=bearing_categories,
            load_categories=load_categories,
            x_test_count=len(x_test),
        )

        healthy, inner, outer, ball = _subset_states(state_train)
        if healthy.sum() == 0 or inner.sum() == 0 or outer.sum() == 0 or ball.sum() == 0:
            raise RuntimeError("NOACE training partition misses required partition")
        healthy_mean = residual_train[healthy].mean(axis=0)
        inner_mean = residual_train[inner].mean(axis=0)
        outer_mean = residual_train[outer].mean(axis=0)
        ball_mean = residual_train[ball].mean(axis=0)
        singleton_effects = [
            healthy_mean,
            inner_mean - healthy_mean,
            outer_mean - healthy_mean,
            ball_mean - healthy_mean,
        ]
        subset_signatures = list(itertools.product([False, True], repeat=3))
        subset_means = np.empty((len(subset_signatures), residual_train.shape[1]), dtype=np.float64)
        for idx, signature in enumerate(subset_signatures):
            prototype = healthy_mean.copy()
            for comp_index, active in enumerate(signature):
                if active:
                    prototype += singleton_effects[comp_index + 1]
            subset_means[idx] = prototype
        train_variance = np.var(residual_train, axis=0)
        train_variance = np.where(
            train_variance > 0,
            train_variance,
            np.finfo(np.float64).eps,
        )
        subset_inv_variance = np.full(
            (len(subset_signatures), residual_train.shape[1]),
            1.0 / train_variance,
            dtype=np.float64,
        )
        fit = NOACEFit(
            subset_signatures=subset_signatures,
            subset_means=subset_means,
            subset_inv_variance=subset_inv_variance,
            temperature=1.0,
        )
        record_scores = fit.component_probabilities(residual_test)
        record_pred = (record_scores >= 0.5).astype(np.int8)
        fold_records[test_index] = record_scores
        fold_pred[test_index] = record_pred

    _, holdout_targets, holdout_scores, holdout_pred = _aggregate_bearing_predictions(
        groups=groups,
        truth=target,
        probs=fold_records,
    )
    metrics = _component_metric(holdout_targets, holdout_scores, holdout_pred)

    return {
        "family": "noace_classical",
        "representation": str(representation),
        "hyperparameter": {
            "nuisance_model": "linear_residual",
            "subset_prototypes": "additive_health_singleton",
            "variance": "diagonal",
            "temperature": 1.0,
            "seed": int(random_seed),
        },
        "fold_count": fold_count,
        "record_count": int(len(values)),
        "bearing_count": int(len(np.unique(groups))),
        "feature_dimension": int(values.shape[1]),
        "group_count": int(len(np.unique(groups))),
        "probabilities": fold_records,
        "group_probabilities": holdout_scores,
        "targets": holdout_targets,
        "predictions": holdout_pred,
        "group_ids": np.asarray(np.unique(groups), dtype=np.str_).tolist(),
        "probability_by_component": {
            name: holdout_scores[:, index].tolist()
            for index, name in enumerate(COMPONENT_NAMES)
        },
        "ground_truth_by_component": {
            name: holdout_targets[:, index].tolist()
            for index, name in enumerate(COMPONENT_NAMES)
        },
        "prediction_by_component": {
            name: holdout_pred[:, index].tolist()
            for index, name in enumerate(COMPONENT_NAMES)
        },
        **metrics,
    }


def _predict_nuisance(
    coeff: np.ndarray,
    *,
    bearing_categories: tuple[int, ...],
    load_categories: tuple[int, ...],
    bearing_test: np.ndarray,
    load_test: np.ndarray,
    x_test_count: int,
) -> np.ndarray:
    """Rebuild nuisance fit matrix for test and predict nuisance contribution."""
    bearing_design, _ = _build_nuisance_design(
        np.asarray(bearing_test, dtype=np.int64),
        categories=bearing_categories,
    )
    load_design, _ = _build_nuisance_design(
        np.asarray(load_test, dtype=np.int64),
        categories=load_categories,
    )
    design = np.hstack(
        [
            np.ones((x_test_count, 1), dtype=np.float64),
            bearing_design,
            load_design,
        ]
    )
    if design.shape != (x_test_count, coeff.shape[0]):
        raise RuntimeError("Nuisance design/test matrix shape mismatch")
    return design @ coeff
