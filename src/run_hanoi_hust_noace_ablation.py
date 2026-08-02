"""Registered NOACE sensitivity and residual audit for HANOI HUST."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut

from electrical_fm.hanoi_hust_baselines import bearing_groups, component_targets, representation_views
from electrical_fm.hanoi_hust_noace import (
    NOACEFit,
    _aggregate_bearing_predictions,
    _component_metric,
    _fit_nuisance,
    _predict_nuisance,
    _prepare_view,
    _subset_states,
)

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "artifacts" / "hanoi_hust" / "source_features.npz"
OUT = ROOT / "results" / "analysis" / "hanoi_hust_noace_ablation.json"


def _fit_variant(x_train, x_test, state_train, bearing_train, bearing_test, load_train, load_test, variant):
    if variant["nuisance"]:
        residual_train, coeff, bearing_categories, load_categories = _fit_nuisance(
            x_train, bearing_train, load_train
        )
        residual_test = x_test - _predict_nuisance(
            coeff,
            bearing_test=bearing_test,
            load_test=load_test,
            bearing_categories=bearing_categories,
            load_categories=load_categories,
            x_test_count=len(x_test),
        )
    else:
        residual_train, residual_test = x_train, x_test

    healthy, inner, outer, ball = _subset_states(state_train)
    healthy_mean = residual_train[healthy].mean(axis=0)
    singleton_means = [
        healthy_mean,
        residual_train[inner].mean(axis=0),
        residual_train[outer].mean(axis=0),
        residual_train[ball].mean(axis=0),
    ]
    effects = [healthy_mean] + [value - healthy_mean for value in singleton_means[1:]]
    signatures = list(itertools.product([False, True], repeat=3))
    prototypes = np.empty((len(signatures), residual_train.shape[1]), dtype=float)
    for idx, signature in enumerate(signatures):
        if variant["prototype"] == "additive":
            prototype = healthy_mean.copy()
            for comp, active in enumerate(signature):
                if active:
                    prototype += effects[comp + 1]
        else:
            prototype = healthy_mean.copy()
        prototypes[idx] = prototype

    variance = np.var(residual_train, axis=0)
    if variant["variance"] == "pooled":
        variance = np.full_like(variance, float(np.mean(variance)))
    variance = np.where(variance > 0, variance, np.finfo(float).eps)
    priors = np.ones(len(signatures), dtype=float)
    if variant["prior"] == "source_frequency":
        # Conservative registered prior: singleton/healthy frequencies define marginal odds;
        # the joint prior is normalized over the eight subsets.
        frequencies = np.array([healthy.mean(), inner.mean(), outer.mean(), ball.mean()])
        priors = np.array([
            np.prod([frequencies[c + 1] if sig[c] else 1.0 - frequencies[c + 1] for c in range(3)])
            for sig in signatures
        ])
        priors = np.maximum(priors, np.finfo(float).eps)
    inv = np.broadcast_to(1.0 / variance, prototypes.shape).copy()
    fit = NOACEFit(
        subset_signatures=signatures,
        subset_means=prototypes,
        subset_inv_variance=inv,
        temperature=float(variant["temperature"]),
    )
    scores = fit.component_probabilities(residual_test)
    if variant["prior"] == "source_frequency":
        # Apply the registered prior as a log-score offset before marginalization.
        energies = np.empty((len(residual_test), len(signatures)), dtype=float)
        for j, _sig in enumerate(signatures):
            diff = residual_test - prototypes[j]
            energies[:, j] = 0.5 * np.sum(diff * diff * inv[j], axis=1)
        logits = -energies / float(variant["temperature"]) + np.log(priors)[None, :]
        logits -= logits.max(axis=1, keepdims=True)
        weights = np.exp(logits)
        weights /= weights.sum(axis=1, keepdims=True)
        indicators = np.asarray([[1.0 if bit else 0.0 for bit in sig] for sig in signatures])
        scores = weights @ indicators
    return scores


def run_variant(name, variant, arrays):
    views = representation_views(arrays)
    x = _prepare_view(views["all"])
    states = arrays["state"].astype(str)
    target = component_targets(states)
    groups = bearing_groups(states, arrays["bearing_type"])
    logo = LeaveOneGroupOut()
    scores = np.zeros_like(target, dtype=float)
    residual_correlations = []
    for train, test in logo.split(x, target, groups):
        if variant["nuisance"]:
            residual_train, _coeff, bcat, lcat = _fit_nuisance(
                x[train], arrays["bearing_type"][train], arrays["load_w"][train]
            )
            design_b = np.eye(len(bcat))[np.searchsorted(bcat, arrays["bearing_type"][train])]
            design_l = np.eye(len(lcat))[np.searchsorted(lcat, arrays["load_w"][train])]
            nuisance = np.hstack([design_b, design_l])
            residual_correlations.append(float(np.max(np.abs(np.corrcoef(residual_train.T, nuisance.T)[:x.shape[1], x.shape[1]:]))))
        scores[test] = _fit_variant(
            x[train], x[test], states[train], arrays["bearing_type"][train], arrays["bearing_type"][test],
            arrays["load_w"][train], arrays["load_w"][test], variant
        )
    group_ids, truth, group_scores, pred = _aggregate_bearing_predictions(groups, target, scores)
    result = _component_metric(truth, group_scores, pred)
    result["variant"] = name
    result["hyperparameters"] = variant
    result["residual_max_abs_corr_mean"] = float(np.mean(residual_correlations)) if residual_correlations else None
    result["residual_max_abs_corr_max"] = float(np.max(residual_correlations)) if residual_correlations else None
    group_correct = np.all(pred == truth, axis=1).astype(float)
    rng = np.random.default_rng(20260728)
    draws = np.asarray([
        group_correct[rng.integers(0, len(group_correct), len(group_correct))].mean()
        for _ in range(10000)
    ])
    result["group_exact_set_accuracy"] = float(group_correct.mean())
    result["group_exact_set_bootstrap_95_ci"] = [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]
    result["group_ids"] = group_ids.tolist()
    result["group_exact_set_outcomes"] = group_correct.astype(int).tolist()
    return result


def main():
    with np.load(CACHE, allow_pickle=False) as payload:
        arrays = {key: np.asarray(payload[key]) for key in payload.files}
    variants = {
        "full": {"nuisance": True, "prototype": "additive", "prior": "equal", "variance": "diagonal", "temperature": 1.0},
        "no_residualization": {"nuisance": False, "prototype": "additive", "prior": "equal", "variance": "diagonal", "temperature": 1.0},
        "no_additive_prototype": {"nuisance": True, "prototype": "healthy_only", "prior": "equal", "variance": "diagonal", "temperature": 1.0},
        "source_frequency_prior": {"nuisance": True, "prototype": "additive", "prior": "source_frequency", "variance": "diagonal", "temperature": 1.0},
        "temperature_0_5": {"nuisance": True, "prototype": "additive", "prior": "equal", "variance": "diagonal", "temperature": 0.5},
        "temperature_2": {"nuisance": True, "prototype": "additive", "prior": "equal", "variance": "diagonal", "temperature": 2.0},
        "pooled_variance": {"nuisance": True, "prototype": "additive", "prior": "equal", "variance": "pooled", "temperature": 1.0},
    }
    results = {name: run_variant(name, spec, arrays) for name, spec in variants.items()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"status": "completed", "variants": results}, indent=2), encoding="utf-8")
    for name, row in results.items():
        print(name, row["mean_component_auroc"], row["exact_set_accuracy"], row["residual_max_abs_corr_mean"])


if __name__ == "__main__":
    main()
