"""Summarize completed G2 cells without changing any predictions."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import kendalltau, spearmanr

from electrical_fm.hanoi_hust_baselines import compute_multilabel_metrics


ROOT = Path(__file__).resolve().parents[1]
INPUTS = {
    (hierarchy, selection): ROOT / "results" / "g2" / f"hanoi_hust_{hierarchy}_{selection}.json"
    for hierarchy in ("record_grouped", "bearing_grouped", "window_random")
    for selection in ("fixed_prespecified", "nested_selection")
}
OUTPUT = ROOT / "results" / "g2" / "protocol_cell_metrics.json"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix(".writing")
    staging.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(staging, path)


def _bootstrap_interval(values: np.ndarray, *, seed: int = 20_260_730, count: int = 10_000) -> list[float]:
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(values), size=(count, len(values)))
    means = values[samples].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _paired_sign_permutation_pvalue(
    delta: np.ndarray,
    *,
    seed: int = 20_261_100,
    draws: int = 10_000,
) -> float:
    observed = abs(float(delta.mean()))
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(draws, len(delta)))
    permuted = np.abs((signs * delta[None, :]).mean(axis=1))
    return float((1 + np.count_nonzero(permuted >= observed)) / (draws + 1))


def _holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, (key, value) in enumerate(ordered):
        running = max(running, min(1.0, value * (total - rank)))
        adjusted[key] = running
    return adjusted


def _entropy(labels: list[str]) -> float:
    if not labels:
        return 0.0
    _, counts = np.unique(np.asarray(labels, dtype=str), return_counts=True)
    probabilities = counts / counts.sum()
    if len(probabilities) == 1:
        return 0.0
    return float(-(probabilities * np.log(probabilities)).sum() / np.log(len(probabilities)))


def _candidate_id(row: dict[str, Any]) -> str:
    return json.dumps(
        {"family": row["family"], "hyperparameter": row["hyperparameter"]},
        sort_keys=True,
    )


def _rank_stability(splits: list[dict[str, Any]], *, top_k: int = 3) -> dict[str, Any]:
    rankings: list[list[str]] = []
    for split in splits:
        rows = split.get("inner_selection", [])
        if not rows:
            continue
        ranked = sorted(
            rows,
            key=lambda row: (
                -(row["metrics"].get("mean_component_auroc") if row["metrics"].get("mean_component_auroc") is not None else -np.inf),
                -row["metrics"]["exact_set_accuracy"],
                row["metrics"]["mean_brier_score"],
            ),
        )
        rankings.append([_candidate_id(row) for row in ranked])
    if len(rankings) < 2:
        return {"status": "not_applicable", "reason": "no_nested_rankings"}
    top_sets = [set(ranking[:top_k]) for ranking in rankings]
    jaccards = []
    spearmans = []
    kendalls = []
    candidate_ids = sorted(set().union(*rankings))
    for left in range(len(rankings)):
        for right in range(left + 1, len(rankings)):
            union = top_sets[left] | top_sets[right]
            jaccards.append(len(top_sets[left] & top_sets[right]) / len(union) if union else 1.0)
            left_rank = {candidate: rank for rank, candidate in enumerate(rankings[left])}
            right_rank = {candidate: rank for rank, candidate in enumerate(rankings[right])}
            x = [left_rank[candidate] for candidate in candidate_ids]
            y = [right_rank[candidate] for candidate in candidate_ids]
            spearman = spearmanr(x, y).statistic
            kendall = kendalltau(x, y).statistic
            if np.isfinite(spearman):
                spearmans.append(float(spearman))
            if np.isfinite(kendall):
                kendalls.append(float(kendall))
    return {
        "status": "completed",
        "ranking_count": len(rankings),
        "top_k": top_k,
        "mean_top_k_jaccard": float(np.mean(jaccards)),
        "mean_spearman_rho": float(np.mean(spearmans)) if spearmans else None,
        "mean_kendall_tau": float(np.mean(kendalls)) if kendalls else None,
        "outer_test_regret": {
            "status": "not_available",
            "reason": "outer_predictions_for_all_candidates_not_saved_in_current_runner",
        },
    }


def _unit_cluster_bootstrap(
    path: Path,
    *,
    seed: int = 20_261_001,
    draws_per_split: int = 100,
) -> dict[str, Any]:
    """Bootstrap the independent-unit rows within each repeated split."""
    with np.load(path, allow_pickle=False) as payload:
        split_ids = np.asarray(payload["split_id"], dtype=np.int16)
        targets = np.asarray(payload["targets"], dtype=np.int8)
        probabilities = np.asarray(payload["probabilities"], dtype=np.float64)
    rng = np.random.default_rng(seed)
    auroc_values: list[float] = []
    exact_values: list[float] = []
    def _fast_macro_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
        values = []
        for head in range(y_true.shape[1]):
            positives = y_score[y_true[:, head] == 1, head]
            negatives = y_score[y_true[:, head] == 0, head]
            if len(positives) == 0 or len(negatives) == 0:
                continue
            comparisons = positives[:, None] - negatives[None, :]
            values.append(float((comparisons > 0).mean() + 0.5 * (comparisons == 0).mean()))
        return float(np.mean(values)) if values else None

    for split_id in np.unique(split_ids):
        rows = np.flatnonzero(split_ids == split_id)
        if len(rows) < 2:
            continue
        sampled = rng.integers(0, len(rows), size=(draws_per_split, len(rows)))
        for draw in sampled:
            sampled_targets = targets[rows[draw]]
            sampled_probabilities = probabilities[rows[draw]]
            auroc = _fast_macro_auroc(sampled_targets, sampled_probabilities)
            if auroc is not None:
                auroc_values.append(auroc)
            sampled_predictions = (sampled_probabilities >= 0.5).astype(np.int8)
            exact_values.append(float(np.all(sampled_targets == sampled_predictions, axis=1).mean()))
    if not auroc_values:
        return {"status": "not_available"}
    return {
        "status": "completed",
        "draws_per_split": draws_per_split,
        "bootstrap_draw_count": len(auroc_values),
        "macro_auroc_mean": float(np.mean(auroc_values)),
        "macro_auroc_interval": [
            float(np.quantile(auroc_values, 0.025)),
            float(np.quantile(auroc_values, 0.975)),
        ],
        "exact_set_mean": float(np.mean(exact_values)),
        "exact_set_interval": [
            float(np.quantile(exact_values, 0.025)),
            float(np.quantile(exact_values, 0.975)),
        ],
        "interpretation": "unit_cluster_bootstrap_within_each_repeated_split",
    }


def _cell_summary(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("split_count") != 100 or data.get("status") not in {"completed", "completed_descriptive_only"}:
        raise RuntimeError(f"G2 cell is not complete: {path}")
    splits = data["splits"]
    metric_names = (
        "mean_component_auroc",
        "mean_component_aupr",
        "mean_component_balanced_accuracy",
        "mean_component_macro_f1",
        "mean_brier_score",
        "exact_set_accuracy",
        "hamming_loss",
    )
    metrics: dict[str, Any] = {}
    for name in metric_names:
        values = np.asarray([row["metrics"][name] for row in splits], dtype=np.float64)
        metrics[name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
            "split_sensitivity_interval": _bootstrap_interval(values),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    selected = [
        json.dumps(
            {
                "family": row["selected_family"],
                "hyperparameter": row["selected_hyperparameter"],
            },
            sort_keys=True,
        )
        for row in splits
    ]
    unique, counts = np.unique(np.asarray(selected, dtype=str), return_counts=True)
    selection_frequency = {
        label: int(count) for label, count in zip(unique.tolist(), counts.tolist(), strict=True)
    }
    selected_regrets = []
    for row in splits:
        candidates = row.get("outer_candidate_metrics", [])
        if not candidates:
            continue
        best = max(
            candidate["metrics"]["mean_component_auroc"]
            for candidate in candidates
            if candidate["metrics"].get("mean_component_auroc") is not None
        )
        selected_value = row["metrics"].get("mean_component_auroc")
        if selected_value is not None:
            selected_regrets.append(float(best - selected_value))
    regret_summary = {
        "status": "completed" if selected_regrets else "not_available",
        "mean_outer_test_regret_macro_auroc": float(np.mean(selected_regrets)) if selected_regrets else None,
        "split_sensitivity_interval": _bootstrap_interval(np.asarray(selected_regrets, dtype=np.float64)) if selected_regrets else None,
    }
    prediction_path = ROOT / data["predictions"]["path"]
    return {
        "hierarchy": data["hierarchy"],
        "selection": data["selection"],
        "inference_allowed": bool(data.get("inference_allowed", True)),
        "split_count": int(data["split_count"]),
        "metrics": metrics,
        "selection_frequency": selection_frequency,
        "normalized_selection_entropy": _entropy(selected),
        "rank_stability": _rank_stability(splits),
        "outer_test_regret": regret_summary,
        "unit_cluster_bootstrap": _unit_cluster_bootstrap(
            prediction_path,
            draws_per_split=100 if data.get("inference_allowed", True) else 25,
        ),
        "prediction_artifact": data["predictions"],
    }


def build_summary(*, output: Path = OUTPUT) -> dict[str, Any]:
    cells = {}
    for key, path in INPUTS.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        cells[f"{key[0]}__{key[1]}"] = _cell_summary(path)
    paired = {}
    p_values: dict[str, float] = {}
    for selection in ("fixed_prespecified", "nested_selection"):
        left = cells[f"record_grouped__{selection}"]
        right = cells[f"bearing_grouped__{selection}"]
        paired[selection] = {}
        for metric in ("mean_component_auroc", "exact_set_accuracy", "hamming_loss"):
            left_values = np.asarray([
                row["metrics"][metric] for row in json.loads(INPUTS[("record_grouped", selection)].read_text(encoding="utf-8"))["splits"]
            ])
            right_values = np.asarray([
                row["metrics"][metric] for row in json.loads(INPUTS[("bearing_grouped", selection)].read_text(encoding="utf-8"))["splits"]
            ])
            delta = right_values - left_values
            paired[selection][metric] = {
                "bearing_minus_record_mean": float(delta.mean()),
                "split_paired_bootstrap_interval": _bootstrap_interval(delta, seed=20_260_900),
            }
            key = f"{selection}__{metric}"
            p_values[key] = _paired_sign_permutation_pvalue(delta)
    holm = _holm_adjust(p_values)
    for selection in paired:
        for metric in paired[selection]:
            paired[selection][metric]["paired_sign_permutation_p"] = p_values[f"{selection}__{metric}"]
            paired[selection][metric]["holm_adjusted_p"] = holm[f"{selection}__{metric}"]
    result = {
        "stage": "hanoi_hust_as_g2_protocol_summary",
        "schema_version": 1,
        "status": "completed",
        "cell_count": len(cells),
        "cells": cells,
        "paired_hierarchy_effects": paired,
        "interpretation_note": (
            "Intervals here quantify repeated-split sensitivity. They are not a substitute "
            "for unit-level cluster bootstrap on one prespecified test set."
        ),
    }
    _atomic_json(output, result)
    return result


def main() -> None:
    result = build_summary()
    print(json.dumps({
        "stage": result["stage"],
        "cell_count": result["cell_count"],
        "paired_hierarchy_effects": result["paired_hierarchy_effects"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
