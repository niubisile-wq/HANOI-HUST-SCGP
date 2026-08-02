"""Run nested group evaluation on the frozen HANOI HUST source cache.

This script uses a leakage-safe outer leave-one-physical-bearing-out split and
an inner grouped stratified validation split for candidate selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut

from electrical_fm.hanoi_hust_baselines import (
    COMPONENT_NAMES,
    Candidate,
    bearing_groups,
    build_candidates,
    build_deep_candidates,
    component_targets,
    representation_views,
)
from electrical_fm.splits import grouped_stratified_validation_split


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "artifacts" / "hanoi_hust" / "source_features.npz"
METADATA = ROOT / "artifacts" / "hanoi_hust" / "source_features_metadata.json"
DEFAULT_OUTPUT = ROOT / "results" / "analysis" / "hanoi_hust_nested_group_screen.json"
DEFAULT_SELECTED_PREDICTIONS = (
    ROOT / "results" / "analysis" / "hanoi_hust_nested_group_screen_selected_predictions.npz"
)


def _as_artifact_path(value: Path) -> str:
    resolved = value.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix(".writing")
    staging.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(staging, path)


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix(".writing")
    with staging.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(staging, path)


def _fit_predict_candidate(
    candidate: Candidate,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.zeros((len(x_eval), len(COMPONENT_NAMES)), dtype=np.float64)
    predictions = np.zeros_like(probabilities, dtype=np.int8)
    for head in range(len(COMPONENT_NAMES)):
        estimator = candidate.build()
        estimator.fit(x_train, y_train[:, head])
        if not hasattr(estimator, "predict_proba"):
            raise RuntimeError("Candidate lacks probability output")
        positive = np.asarray(estimator.predict_proba(x_eval)[:, 1], dtype=np.float64)
        probabilities[:, head] = positive
        predictions[:, head] = (positive >= 0.5).astype(np.int8)
    return probabilities, predictions


def _component_metric(
    truth: np.ndarray,
    score: np.ndarray,
    pred: np.ndarray,
    *,
    allow_undefined: bool = False,
) -> dict[str, float]:
    def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
        if np.unique(y_true).size < 2:
            return None if allow_undefined else float("nan")
        return float(roc_auc_score(y_true, y_score))

    def _safe_average_precision(
        y_true: np.ndarray, y_score: np.ndarray
    ) -> float | None:
        if np.unique(y_true).size < 2:
            return None if allow_undefined else float("nan")
        return float(average_precision_score(y_true, y_score))

    balanced = [
        float(balanced_accuracy_score(truth[:, idx], pred[:, idx]))
        for idx in range(len(COMPONENT_NAMES))
    ]
    auroc = [
        _safe_roc_auc(truth[:, idx], score[:, idx])
        for idx in range(len(COMPONENT_NAMES))
    ]
    aupr = [
        _safe_average_precision(truth[:, idx], score[:, idx])
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
        "mean_component_auroc": (
            None
            if any(value is None for value in auroc)
            else float(np.mean(np.asarray(auroc, dtype=np.float64)))
        ),
        "aupr_by_component": dict(zip(COMPONENT_NAMES, aupr, strict=True)),
        "mean_component_aupr": (
            None
            if any(value is None for value in aupr)
            else float(np.mean(np.asarray(aupr, dtype=np.float64)))
        ),
        "brier_by_component": dict(zip(COMPONENT_NAMES, brier, strict=True)),
        "mean_brier_score": float(np.mean(brier)),
        "macro_f1_by_component": dict(
            zip(COMPONENT_NAMES, macro_f1, strict=True)
        ),
        "mean_component_macro_f1": float(np.mean(macro_f1)),
        "exact_set_accuracy": float(np.mean(exact)),
        "hamming_loss": float(np.mean(truth != pred)),
    }


def _selection_score(metric: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(metric["mean_component_balanced_accuracy"]),
        float(metric["exact_set_accuracy"]),
        -float(metric["mean_brier_score"]),
    )


def _score_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -row["selection_score"][0],
        -row["selection_score"][1],
        -row["selection_score"][2],
        row["feature_dimension"],
        row["family"],
        row["representation"],
        json.dumps(row["hyperparameter"], sort_keys=True),
    )


def _candidate_grid(random_state: int) -> list[Candidate]:
    grid: list[Candidate] = []
    for candidate in build_candidates(random_state=random_state):
        if candidate.family not in {"logistic_l2", "extra_trees", "empirical_prior"}:
            continue
        grid.append(candidate)
    return grid


def build_nested_group_screen(
    *,
    random_state: int,
    output: Path,
    selected_predictions: Path,
) -> dict[str, Any]:
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    if metadata.get("status") != "source_features_complete_compound_sealed":
        raise RuntimeError("HANOI HUST source cache is not complete")
    if _sha256(CACHE) != metadata.get("cache", {}).get("sha256"):
        raise RuntimeError("HANOI HUST source cache hash changed")

    with np.load(CACHE, allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}

    all_features_by_view = representation_views(arrays)
    features_by_view = {
        name: all_features_by_view[name]
        for name in ("all", "statistics", "fixed_log_power")
    }
    states = arrays["state"].astype(str)
    labels = component_targets(states)
    state_codes = np.asarray(
        [{"N": 0, "I": 1, "O": 2, "B": 3}[state] for state in states],
        dtype=np.int64,
    )
    groups = bearing_groups(states, arrays["bearing_type"])
    group_values = np.asarray(groups, dtype=str)
    splitter = LeaveOneGroupOut()

    candidate_grid = _candidate_grid(random_state=random_state)
    fold_probabilities = np.zeros_like(labels, dtype=np.float64)
    fold_predictions = np.zeros_like(labels, dtype=np.int8)
    outer_selection: list[dict[str, Any]] = []
    outer_metrics: list[dict[str, Any]] = []

    fold_count = 0
    for fold_index, (outer_train, outer_test) in enumerate(
        splitter.split(np.zeros((len(labels), 1)), labels, group_values)
    ):
        fold_count += 1
        inner_train, inner_validation = grouped_stratified_validation_split(
            outer_train,
            state_codes,
            group_values.tolist(),
            fraction=0.25,
            seed=random_state + fold_index,
        )

        candidate_rows: list[dict[str, Any]] = []
        for view_name, features in features_by_view.items():
            x_train = features[inner_train]
            x_validation = features[inner_validation]
            y_train = labels[inner_train]
            y_validation = labels[inner_validation]
            for candidate in candidate_grid:
                screened = Candidate(
                    family=candidate.family,
                    representation=view_name,
                    hyperparameter=candidate.hyperparameter,
                    build=candidate.build,
                )
                try:
                    validation_probabilities, validation_predictions = _fit_predict_candidate(
                        screened,
                        x_train,
                        y_train,
                        x_validation,
                    )
                    validation_metrics = _component_metric(
                        y_validation,
                        validation_probabilities,
                        validation_predictions,
                        allow_undefined=True,
                    )
                except Exception as exc:  # pragma: no cover - defensive selection guard
                    validation_metrics = {
                        "mean_component_balanced_accuracy": float("-inf"),
                        "exact_set_accuracy": float("-inf"),
                        "mean_brier_score": float("inf"),
                        "selection_error": type(exc).__name__,
                    }
                row = {
                    "family": screened.family,
                    "representation": screened.representation,
                    "hyperparameter": screened.hyperparameter,
                    "feature_dimension": int(features.shape[1]),
                    "selection_score": _selection_score(validation_metrics),
                    **validation_metrics,
                }
                candidate_rows.append(row)

        candidate_rows.sort(key=_score_key)
        selected_candidate = candidate_rows[0]
        outer_selection.append(
            {
                "fold_index": fold_index,
                "held_out_group": str(np.unique(group_values[outer_test])[0]),
                "selected_family": selected_candidate["family"],
                "selected_representation": selected_candidate["representation"],
                "selected_hyperparameter": selected_candidate["hyperparameter"],
                "inner_validation": {
                    key: value
                    for key, value in selected_candidate.items()
                    if key not in {"selection_error"}
                },
                "candidate_count": len(candidate_rows),
            }
        )

        features = features_by_view[selected_candidate["representation"]]
        selected = Candidate(
            family=selected_candidate["family"],
            representation=selected_candidate["representation"],
            hyperparameter=selected_candidate["hyperparameter"],
            build=next(
                candidate.build
                for candidate in candidate_grid
                if candidate.family == selected_candidate["family"]
                and candidate.hyperparameter == selected_candidate["hyperparameter"]
            ),
        )
        outer_probabilities, outer_predictions = _fit_predict_candidate(
            selected,
            features[outer_train],
            labels[outer_train],
            features[outer_test],
        )
        fold_probabilities[outer_test] = outer_probabilities
        fold_predictions[outer_test] = outer_predictions
        outer_metrics.append(
            {
                "fold_index": fold_index,
                "held_out_group": str(np.unique(group_values[outer_test])[0]),
                "metrics": _component_metric(
                    labels[outer_test],
                    outer_probabilities,
                    outer_predictions,
                    allow_undefined=True,
                ),
            }
        )

    metrics = _component_metric(labels, fold_probabilities, fold_predictions)
    selection_histogram = Counter(
        (
            row["selected_family"],
            row["selected_representation"],
            json.dumps(row["selected_hyperparameter"], sort_keys=True),
        )
        for row in outer_selection
    )
    most_common_selection = selection_histogram.most_common(1)[0][0]
    selected_predictions_output = {
        "probabilities": fold_probabilities.astype(np.float32),
        "predictions": fold_predictions.astype(np.int8),
        "targets": labels.astype(np.int8),
        "groups": group_values.astype("U"),
        "bearing_type": arrays["bearing_type"].astype(np.int16),
        "load_w": arrays["load_w"].astype(np.int16),
        "contract_index": arrays["contract_index"].astype(np.int16),
    }
    _atomic_npz(selected_predictions, selected_predictions_output)

    payload: dict[str, Any] = {
        "stage": "hanoi_hust_nested_group_screen",
        "schema_version": 1,
        "status": "completed",
        "cache": {
            "path": CACHE.relative_to(ROOT).as_posix(),
            "sha256": _sha256(CACHE),
        },
        "metadata": {
            "path": METADATA.relative_to(ROOT).as_posix(),
            "sha256": _sha256(METADATA),
        },
        "record_count": int(len(labels)),
        "bearing_count": int(len(np.unique(group_values))),
        "candidate_count": int(len(candidate_grid) * len(features_by_view)),
        "fold_count": fold_count,
        "outer_selection": outer_selection,
        "outer_metrics": outer_metrics,
        "selection_histogram": [
            {
                "family": family,
                "representation": representation,
                "hyperparameter": json.loads(hyperparameter_json),
                "count": count,
            }
            for (family, representation, hyperparameter_json), count in selection_histogram.most_common()
        ],
        "most_common_selection": {
            "family": most_common_selection[0],
            "representation": most_common_selection[1],
            "hyperparameter": json.loads(most_common_selection[2]),
        },
        **metrics,
        "selected_predictions": _as_artifact_path(selected_predictions),
        "information_boundary": {
            "source_numeric_files_opened": int(len(labels)),
            "compound_numeric_files_opened": 0,
            "transient_numeric_files_opened": 0,
        },
        "provenance": {
            "git_commit": _git("rev-parse", "HEAD"),
            "builder": "src/run_hanoi_hust_nested_group_screen.py",
            "random_state": int(random_state),
        },
    }
    _atomic_json(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-state", type=int, default=20_260_727)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--selected-predictions", type=Path, default=DEFAULT_SELECTED_PREDICTIONS)
    args = parser.parse_args()
    print(
        json.dumps(
            build_nested_group_screen(
                random_state=args.random_state,
                output=args.output,
                selected_predictions=args.selected_predictions,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
