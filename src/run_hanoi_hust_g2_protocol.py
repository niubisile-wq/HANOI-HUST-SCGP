"""Run the implemented record/bearing G2 protocol cells on frozen features."""

from __future__ import annotations

import hashlib
import json
import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"sklearn\.svm\._base",
)

from electrical_fm.hanoi_hust_baselines import (
    Candidate,
    bearing_groups,
    build_g2_candidates,
    component_targets,
    compute_multilabel_metrics,
    representation_views,
)


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "artifacts" / "hanoi_hust" / "source_features.npz"
WINDOW_CACHE = ROOT / "artifacts" / "hanoi_hust_window" / "source_window_features.npz"
MANIFESTS = ROOT / "results" / "g2" / "split_manifests" / "hanoi_hust_g2_manifests.json"
OUTPUT_DIR = ROOT / "results" / "g2"


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


def _candidate_with_view(candidate: Candidate, view: str) -> Candidate:
    return Candidate(
        family=candidate.family,
        representation=view,
        hyperparameter=candidate.hyperparameter,
        build=candidate.build,
    )


def _predict(
    features: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    eval_indices: np.ndarray,
    candidate: Candidate,
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.zeros((len(eval_indices), labels.shape[1]), dtype=np.float64)
    for head in range(labels.shape[1]):
        estimator = candidate.build()
        estimator.fit(features[train_indices], labels[train_indices, head])
        if not hasattr(estimator, "predict_proba"):
            raise RuntimeError(f"candidate {candidate.family} lacks predict_proba")
        positive = np.asarray(estimator.predict_proba(features[eval_indices])[:, 1], dtype=np.float64)
        probabilities[:, head] = positive
    return probabilities, (probabilities >= 0.5).astype(np.int8)


def _aggregate_test(
    labels: np.ndarray,
    probabilities: np.ndarray,
    groups: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    from electrical_fm.hanoi_hust_baselines import aggregate_group_predictions

    grouped = aggregate_group_predictions(labels, probabilities, groups)
    metrics = compute_multilabel_metrics(
        grouped["targets"], grouped["probabilities"], grouped["predictions"]
    )
    return grouped, metrics


def _selection_key(metrics: dict[str, Any]) -> tuple[float, float, float]:
    auroc = metrics.get("mean_component_auroc")
    exact = metrics.get("exact_set_accuracy")
    brier = metrics.get("mean_brier_score")
    return (
        -(float(auroc) if auroc is not None else -np.inf),
        -float(exact),
        float(brier),
    )


def _choose_nested(
    features: np.ndarray,
    labels: np.ndarray,
    metric_groups: np.ndarray,
    inner_train: np.ndarray,
    inner_validation: np.ndarray,
    candidates: list[Candidate],
) -> tuple[Candidate, list[dict[str, Any]]]:
    rows = []
    for candidate in candidates:
        probabilities, _ = _predict(features, labels, inner_train, inner_validation, candidate)
        _, metrics = _aggregate_test(labels[inner_validation], probabilities, metric_groups[inner_validation])
        rows.append({
            "family": candidate.family,
            "representation": candidate.representation,
            "hyperparameter": candidate.hyperparameter,
            "metrics": metrics,
        })
    best_index = min(range(len(rows)), key=lambda index: _selection_key(rows[index]["metrics"]))
    return candidates[best_index], rows


def run_cell(
    *,
    hierarchy: str,
    selection: str,
    count: int,
    cache: Path = CACHE,
    manifests_path: Path = MANIFESTS,
) -> dict[str, Any]:
    if hierarchy not in {"record_grouped", "bearing_grouped", "window_random"}:
        raise ValueError("unknown G2 split hierarchy")
    if selection not in {"fixed_prespecified", "nested_selection"}:
        raise ValueError("unknown G2 model-selection cell")
    data_path = WINDOW_CACHE if hierarchy == "window_random" else cache
    with np.load(data_path, allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    if hierarchy == "window_random":
        features = np.asarray(arrays["envelope_log_power"], dtype=np.float32)
    else:
        views = representation_views(arrays)
        features = views["envelope_log_power"]
    labels = component_targets(arrays["state"].astype(str))
    record_groups = np.asarray([f"record_{value}" for value in arrays["contract_index"]], dtype=str)
    if hierarchy == "window_random":
        bearing_group_values = np.asarray(arrays["bearing_id"], dtype=str)
        group_values = np.asarray([f"window_{index}" for index in range(len(labels))], dtype=str)
    else:
        bearing_group_values = bearing_groups(arrays["state"].astype(str), arrays["bearing_type"])
        group_values = record_groups if hierarchy == "record_grouped" else bearing_group_values
    metric_groups = bearing_group_values
    manifest_data = json.loads(manifests_path.read_text(encoding="utf-8"))
    manifests = (manifest_data["window_random"]["splits"] if hierarchy == "window_random" else manifest_data[hierarchy])[:count]
    candidates = [_candidate_with_view(candidate, "envelope_log_power") for candidate in build_g2_candidates(random_state=20_260_730)]
    fixed = next(
        candidate for candidate in candidates
        if candidate.family == "logistic_l2" and candidate.hyperparameter == {"C": 10.0}
    )

    split_rows: list[dict[str, Any]] = []
    flat_split: list[int] = []
    flat_groups: list[str] = []
    flat_targets: list[np.ndarray] = []
    flat_probabilities: list[np.ndarray] = []
    flat_predictions: list[np.ndarray] = []
    raw_split: list[int] = []
    raw_groups: list[str] = []
    raw_targets: list[np.ndarray] = []
    raw_probabilities: list[np.ndarray] = []
    raw_predictions: list[np.ndarray] = []
    for split_id, manifest in enumerate(manifests):
        outer_train = np.asarray(manifest["train_indices"], dtype=int)
        outer_test = np.asarray(manifest["test_indices"], dtype=int)
        selection_rows: list[dict[str, Any]] = []
        if selection == "fixed_prespecified":
            selected = fixed
        else:
            inner = manifest["inner"]
            selected, selection_rows = _choose_nested(
                features,
                labels,
                metric_groups,
                np.asarray(inner["train_indices"], dtype=int),
                np.asarray(inner["validation_indices"], dtype=int),
                candidates,
            )
        probabilities, _ = _predict(features, labels, outer_train, outer_test, selected)
        grouped, metrics = _aggregate_test(labels[outer_test], probabilities, metric_groups[outer_test])
        outer_candidate_metrics = []
        for candidate in candidates:
            candidate_probabilities, _ = _predict(
                features, labels, outer_train, outer_test, candidate
            )
            _, candidate_metrics = _aggregate_test(
                labels[outer_test], candidate_probabilities, metric_groups[outer_test]
            )
            outer_candidate_metrics.append({
                "family": candidate.family,
                "representation": candidate.representation,
                "hyperparameter": candidate.hyperparameter,
                "metrics": candidate_metrics,
            })
        split_rows.append({
            "split_id": int(split_id),
            "selected_family": selected.family,
            "selected_representation": selected.representation,
            "selected_hyperparameter": selected.hyperparameter,
            "outer_train_count": int(len(outer_train)),
            "outer_test_record_count": int(len(outer_test)),
            "metrics": metrics,
            "outer_candidate_metrics": outer_candidate_metrics,
            "inner_selection": selection_rows,
        })
        flat_split.extend([split_id] * len(grouped["groups"]))
        flat_groups.extend(grouped["groups"].tolist())
        flat_targets.append(grouped["targets"])
        flat_probabilities.append(grouped["probabilities"])
        flat_predictions.append(grouped["predictions"])
        raw_split.extend([split_id] * len(outer_test))
        raw_groups.extend(metric_groups[outer_test].tolist())
        raw_targets.append(labels[outer_test])
        raw_probabilities.append(probabilities.astype(np.float32))
        raw_predictions.append((probabilities >= 0.5).astype(np.int8))

    stem = f"hanoi_hust_{hierarchy}_{selection}"
    output_root = OUTPUT_DIR if count == 100 else OUTPUT_DIR / "smoke"
    prediction_path = output_root / "unit_level_predictions" / f"{stem}.npz"
    result_path = output_root / f"{stem}.json"
    _atomic_npz(prediction_path, {
        "split_id": np.asarray(flat_split, dtype=np.int16),
        "groups": np.asarray(flat_groups, dtype="U"),
        "targets": np.concatenate(flat_targets, axis=0).astype(np.int8),
        "probabilities": np.concatenate(flat_probabilities, axis=0).astype(np.float32),
        "predictions": np.concatenate(flat_predictions, axis=0).astype(np.int8),
    })
    raw_prediction_path = output_root / "unit_level_predictions" / f"{stem}_record_predictions.npz"
    _atomic_npz(raw_prediction_path, {
        "split_id": np.asarray(raw_split, dtype=np.int16),
        "groups": np.asarray(raw_groups, dtype="U"),
        "targets": np.concatenate(raw_targets, axis=0).astype(np.int8),
        "probabilities": np.concatenate(raw_probabilities, axis=0).astype(np.float32),
        "predictions": np.concatenate(raw_predictions, axis=0).astype(np.int8),
    })
    result = {
        "stage": "hanoi_hust_as_g2_protocol_cell",
        "schema_version": 1,
        "status": (
            "completed_descriptive_only" if hierarchy == "window_random" and count == 100
            else "completed" if count == 100
            else "smoke_partial_not_for_manuscript"
        ),
        "hierarchy": hierarchy,
        "selection": selection,
        "split_count": len(split_rows),
        "cache_sha256": _sha256(data_path),
        "manifest_sha256": _sha256(manifests_path),
        "representation": "envelope_log_power",
        "information_budget": "I0_source_only",
        "independent_metric_unit": "physical_bearing",
        "inference_allowed": hierarchy != "window_random",
        "primary_metric": "bearing_level_macro_component_auroc",
        "splits": split_rows,
        "predictions": {
            "path": prediction_path.relative_to(ROOT).as_posix(),
            "sha256": _sha256(prediction_path),
        },
        "record_predictions": {
            "path": raw_prediction_path.relative_to(ROOT).as_posix(),
            "sha256": _sha256(raw_prediction_path),
        },
    }
    _atomic_json(result_path, result)
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--hierarchy", choices=("record_grouped", "bearing_grouped", "window_random"), required=True)
    parser.add_argument("--selection", choices=("fixed_prespecified", "nested_selection"), required=True)
    parser.add_argument("--count", type=int, default=100)
    args = parser.parse_args()
    result = run_cell(hierarchy=args.hierarchy, selection=args.selection, count=args.count)
    print(json.dumps({
        "stage": result["stage"],
        "hierarchy": result["hierarchy"],
        "selection": result["selection"],
        "split_count": result["split_count"],
        "prediction_path": result["predictions"]["path"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
