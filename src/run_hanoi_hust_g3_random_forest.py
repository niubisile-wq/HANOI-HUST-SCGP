"""Run the frozen Random Forest G3 feature baseline on bearing splits."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from electrical_fm.hanoi_hust_baselines import (
    aggregate_group_predictions,
    bearing_groups,
    build_g3_random_forest_candidate,
    component_targets,
    compute_multilabel_metrics,
    representation_views,
)


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "artifacts" / "hanoi_hust" / "source_features.npz"
MANIFESTS = ROOT / "results" / "g2" / "split_manifests" / "hanoi_hust_g2_manifests.json"
OUTPUT = ROOT / "results" / "g3" / "hanoi_hust_random_forest_bearing_grouped.json"
PREDICTIONS = ROOT / "results" / "g3" / "unit_level_predictions" / "hanoi_hust_random_forest_bearing_grouped.npz"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix(".writing")
    staging.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(staging, path)


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix(".writing")
    with staging.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(staging, path)


def run(*, cache: Path = CACHE, manifests_path: Path = MANIFESTS) -> dict[str, Any]:
    with np.load(cache, allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    features = representation_views(arrays)["envelope_log_power"]
    labels = component_targets(arrays["state"].astype(str))
    groups = bearing_groups(arrays["state"].astype(str), arrays["bearing_type"])
    manifest_data = json.loads(manifests_path.read_text(encoding="utf-8"))
    manifests = manifest_data["bearing_grouped"]
    candidate = build_g3_random_forest_candidate(random_state=20_260_730)
    rows: list[dict[str, Any]] = []
    split_ids: list[int] = []
    grouped_ids: list[str] = []
    grouped_targets: list[np.ndarray] = []
    grouped_probabilities: list[np.ndarray] = []
    grouped_predictions: list[np.ndarray] = []
    for split_id, manifest in enumerate(manifests):
        train = np.asarray(manifest["train_indices"], dtype=int)
        test = np.asarray(manifest["test_indices"], dtype=int)
        probabilities = np.zeros((len(test), labels.shape[1]), dtype=np.float64)
        for head in range(labels.shape[1]):
            estimator = candidate.build()
            estimator.fit(features[train], labels[train, head])
            probabilities[:, head] = estimator.predict_proba(features[test])[:, 1]
        grouped = aggregate_group_predictions(labels[test], probabilities, groups[test])
        metrics = compute_multilabel_metrics(grouped["targets"], grouped["probabilities"], grouped["predictions"])
        rows.append({"split_id": split_id, "metrics": metrics})
        split_ids.extend([split_id] * len(grouped["groups"]))
        grouped_ids.extend(grouped["groups"].tolist())
        grouped_targets.append(grouped["targets"])
        grouped_probabilities.append(grouped["probabilities"])
        grouped_predictions.append(grouped["predictions"])
    _atomic_npz(PREDICTIONS, {
        "split_id": np.asarray(split_ids, dtype=np.int16),
        "groups": np.asarray(grouped_ids, dtype="U"),
        "targets": np.concatenate(grouped_targets, axis=0).astype(np.int8),
        "probabilities": np.concatenate(grouped_probabilities, axis=0).astype(np.float32),
        "predictions": np.concatenate(grouped_predictions, axis=0).astype(np.int8),
    })
    metric_names = ("mean_component_auroc", "mean_component_aupr", "mean_component_balanced_accuracy", "mean_component_macro_f1", "mean_brier_score", "exact_set_accuracy", "hamming_loss")
    summary = {
        name: {"mean": float(np.mean([row["metrics"][name] for row in rows])), "std": float(np.std([row["metrics"][name] for row in rows], ddof=1))}
        for name in metric_names
    }
    result = {
        "stage": "hanoi_hust_as_g3_random_forest",
        "schema_version": 1,
        "status": "completed",
        "information_budget": "I0_source_only",
        "independent_metric_unit": "physical_bearing",
        "candidate": {"family": candidate.family, "hyperparameter": candidate.hyperparameter},
        "cache": {"path": cache.relative_to(ROOT).as_posix(), "sha256": _sha256(cache)},
        "manifests_sha256": _sha256(manifests_path),
        "split_count": len(rows),
        "summary": summary,
        "splits": rows,
        "predictions": {"path": PREDICTIONS.relative_to(ROOT).as_posix(), "sha256": _sha256(PREDICTIONS)},
    }
    _atomic_json(OUTPUT, result)
    return result


def main() -> None:
    print(json.dumps(run(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
