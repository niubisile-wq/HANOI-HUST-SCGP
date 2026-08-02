"""Run the pre-specified common-bearing G2 paired primary split."""

from __future__ import annotations

import hashlib
import itertools
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from electrical_fm.hanoi_hust_baselines import (
    aggregate_group_predictions,
    bearing_groups,
    build_g2_candidates,
    component_targets,
    compute_multilabel_metrics,
    representation_views,
)


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "artifacts" / "hanoi_hust" / "source_features.npz"
OUTPUT = ROOT / "results" / "g2" / "primary_paired_split.json"

TEST_BEARINGS = ("N4", "N5", "I4", "O4", "B5")


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


def _fit_predict(features: np.ndarray, labels: np.ndarray, train: np.ndarray, test: np.ndarray):
    candidate = next(
        candidate for candidate in build_g2_candidates(random_state=20_260_730)
        if candidate.family == "logistic_l2" and candidate.hyperparameter == {"C": 10.0}
    )
    probabilities = np.zeros((len(test), labels.shape[1]), dtype=np.float64)
    for head in range(labels.shape[1]):
        estimator = candidate.build()
        estimator.fit(features[train], labels[train, head])
        probabilities[:, head] = estimator.predict_proba(features[test])[:, 1]
    return probabilities


def _paired_sign_pvalue(values: np.ndarray) -> float:
    observed = abs(float(values.mean()))
    exceed = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        permuted = abs(float((values * np.asarray(signs)).mean()))
        exceed += int(permuted >= observed)
        total += 1
    return float(exceed / total)


def _paired_metrics(record: dict[str, np.ndarray], bearing: dict[str, np.ndarray]) -> dict[str, Any]:
    record_metrics = compute_multilabel_metrics(record["targets"], record["probabilities"], record["predictions"])
    bearing_metrics = compute_multilabel_metrics(bearing["targets"], bearing["probabilities"], bearing["predictions"])
    record_correct = np.all(record["targets"] == record["predictions"], axis=1).astype(float)
    bearing_correct = np.all(bearing["targets"] == bearing["predictions"], axis=1).astype(float)
    record_hamming = np.mean(record["targets"] != record["predictions"], axis=1)
    bearing_hamming = np.mean(bearing["targets"] != bearing["predictions"], axis=1)
    return {
        "record_grouped": record_metrics,
        "bearing_grouped": bearing_metrics,
        "paired_unit_differences": {
            "bearing_minus_record_exact_indicator": (bearing_correct - record_correct).tolist(),
            "bearing_minus_record_hamming_loss": (bearing_hamming - record_hamming).tolist(),
        },
        "paired_sign_permutation": {
            "exact_indicator_p": _paired_sign_pvalue(bearing_correct - record_correct),
            "hamming_loss_p": _paired_sign_pvalue(bearing_hamming - record_hamming),
        },
    }


def build_result(*, cache: Path = CACHE, output: Path = OUTPUT) -> dict[str, Any]:
    with np.load(cache, allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    views = representation_views(arrays)
    features = views["envelope_log_power"]
    labels = component_targets(arrays["state"].astype(str))
    groups = bearing_groups(arrays["state"].astype(str), arrays["bearing_type"])
    missing = sorted(set(TEST_BEARINGS) - set(groups.tolist()))
    if missing:
        raise ValueError(f"pre-specified primary bearings missing: {missing}")

    bearing_test = np.flatnonzero(np.isin(groups, TEST_BEARINGS))
    bearing_train = np.flatnonzero(~np.isin(groups, TEST_BEARINGS))
    selected_record_indices = []
    for group in TEST_BEARINGS:
        members = np.flatnonzero(groups == group)
        selected_record_indices.append(int(members[np.argmin(arrays["contract_index"][members])]))
    record_test = np.asarray(selected_record_indices, dtype=int)
    record_train = np.setdiff1d(np.arange(len(groups)), record_test)

    record_probabilities = _fit_predict(features, labels, record_train, record_test)
    bearing_probabilities = _fit_predict(features, labels, bearing_train, bearing_test)
    record_grouped = aggregate_group_predictions(labels[record_test], record_probabilities, groups[record_test])
    bearing_grouped = aggregate_group_predictions(labels[bearing_test], bearing_probabilities, groups[bearing_test])
    order_record = {group: index for index, group in enumerate(record_grouped["groups"].tolist())}
    order_bearing = {group: index for index, group in enumerate(bearing_grouped["groups"].tolist())}
    if set(order_record) != set(order_bearing) or set(order_record) != set(TEST_BEARINGS):
        raise RuntimeError("paired primary split unit sets are not identical")
    order = np.asarray([order_record[group] for group in TEST_BEARINGS], dtype=int)
    bearing_order = np.asarray([order_bearing[group] for group in TEST_BEARINGS], dtype=int)
    record_grouped = {name: values[order] if name != "groups" else values[order] for name, values in record_grouped.items()}
    bearing_grouped = {name: values[bearing_order] if name != "groups" else values[bearing_order] for name, values in bearing_grouped.items()}
    result = {
        "stage": "hanoi_hust_as_g2_primary_paired_split",
        "schema_version": 1,
        "status": "completed",
        "pre_specified_test_bearings": list(TEST_BEARINGS),
        "selection": "fixed_prespecified_logistic_l2_C10",
        "cache": {"path": cache.relative_to(ROOT).as_posix(), "sha256": _sha256(cache)},
        "record_grouped": {
            "train_record_count": int(len(record_train)),
            "test_record_count": int(len(record_test)),
        },
        "bearing_grouped": {
            "train_record_count": int(len(bearing_train)),
            "test_record_count": int(len(bearing_test)),
        },
        "metrics": _paired_metrics(record_grouped, bearing_grouped),
        "interpretation": "paired comparison on the same five pre-specified physical bearings; small-sample exact sign permutation",
    }
    _atomic_json(output, result)
    return result


def main() -> None:
    print(json.dumps(build_result(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
