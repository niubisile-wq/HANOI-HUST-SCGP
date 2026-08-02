"""Build deterministic G2 split manifests without fitting any model."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from electrical_fm.hanoi_hust_baselines import bearing_groups, component_targets


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "artifacts" / "hanoi_hust" / "source_features.npz"
WINDOW_CACHE = ROOT / "artifacts" / "hanoi_hust_window" / "source_window_features.npz"
OUTPUT = ROOT / "results" / "g2" / "split_manifests" / "hanoi_hust_g2_manifests.json"
MANIFEST_COUNT = 100
TEST_FRACTION = 0.25


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


def _valid_support(labels: np.ndarray, indices: np.ndarray) -> bool:
    """Require both classes for every component in train and test."""
    subset = labels[indices]
    return all(np.unique(subset[:, head]).size == 2 for head in range(labels.shape[1]))


def _split_groups(
    labels: np.ndarray,
    group_values: np.ndarray,
    *,
    seed: int,
) -> dict[str, Any]:
    unique_groups = np.unique(group_values)
    if len(unique_groups) < 4:
        raise ValueError("G2 split requires at least four groups")
    for attempt in range(10_000):
        rng = np.random.default_rng(seed + attempt)
        permutation = rng.permutation(unique_groups)
        test_count = max(1, int(round(len(unique_groups) * TEST_FRACTION)))
        test_groups = np.sort(permutation[:test_count])
        train_groups = np.sort(permutation[test_count:])
        test_mask = np.isin(group_values, test_groups)
        train_mask = ~test_mask
        train_indices = np.flatnonzero(train_mask)
        test_indices = np.flatnonzero(test_mask)
        if _valid_support(labels, train_indices) and _valid_support(labels, test_indices):
            return {
                "seed": int(seed),
                "attempt": int(attempt),
                "group_count": int(len(unique_groups)),
                "train_groups": train_groups.astype(str).tolist(),
                "test_groups": test_groups.astype(str).tolist(),
                "train_indices": train_indices.astype(int).tolist(),
                "test_indices": test_indices.astype(int).tolist(),
            }
    raise RuntimeError(f"unable to find a supported split for seed {seed}")


def _inner_split(
    labels: np.ndarray,
    group_values: np.ndarray,
    outer_train_indices: np.ndarray,
    *,
    seed: int,
) -> dict[str, Any]:
    """Generate an inner split and map its indices back to the full cache."""
    local = _split_groups(
        labels[outer_train_indices],
        group_values[outer_train_indices],
        seed=seed,
    )
    return {
        "seed": local["seed"],
        "attempt": local["attempt"],
        "group_count": local["group_count"],
        "train_groups": local["train_groups"],
        "validation_groups": local["test_groups"],
        "train_indices": outer_train_indices[np.asarray(local["train_indices"], dtype=int)].astype(int).tolist(),
        "validation_indices": outer_train_indices[np.asarray(local["test_indices"], dtype=int)].astype(int).tolist(),
    }


def build_manifests(
    *,
    cache: Path = CACHE,
    output: Path = OUTPUT,
    count: int = MANIFEST_COUNT,
) -> dict[str, Any]:
    with np.load(cache, allow_pickle=False) as payload:
        required = {"state", "bearing_type", "contract_index"}
        missing = required - set(payload.files)
        if missing:
            raise ValueError(f"feature cache missing manifest fields: {sorted(missing)}")
        states = np.asarray(payload["state"], dtype=str)
        bearing_type = np.asarray(payload["bearing_type"], dtype=np.int16)
        contract_index = np.asarray(payload["contract_index"], dtype=np.int16)
    labels = component_targets(states)
    bearing = bearing_groups(states, bearing_type)
    record = np.asarray([f"record_{value}" for value in contract_index], dtype=str)
    if len(np.unique(record)) != len(record):
        raise ValueError("record groups are not unique in the source cache")

    record_splits = []
    bearing_splits = []
    for i in range(count):
        record_outer = _split_groups(labels, record, seed=20_260_730 + i)
        record_outer["inner"] = _inner_split(
            labels,
            record,
            np.asarray(record_outer["train_indices"], dtype=int),
            seed=30_260_730 + i,
        )
        record_splits.append(record_outer)
        bearing_outer = _split_groups(labels, bearing, seed=20_270_730 + i)
        bearing_outer["inner"] = _inner_split(
            labels,
            bearing,
            np.asarray(bearing_outer["train_indices"], dtype=int),
            seed=30_270_730 + i,
        )
        bearing_splits.append(bearing_outer)
    with np.load(WINDOW_CACHE, allow_pickle=False) as window_payload:
        window_states = np.asarray(window_payload["state"], dtype=str)
    window_labels = component_targets(window_states)
    window_groups = np.asarray([f"window_{index}" for index in range(len(window_states))], dtype=str)
    window_splits = []
    for i in range(count):
        window_outer = _split_groups(window_labels, window_groups, seed=20_280_730 + i)
        window_outer["inner"] = _inner_split(
            window_labels,
            window_groups,
            np.asarray(window_outer["train_indices"], dtype=int),
            seed=30_280_730 + i,
        )
        window_splits.append(window_outer)
    payload = {
        "stage": "hanoi_hust_as_g2_split_manifests",
        "schema_version": 1,
        "status": "generated_before_g2_model_results",
        "cache": {
            "path": cache.relative_to(ROOT).as_posix(),
            "sha256": _sha256(cache),
            "record_count": int(len(states)),
            "bearing_count": int(len(np.unique(bearing))),
        },
        "window_cache": {
            "path": WINDOW_CACHE.relative_to(ROOT).as_posix(),
            "sha256": _sha256(WINDOW_CACHE),
            "window_count": int(len(window_states)),
        },
        "contract": {
            "split_manifest_count": int(count),
            "test_fraction": TEST_FRACTION,
            "independent_unit": "physical_bearing_for_primary_inference",
            "support_rule": "both_classes_for_each_component_in_train_and_test",
        },
        "record_grouped": record_splits,
        "bearing_grouped": bearing_splits,
        "window_random": {
            "status": "generated_descriptive_only",
            "inference_allowed": False,
            "splits": window_splits,
            "reason": "window_rows_are_repeated_measurements_within_physical_bearings",
        },
    }
    _atomic_json(output, payload)
    return payload


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--count", type=int, default=MANIFEST_COUNT)
    args = parser.parse_args()
    result = build_manifests(cache=args.cache, output=args.output, count=args.count)
    print(json.dumps({
        "stage": result["stage"],
        "record_grouped": len(result["record_grouped"]),
        "bearing_grouped": len(result["bearing_grouped"]),
        "window_random": {
            "status": result["window_random"]["status"],
            "split_count": len(result["window_random"]["splits"]),
            "inference_allowed": result["window_random"]["inference_allowed"],
        },
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
