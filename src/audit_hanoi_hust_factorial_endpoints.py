"""Audit the manuscript's crossed G2 endpoints from retained record predictions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from electrical_fm.hanoi_hust_baselines import (
    aggregate_group_predictions,
    compute_multilabel_metrics,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "g2"
PREDICTIONS = RESULTS / "unit_level_predictions"
EXPECTED = RESULTS / "factorial_endpoint_metrics.json"
STEMS = (
    "hanoi_hust_record_grouped_fixed_prespecified",
    "hanoi_hust_bearing_grouped_fixed_prespecified",
    "hanoi_hust_bearing_grouped_nested_selection",
)
FIELDS = (
    "record_auroc",
    "record_exact",
    "bearing_auroc",
    "bearing_exact",
    "n_records",
    "n_bearings",
)


def _rebuild_rows(stem: str) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    with np.load(PREDICTIONS / f"{stem}_record_predictions.npz", allow_pickle=False) as data:
        raw = {name: np.asarray(data[name]) for name in data.files}

    rows: list[dict[str, Any]] = []
    split_ids: list[int] = []
    groups: list[str] = []
    targets: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    for split_id in np.unique(raw["split_id"]):
        mask = raw["split_id"] == split_id
        record_metrics = compute_multilabel_metrics(
            raw["targets"][mask], raw["probabilities"][mask], raw["predictions"][mask]
        )
        grouped = aggregate_group_predictions(
            raw["targets"][mask], raw["probabilities"][mask], raw["groups"][mask]
        )
        bearing_metrics = compute_multilabel_metrics(
            grouped["targets"], grouped["probabilities"], grouped["predictions"]
        )
        rows.append(
            {
                "split": int(split_id),
                "record_auroc": record_metrics["mean_component_auroc"],
                "record_exact": record_metrics["exact_set_accuracy"],
                "bearing_auroc": bearing_metrics["mean_component_auroc"],
                "bearing_exact": bearing_metrics["exact_set_accuracy"],
                "n_records": int(mask.sum()),
                "n_bearings": int(len(grouped["groups"])),
            }
        )
        split_ids.extend([int(split_id)] * len(grouped["groups"]))
        groups.extend(grouped["groups"].tolist())
        targets.append(grouped["targets"])
        probabilities.append(grouped["probabilities"])
        predictions.append(grouped["predictions"])

    aggregate_arrays = {
        "split_id": np.asarray(split_ids, dtype=np.int16),
        "groups": np.asarray(groups, dtype="U"),
        "targets": np.concatenate(targets).astype(np.int8),
        "probabilities": np.concatenate(probabilities).astype(np.float32),
        "predictions": np.concatenate(predictions).astype(np.int8),
    }
    return rows, aggregate_arrays


def build_audit() -> dict[str, Any]:
    expected = json.loads(EXPECTED.read_text(encoding="utf-8"))
    failures: list[str] = []
    headline: dict[str, dict[str, float]] = {}
    for stem in STEMS:
        rebuilt, aggregate_arrays = _rebuild_rows(stem)
        stored_rows = expected.get(stem)
        if stored_rows is None or len(stored_rows) != 100:
            failures.append(f"missing or incomplete endpoint rows: {stem}")
            continue
        if len(rebuilt) != len(stored_rows):
            failures.append(f"split count mismatch: {stem}")
            continue
        for observed, stored in zip(rebuilt, stored_rows, strict=True):
            if observed["split"] != stored["split"]:
                failures.append(f"split identifier mismatch: {stem}")
                break
            for field in FIELDS:
                if not np.isclose(observed[field], stored[field], rtol=0.0, atol=1e-12):
                    failures.append(f"endpoint mismatch: {stem}:{stored['split']}:{field}")
                    break

        with np.load(PREDICTIONS / f"{stem}.npz", allow_pickle=False) as data:
            stored_aggregate = {name: np.asarray(data[name]) for name in data.files}
        for name, observed in aggregate_arrays.items():
            stored = stored_aggregate.get(name)
            if stored is None or stored.shape != observed.shape:
                failures.append(f"aggregate array shape mismatch: {stem}:{name}")
            elif np.issubdtype(observed.dtype, np.floating):
                if not np.allclose(stored, observed, rtol=0.0, atol=1e-7):
                    failures.append(f"aggregate probability mismatch: {stem}:{name}")
            elif not np.array_equal(stored, observed):
                failures.append(f"aggregate array mismatch: {stem}:{name}")

        headline[stem] = {
            "record_auroc": float(np.mean([row["record_auroc"] for row in rebuilt])),
            "record_exact": float(np.mean([row["record_exact"] for row in rebuilt])),
            "bearing_auroc": float(np.mean([row["bearing_auroc"] for row in rebuilt])),
            "bearing_exact": float(np.mean([row["bearing_exact"] for row in rebuilt])),
        }

    return {
        "stage": "hanoi_hust_factorial_endpoint_audit",
        "status": "passed" if not failures else "failed",
        "cell_count": len(headline),
        "split_count_per_cell": 100,
        "headline": headline,
        "failures": failures,
    }


def main() -> None:
    result = build_audit()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
