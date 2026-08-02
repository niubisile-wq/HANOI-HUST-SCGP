"""Audit selected HANOI HUST predictions at the physical-bearing level."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from electrical_fm.hanoi_hust_baselines import (
    COMPONENT_NAMES,
    aggregate_group_predictions,
    compute_multilabel_metrics,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results" / "development" / "hanoi_hust_source_selected_predictions.npz"
DEFAULT_OUTPUT = ROOT / "results" / "analysis" / "hanoi_hust_source_bearing_level_audit.json"
DEFAULT_GROUPED_OUTPUT = ROOT / "results" / "analysis" / "hanoi_hust_source_bearing_level_predictions.npz"


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


def build_audit(
    *,
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
    grouped_output: Path = DEFAULT_GROUPED_OUTPUT,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Create a bearing-level audit without retraining or opening raw MAT files."""
    with np.load(input_path, allow_pickle=False) as payload:
        required = {"probabilities", "predictions", "targets", "groups"}
        missing = required - set(payload.files)
        if missing:
            raise ValueError(f"selected predictions missing arrays: {sorted(missing)}")
        arrays = {name: np.asarray(payload[name]) for name in required}

    grouped = aggregate_group_predictions(
        arrays["targets"],
        arrays["probabilities"],
        arrays["groups"],
        threshold=threshold,
    )
    metrics = compute_multilabel_metrics(
        grouped["targets"], grouped["probabilities"], grouped["predictions"]
    )
    _atomic_npz(grouped_output, grouped)
    result = {
        "stage": "hanoi_hust_source_bearing_level_audit",
        "schema_version": 1,
        "status": "completed",
        "aggregation": {
            "independent_unit": "physical_bearing",
            "probability_rule": "arithmetic_mean_across_records",
            "threshold": float(threshold),
            "component_names": list(COMPONENT_NAMES),
            "labels_must_be_invariant_within_group": True,
        },
        "input": {
            "path": input_path.relative_to(ROOT).as_posix(),
            "sha256": _sha256(input_path),
            "record_count": int(len(arrays["targets"])),
            "record_group_count": int(len(np.unique(arrays["groups"]))),
        },
        "output": {
            "path": grouped_output.relative_to(ROOT).as_posix(),
            "sha256": _sha256(grouped_output),
        },
        "metrics": metrics,
    }
    _atomic_json(output_path, result)
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--grouped-output", type=Path, default=DEFAULT_GROUPED_OUTPUT)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    print(json.dumps(build_audit(
        input_path=args.input,
        output_path=args.output,
        grouped_output=args.grouped_output,
        threshold=args.threshold,
    ), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
