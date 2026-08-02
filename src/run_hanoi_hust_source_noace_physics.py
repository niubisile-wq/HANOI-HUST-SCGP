"""Run frozen source-only NOACE-Physics baselines on HANOI HUST cache."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io

from download_hanoi_hust import ARCHIVE
from electrical_fm.hanoi_hust_baselines import bearing_groups
from electrical_fm.hanoi_hust_noace import evaluate_noace_candidate


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "artifacts" / "hanoi_hust" / "source_features.npz"
METADATA = ROOT / "artifacts" / "hanoi_hust" / "source_features_metadata.json"
SOURCE_FREEZE = ROOT / "research" / "HANOI_HUST_SOURCE_ACCESS_FREEZE.json"
DEFAULT_OUTPUT = ROOT / "results" / "analysis" / "hanoi_hust_noace_physics.json"
DEFAULT_SELECTED_PREDICTIONS = (
    ROOT / "results" / "analysis" / "hanoi_hust_noace_physics_selected_predictions.npz"
)
FREQUENCY_BINS = 64


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
        json.dumps(_jsonable(payload), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(staging, path)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix(".writing")
    with staging.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(staging, path)


def _load_mat_keys_and_speed(member_path: str) -> tuple[dict[str, np.ndarray], float | None]:
    with zipfile.ZipFile(ARCHIVE) as archive:
        with archive.open(member_path) as handle:
            payload = handle.read()
    mat = scipy.io.loadmat(
        io.BytesIO(payload),
        squeeze_me=True,
        struct_as_record=False,
    )
    keys = sorted(key for key in mat if not key.startswith("__"))
    speed = None
    if "rpm" in mat:
        rpm = np.asarray(mat["rpm"], dtype=np.float64).reshape(-1)
        if rpm.size and np.isfinite(rpm).all():
            speed = float(np.median(rpm))
    return {key: np.asarray(mat[key]) for key in keys}, speed


def _to_order_space(values: np.ndarray, speed_rpm: float | None) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.ndim != 1:
        raise ValueError("Physics view expects 1-D values")
    freq_axis = np.linspace(0.0, 25_600.0, len(values), dtype=np.float64)
    if speed_rpm is None or speed_rpm <= 0:
        return np.zeros_like(values)
    shaft_hz = speed_rpm / 60.0
    order_axis = np.linspace(0.5, 256.0, len(values), dtype=np.float64)
    source_freq = np.clip(order_axis * shaft_hz, freq_axis[0], freq_axis[-1])
    return np.interp(source_freq, freq_axis, values)


def _build_physics_views(
    arrays: dict[str, np.ndarray],
    contracts: list[dict[str, Any]],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    statistics = np.asarray(arrays["statistics"], dtype=np.float32).mean(axis=1)
    fixed = np.asarray(arrays["fixed_log_power"], dtype=np.float32).mean(axis=1)
    envelope = np.asarray(arrays["envelope_log_power"], dtype=np.float32).mean(axis=1)

    order_fixed_rows = []
    order_envelope_rows = []
    speed_available = []
    speed_values = []
    raw_keys = []
    for row in contracts:
        keys, speed = _load_mat_keys_and_speed(row["path"])
        raw_keys.append(keys.get("rpm") is not None)
        if speed is None:
            order_fixed_rows.append(np.zeros(FREQUENCY_BINS * 2, dtype=np.float32))
            order_envelope_rows.append(np.zeros(FREQUENCY_BINS * 2, dtype=np.float32))
            speed_available.append(0.0)
            speed_values.append(0.0)
            continue
        fixed_row = _to_order_space(fixed[len(order_fixed_rows)], speed)
        envelope_row = _to_order_space(envelope[len(order_envelope_rows)], speed)
        order_fixed_rows.append(fixed_row.astype(np.float32, copy=False))
        order_envelope_rows.append(envelope_row.astype(np.float32, copy=False))
        speed_available.append(1.0)
        speed_values.append(float(speed))

    speed_available_arr = np.asarray(speed_available, dtype=np.float32)[:, None]
    speed_values_arr = np.asarray(speed_values, dtype=np.float32)[:, None]
    speed_log_arr = np.log1p(np.maximum(speed_values_arr, 0.0)).astype(np.float32)

    physics_frequency = np.concatenate([statistics, fixed, envelope], axis=1)
    physics_order = np.concatenate(
        [statistics, np.stack(order_fixed_rows), np.stack(order_envelope_rows), speed_available_arr, speed_values_arr, speed_log_arr],
        axis=1,
    )
    physics_hybrid = np.concatenate(
        [
            statistics,
            fixed,
            envelope,
            np.stack(order_fixed_rows),
            np.stack(order_envelope_rows),
            speed_available_arr,
            speed_values_arr,
            speed_log_arr,
        ],
        axis=1,
    )
    views = {
        "physics_frequency": physics_frequency.astype(np.float32),
        "physics_order": physics_order.astype(np.float32),
        "physics_hybrid": physics_hybrid.astype(np.float32),
    }
    summary = {
        "speed_available_count": int(np.sum(speed_available_arr)),
        "speed_missing_count": int(len(speed_available_arr) - np.sum(speed_available_arr)),
        "speed_available_fraction": float(np.mean(speed_available_arr)),
        "rpm_mat_key_present_count": int(np.sum(raw_keys)),
    }
    return views, summary


def build_noace_physics(
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

    contracts = metadata["contracts"]
    views, physics_summary = _build_physics_views(arrays, contracts)
    states = arrays["state"].astype(str)
    groups = bearing_groups(states, arrays["bearing_type"])
    targets = []
    for state in states:
        row = [0, 0, 0]
        if state == "I":
            row[0] = 1
        elif state == "O":
            row[1] = 1
        elif state == "B":
            row[2] = 1
        targets.append(row)

    rows: list[dict[str, Any]] = []
    for representation, features in views.items():
        row = evaluate_noace_candidate(
            features=features,
            states=states,
            bearing_type=arrays["bearing_type"],
            load_w=arrays["load_w"],
            groups=groups,
            random_seed=random_state,
            representation=representation,
        )
        rows.append(row)

    champion = max(
        rows,
        key=lambda row: (
            row["mean_component_auroc"],
            row["exact_set_accuracy"],
            -row["mean_brier_score"],
            row["feature_dimension"],
            row["representation"],
        ),
    )
    selected = {
        key: value
        for key, value in champion.items()
        if key not in {"probabilities", "predictions", "targets"}
    }
    selected_predictions_payload = {
        "probabilities": np.asarray(champion["probabilities"], dtype=np.float32),
        "predictions": np.asarray(champion["predictions"], dtype=np.int8),
        "targets": np.asarray(champion["targets"], dtype=np.int8),
        "groups": np.asarray(groups, dtype="U"),
        "bearing_type": np.asarray(arrays["bearing_type"], dtype=np.int16),
        "load_w": np.asarray(arrays["load_w"], dtype=np.int16),
        "contract_index": np.asarray(arrays["contract_index"], dtype=np.int16),
        "representation": np.asarray([champion["representation"]], dtype="U"),
    }
    _atomic_npz(selected_predictions, selected_predictions_payload)
    candidate_table = [
        {
            key: value
            for key, value in row.items()
            if key not in {"probabilities", "predictions", "targets"}
        }
        for row in rows
    ]
    payload = {
        "stage": "hanoi_hust_noace_physics",
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
        "record_count": int(len(states)),
        "bearing_count": int(len(np.unique(groups))),
        "candidate_count": len(rows),
        "selected": selected,
        "candidate_table": candidate_table,
        "selected_predictions": _as_artifact_path(selected_predictions),
        "physics_summary": physics_summary,
        "information_boundary": {
            "source_numeric_files_opened": int(len(states)),
            "compound_numeric_files_opened": 0,
            "transient_numeric_files_opened": 0,
        },
        "provenance": {
            "git_commit": _git("rev-parse", "HEAD"),
            "builder": "src/run_hanoi_hust_source_noace_physics.py",
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
            _jsonable(
                build_noace_physics(
                random_state=args.random_state,
                output=args.output,
                selected_predictions=args.selected_predictions,
                )
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
