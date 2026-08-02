"""Run frozen source-only NOACE baselines on HANOI HUST cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from electrical_fm.hanoi_hust_baselines import bearing_groups
from electrical_fm.hanoi_hust_noace import _select_view, evaluate_noace_candidate


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "artifacts" / "hanoi_hust" / "source_features.npz"
METADATA = ROOT / "artifacts" / "hanoi_hust" / "source_features_metadata.json"
DEFAULT_OUTPUT = ROOT / "results" / "analysis" / "hanoi_hust_noace_classical.json"
DEFAULT_GRID_OUTPUT = ROOT / "results" / "analysis" / "hanoi_hust_noace_grid.json"
DEFAULT_SELECTED_PREDICTIONS = (
    ROOT / "results" / "analysis" / "hanoi_hust_noace_classical_selected_predictions.npz"
)
DEFAULT_CANDIDATE_SELECTED_DIR = ROOT / "results" / "analysis"
DEFAULT_REPRESENTATIONS: tuple[str, ...] = (
    "statistics",
    "fixed_log_power",
    "envelope_log_power",
    "all",
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


def _run_candidate(
    arrays: dict[str, np.ndarray],
    *,
    representation: str,
    random_state: int,
    candidate_output: Path,
) -> dict[str, Any]:
    states = arrays["state"].astype(str)
    bearing_type = arrays["bearing_type"]
    groups = bearing_groups(states, bearing_type)
    features = _select_view(arrays, representation=representation)
    result = evaluate_noace_candidate(
        features=features,
        states=states,
        bearing_type=arrays["bearing_type"],
        load_w=arrays["load_w"],
        groups=groups,
        random_seed=random_state,
        representation=representation,
    )
    _atomic_npz(
        candidate_output,
        {
            "probabilities": result["group_probabilities"].astype(np.float32),
            "predictions": result["predictions"].astype(np.int8),
            "targets": result["targets"].astype(np.int8),
            "group_ids": np.asarray(result["group_ids"], dtype="U"),
            "bearing_type": arrays["bearing_type"].astype(np.int16),
            "load_w": arrays["load_w"].astype(np.int16),
            "contract_index": arrays["contract_index"].astype(np.int16),
            "representation": np.asarray([representation], dtype="U"),
        },
    )
    return {
        key: value
        for key, value in result.items()
        if key not in {"probabilities", "predictions", "targets", "group_probabilities"}
    }


def build_noace(
    *,
    random_state: int,
    output: Path,
    selected_predictions: Path,
    representation: str = "all",
) -> dict[str, Any]:
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    if metadata.get("status") != "source_features_complete_compound_sealed":
        raise RuntimeError("HANOI HUST source cache is not complete")
    if _sha256(CACHE) != metadata.get("cache", {}).get("sha256"):
        raise RuntimeError("HANOI HUST source cache hash changed")

    with np.load(CACHE, allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    bearing_count = int(
        len(np.unique(bearing_groups(arrays["state"], arrays["bearing_type"])))
    )
    rows = [
        _run_candidate(
            arrays,
            representation=representation,
            random_state=random_state,
            candidate_output=selected_predictions,
        )
    ]
    selected = rows[0]
    payload = {
        "stage": "hanoi_hust_noace_classical",
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
        "record_count": int(len(arrays["state"])),
        "bearing_count": bearing_count,
        "candidate_count": len(rows),
        "selected": selected,
        "result": selected,
        "selected_predictions": _as_artifact_path(selected_predictions),
        "information_boundary": {
            "source_numeric_files_opened": int(len(arrays["state"])),
            "compound_numeric_files_opened": 0,
            "transient_numeric_files_opened": 0,
        },
        "provenance": {
            "git_commit": _git("rev-parse", "HEAD"),
            "builder": "src/run_hanoi_hust_source_noace.py",
            "random_state": int(random_state),
        },
    }
    _atomic_json(output, payload)
    return payload


def build_noace_grid(
    *,
    random_state: int,
    output: Path,
    representations: tuple[str, ...],
    candidate_selected_dir: Path,
) -> dict[str, Any]:
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    if metadata.get("status") != "source_features_complete_compound_sealed":
        raise RuntimeError("HANOI HUST source cache is not complete")
    if _sha256(CACHE) != metadata.get("cache", {}).get("sha256"):
        raise RuntimeError("HANOI HUST source cache hash changed")

    with np.load(CACHE, allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}

    if not representations:
        raise ValueError("At least one NOACE representation is required")
    bearing_count = int(
        len(np.unique(bearing_groups(arrays["state"], arrays["bearing_type"])))
    )
    rows: list[dict[str, Any]] = []
    for representation in representations:
        candidate_output = candidate_selected_dir / f"noace_{representation}_selected_predictions.npz"
        row = _run_candidate(
            arrays,
            representation=representation,
            random_state=random_state,
            candidate_output=candidate_output,
        )
        row["selected_predictions"] = _as_artifact_path(candidate_output)
        rows.append(row)

    champion = max(
        rows,
        key=lambda row: (
            row["mean_component_auroc"],
            row["exact_set_accuracy"],
            -row["mean_brier_score"],
            row["feature_dimension"],
            row["family"],
            row["representation"],
            json.dumps(row["hyperparameter"], sort_keys=True),
        ),
    )
    selected_path = candidate_selected_dir / f'noace_{champion["representation"]}_selected_predictions.npz'
    payload = {
        "stage": "hanoi_hust_noace_grid",
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
        "record_count": int(len(arrays["state"])),
        "bearing_count": bearing_count,
        "candidate_count": len(rows),
        "selected": champion,
        "candidate_table": rows,
        "selected_predictions": _as_artifact_path(selected_path),
        "information_boundary": {
            "source_numeric_files_opened": int(len(arrays["state"])),
            "compound_numeric_files_opened": 0,
            "transient_numeric_files_opened": 0,
        },
        "provenance": {
            "git_commit": _git("rev-parse", "HEAD"),
            "builder": "src/run_hanoi_hust_source_noace.py",
            "random_state": int(random_state),
        },
    }
    _atomic_json(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-state", type=int, default=20_260_727)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--mode",
        choices=("classical", "grid"),
        default="classical",
    )
    parser.add_argument(
        "--representations",
        type=str,
        nargs="*",
        default=list(DEFAULT_REPRESENTATIONS),
    )
    parser.add_argument(
        "--selected-predictions",
        type=Path,
        default=DEFAULT_SELECTED_PREDICTIONS,
    )
    parser.add_argument(
        "--candidate-selected-dir",
        type=Path,
        default=DEFAULT_CANDIDATE_SELECTED_DIR,
    )
    args = parser.parse_args()
    output = args.output
    if args.mode == "grid" and args.output == DEFAULT_OUTPUT:
        output = DEFAULT_GRID_OUTPUT
    if args.mode == "classical":
        result = build_noace(
            random_state=args.random_state,
            output=output,
            selected_predictions=args.selected_predictions,
        )
    else:
        result = build_noace_grid(
            random_state=args.random_state,
            output=output,
            representations=tuple(args.representations),
            candidate_selected_dir=args.candidate_selected_dir,
        )
    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
