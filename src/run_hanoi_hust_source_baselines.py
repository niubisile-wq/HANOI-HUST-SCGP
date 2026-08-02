"""Run frozen source-only baselines on the cached HANOI HUST source records."""

from __future__ import annotations

import hashlib
import json
import os
import warnings
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from electrical_fm.hanoi_hust_baselines import (
    bearing_groups,
    component_targets,
    evaluate_candidates,
    representation_views,
)


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "artifacts" / "hanoi_hust" / "source_features.npz"
METADATA = ROOT / "artifacts" / "hanoi_hust" / "source_features_metadata.json"
DEFAULT_OUTPUT = ROOT / "results" / "development" / "hanoi_hust_source_baselines.json"
DEFAULT_SELECTED_PREDICTIONS = (
    ROOT / "results" / "development" / "hanoi_hust_source_selected_predictions.npz"
)

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"sklearn\.svm\._base",
)


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported Hanoi HUST JSON value: {type(value).__name__}")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix(".writing")
    staging.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False, default=_json_default),
        encoding="utf-8",
    )
    os.replace(staging, path)


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix(".writing")
    with staging.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(staging, path)


def build_result(
    *,
    random_state: int = 20_260_727,
    output: Path = DEFAULT_OUTPUT,
    selected_predictions: Path = DEFAULT_SELECTED_PREDICTIONS,
) -> dict[str, Any]:
    """Evaluate the frozen source-only comparator grid."""
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    if metadata.get("status") != "source_features_complete_compound_sealed":
        raise RuntimeError("HANOI HUST source cache is not complete")
    if _sha256(CACHE) != metadata.get("cache", {}).get("sha256"):
        raise RuntimeError("HANOI HUST source cache hash changed")
    with np.load(CACHE, allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    views = representation_views(arrays)
    states = arrays["state"].astype(str)
    targets = component_targets(states)
    groups = bearing_groups(states, arrays["bearing_type"])
    ranked = evaluate_candidates(views, targets, groups, random_state=random_state)
    selected = ranked[0]
    _atomic_npz(
        selected_predictions,
        {
            "probabilities": selected["probabilities"].astype(np.float32),
            "predictions": selected["predictions"].astype(np.int8),
            "targets": selected["targets"].astype(np.int8),
            "groups": groups.astype("U"),
            "bearing_type": arrays["bearing_type"].astype(np.int16),
            "load_w": arrays["load_w"].astype(np.int16),
            "contract_index": arrays["contract_index"].astype(np.int16),
        },
    )
    selected_summary = {
        key: value
        for key, value in selected.items()
        if key
        not in {
            "probabilities",
            "predictions",
            "targets",
        }
    }
    result = {
        "stage": "hanoi_hust_source_baselines",
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
        "candidate_count": int(len(ranked)),
        "selected": selected_summary,
        "candidate_table": [
            {
                key: value
                for key, value in row.items()
                if key not in {"probabilities", "predictions", "targets"}
            }
            for row in ranked
        ],
        "information_boundary": {
            "source_numeric_files_opened": int(len(states)),
            "compound_numeric_files_opened": 0,
            "transient_numeric_files_opened": 0,
        },
        "provenance": {
            "git_commit": _git("rev-parse", "HEAD"),
            "builder": "src/run_hanoi_hust_source_baselines.py",
            "random_state": int(random_state),
        },
    }
    _atomic_json(output, result)
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--random-state", type=int, default=20_260_727)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--selected-predictions",
        type=Path,
        default=DEFAULT_SELECTED_PREDICTIONS,
    )
    args = parser.parse_args()
    print(
        json.dumps(
            build_result(
                random_state=args.random_state,
                output=args.output,
                selected_predictions=args.selected_predictions,
            ),
            indent=2,
            ensure_ascii=False,
            default=_json_default,
        )
    )


if __name__ == "__main__":
    main()
