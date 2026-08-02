"""Run the first-batch high-ROI HANOI HUST baselines.

This script evaluates three additional comparator families on the frozen
HANOI HUST source cache:

- adaptive_signal_processing
- physics_aware_time_frequency
- multiscale_time_series

The goal is not to replace the frozen source champion, but to test whether the
protocol-sensitivity story survives stronger modern DSP-style comparators.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from electrical_fm.hanoi_hust_baselines import (
    Candidate,
    bearing_groups,
    component_targets,
    evaluate_candidate,
    representation_views,
)


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "artifacts" / "hanoi_hust" / "source_features.npz"
METADATA = ROOT / "artifacts" / "hanoi_hust" / "source_features_metadata.json"
DEFAULT_OUTPUT = ROOT / "results" / "analysis" / "hanoi_hust_high_roi_baselines.json"
DEFAULT_SELECTED_PREDICTIONS = (
    ROOT / "results" / "analysis" / "hanoi_hust_high_roi_baselines_selected_predictions.npz"
)
DEFAULT_MARKDOWN = (
    ROOT
    / "research"
    / "HANOI_HUST_20260727"
    / "HANOI_HUST_第一批高ROI基线结果页_20260728.md"
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
    raise TypeError(f"Unsupported HANOI HUST JSON value: {type(value).__name__}")


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


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix(".writing")
    staging.write_text(text, encoding="utf-8")
    os.replace(staging, path)


def _load_cache() -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    if metadata.get("status") != "source_features_complete_compound_sealed":
        raise RuntimeError("HANOI HUST source cache is not complete")
    if _sha256(CACHE) != metadata.get("cache", {}).get("sha256"):
        raise RuntimeError("HANOI HUST source cache hash changed")
    with np.load(CACHE, allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    return arrays, metadata


def _flatten_views(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return representation_views(arrays)


def _pool_blocks(features: np.ndarray, bins: int) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("Pooling expects a 2-D feature matrix")
    if values.shape[1] % bins != 0:
        raise ValueError(f"Feature dimension {values.shape[1]} is not divisible by {bins}")
    blocks = values.reshape(len(values), bins, -1)
    mean = blocks.mean(axis=2)
    std = blocks.std(axis=2, ddof=0)
    return np.concatenate([mean, std], axis=1).astype(np.float32)


def _adaptive_spectral_view(base: dict[str, np.ndarray]) -> np.ndarray:
    statistics = base["statistics"]
    fixed = base["fixed_log_power"]
    envelope = base["envelope_log_power"]
    delta = envelope - fixed
    abs_delta = np.abs(delta)
    sum_view = 0.5 * (envelope + fixed)
    ratio = delta / np.maximum(np.abs(fixed), 1e-6)
    grad_delta = np.diff(delta, axis=1, prepend=delta[:, :1])
    grad_envelope = np.diff(envelope, axis=1, prepend=envelope[:, :1])
    grad_fixed = np.diff(fixed, axis=1, prepend=fixed[:, :1])
    pooled_delta = np.concatenate(
        [_pool_blocks(delta, bins) for bins in (4, 8, 16)],
        axis=1,
    )
    return np.concatenate(
        [
            statistics,
            fixed,
            envelope,
            delta,
            abs_delta,
            sum_view,
            ratio,
            grad_delta,
            grad_envelope,
            grad_fixed,
            pooled_delta,
        ],
        axis=1,
    )


def _physics_aware_view(base: dict[str, np.ndarray]) -> np.ndarray:
    statistics = base["statistics"]
    fixed = base["fixed_log_power"]
    envelope = base["envelope_log_power"]
    delta = envelope - fixed
    pooled_statistics = np.concatenate(
        [_pool_blocks(statistics, bins) for bins in (3, 6, 7)],
        axis=1,
    )
    pooled_fixed = np.concatenate(
        [_pool_blocks(fixed, bins) for bins in (4, 8, 16)],
        axis=1,
    )
    pooled_envelope = np.concatenate(
        [_pool_blocks(envelope, bins) for bins in (4, 8, 16)],
        axis=1,
    )
    pooled_delta = np.concatenate(
        [_pool_blocks(delta, bins) for bins in (4, 8, 16)],
        axis=1,
    )
    slope_fixed = np.diff(fixed, axis=1, prepend=fixed[:, :1])
    slope_envelope = np.diff(envelope, axis=1, prepend=envelope[:, :1])
    return np.concatenate(
        [
            statistics,
            pooled_statistics,
            pooled_fixed,
            pooled_envelope,
            pooled_delta,
            slope_fixed,
            slope_envelope,
        ],
        axis=1,
    )


def _multiscale_unified_view(base: dict[str, np.ndarray]) -> np.ndarray:
    all_view = base["all"]
    pooled = np.concatenate(
        [_pool_blocks(all_view, bins) for bins in (3, 4, 6, 12)],
        axis=1,
    )
    gradients = np.diff(all_view, axis=1, prepend=all_view[:, :1])
    return np.concatenate([all_view, pooled, gradients], axis=1)


def _build_candidates(random_state: int) -> tuple[Candidate, ...]:
    candidates: list[Candidate] = []

    for c_value in (0.1, 1.0, 10.0):
        candidates.append(
            Candidate(
                family="adaptive_signal_processing",
                representation="adaptive_spectral",
                hyperparameter={"C": c_value},
                build=lambda value=c_value: make_pipeline(
                    StandardScaler(),
                    LogisticRegression(
                        C=value,
                        class_weight="balanced",
                        max_iter=5_000,
                        random_state=random_state,
                    ),
                ),
            )
        )

    for c_value in (0.1, 1.0, 10.0):
        for gamma in ("scale", "auto"):
            candidates.append(
                Candidate(
                    family="physics_aware_time_frequency",
                    representation="physics_time_frequency",
                    hyperparameter={"C": c_value, "gamma": gamma},
                    build=lambda value=c_value, g=gamma: CalibratedClassifierCV(
                        make_pipeline(
                            StandardScaler(),
                            SVC(
                                C=value,
                                gamma=g,
                                kernel="rbf",
                                class_weight="balanced",
                                probability=False,
                                random_state=random_state,
                            ),
                        ),
                        ensemble=False,
                    ),
                )
            )

    for hidden_layers, alpha, learning_rate in (
        ((128,), 1e-4, 1e-3),
        ((128, 64), 1e-4, 1e-3),
        ((256, 128), 1e-5, 5e-4),
    ):
        candidates.append(
            Candidate(
                family="multiscale_time_series",
                representation="multiscale_unified",
                hyperparameter={
                    "hidden_layer_sizes": hidden_layers,
                    "alpha": alpha,
                    "learning_rate_init": learning_rate,
                    "early_stopping": True,
                },
                build=lambda layers=hidden_layers, reg=alpha, lr=learning_rate: make_pipeline(
                    StandardScaler(),
                    MLPClassifier(
                        hidden_layer_sizes=layers,
                        activation="relu",
                        solver="adam",
                        alpha=reg,
                        batch_size="auto",
                        learning_rate_init=lr,
                        max_iter=800,
                        early_stopping=True,
                        n_iter_no_change=30,
                        validation_fraction=0.2,
                        shuffle=True,
                        random_state=random_state,
                    ),
                ),
            )
        )

    return tuple(candidates)


def _build_views(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    base = _flatten_views(arrays)
    return {
        "adaptive_spectral": _adaptive_spectral_view(base),
        "physics_time_frequency": _physics_aware_view(base),
        "multiscale_unified": _multiscale_unified_view(base),
    }


def build_high_roi_baselines(
    *,
    random_state: int,
    output: Path,
    selected_predictions: Path,
    markdown: Path,
) -> dict[str, Any]:
    arrays, metadata = _load_cache()
    views = _build_views(arrays)
    states = arrays["state"].astype(str)
    targets = component_targets(states)
    groups = bearing_groups(states, arrays["bearing_type"])

    ranked: list[dict[str, Any]] = []
    for candidate in _build_candidates(random_state=random_state):
        features = views[candidate.representation]
        result = evaluate_candidate(features, targets, groups, candidate)
        ranked.append(result)

    ranked.sort(
        key=lambda row: (
            -row["mean_component_auroc"],
            -row["exact_set_accuracy"],
            row["mean_brier_score"],
            row["feature_dimension"],
            row["family"],
            row["representation"],
            json.dumps(row["hyperparameter"], sort_keys=True),
        )
    )
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

    def _strip(row: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in row.items() if k not in {"probabilities", "predictions", "targets"}}

    family_champions: list[dict[str, Any]] = []
    for family in ("adaptive_signal_processing", "physics_aware_time_frequency", "multiscale_time_series"):
        family_rows = [row for row in ranked if row["family"] == family]
        champion = family_rows[0]
        family_champions.append(_strip(champion))

    result = {
        "stage": "hanoi_hust_high_roi_baselines",
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
        "family_champions": family_champions,
        "selected": _strip(selected),
        "candidate_table": [_strip(row) for row in ranked],
        "information_boundary": {
            "source_numeric_files_opened": int(len(states)),
            "compound_numeric_files_opened": 0,
            "transient_numeric_files_opened": 0,
        },
        "provenance": {
            "git_commit": _git("rev-parse", "HEAD"),
            "builder": "src/run_hanoi_hust_high_roi_baselines.py",
            "random_state": int(random_state),
        },
        "view_shapes": {name: list(values.shape) for name, values in views.items()},
        "markdown": markdown.relative_to(ROOT).as_posix(),
    }
    _atomic_json(output, result)

    md_lines = [
        "# HANOI HUST First-Batch High-ROI Baseline Results",
        "",
        f"Date: 2026-07-28",
        "",
        "This result page records the three additional comparator families requested in the补强 plan.",
        "",
        "## Family champions",
        "",
    ]
    for row in family_champions:
        md_lines.extend(
            [
                f"- {row['family']} / {row['representation']} / `C={json.dumps(row['hyperparameter'], ensure_ascii=False)}`",
                f"  - mean_component_auroc: {row['mean_component_auroc']}",
                f"  - exact_set_accuracy: {row['exact_set_accuracy']}",
                f"  - mean_brier_score: {row['mean_brier_score']}",
            ]
        )
    md_lines.extend(
        [
            "",
            "## Selected overall champion",
            "",
            f"- family: {selected['family']}",
            f"- representation: {selected['representation']}",
            f"- hyperparameter: `{json.dumps(selected['hyperparameter'], ensure_ascii=False)}`",
            f"- mean_component_auroc: {selected['mean_component_auroc']}",
            f"- exact_set_accuracy: {selected['exact_set_accuracy']}",
            f"- mean_brier_score: {selected['mean_brier_score']}",
            "",
            "## Interpretation",
            "",
            "- These are the first-batch high-ROI comparators requested in the plan.",
            "- If one of these beats the prior source-side champion, the manuscript should be updated before freezing the package.",
            "- If none of these changes the ranking picture, that negative result still strengthens the SCGP claim.",
        ]
    )
    _atomic_text(markdown, "\n".join(md_lines) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-state", type=int, default=20_260_729)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--selected-predictions", type=Path, default=DEFAULT_SELECTED_PREDICTIONS)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    print(
        json.dumps(
            build_high_roi_baselines(
                random_state=args.random_state,
                output=args.output,
                selected_predictions=args.selected_predictions,
                markdown=args.markdown,
            ),
            indent=2,
            ensure_ascii=False,
            default=_json_default,
        )
    )


if __name__ == "__main__":
    main()
