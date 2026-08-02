"""Run HANOI HUST NOACE baselines across multiple random seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from run_hanoi_hust_source_noace import (
    DEFAULT_CANDIDATE_SELECTED_DIR,
    DEFAULT_REPRESENTATIONS,
    build_noace,
    build_noace_grid,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results" / "analysis" / "hanoi_hust_source_noace_multiseed.json"
DEFAULT_SEEDS = (20260727, 42, 84)


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


def _run_noace_grid_one_seed(
    *,
    seed: int,
    output: Path,
    representations: tuple[str, ...],
    candidate_root: Path,
) -> dict[str, Any]:
    seed_output = output.with_name(f"{output.stem}_seed_{seed}{output.suffix}")
    candidate_dir = candidate_root / f"seed_{seed}"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    result = build_noace_grid(
        random_state=seed,
        output=seed_output,
        representations=representations,
        candidate_selected_dir=candidate_dir,
    )
    selected = result["selected"]
    return {
        "seed": int(seed),
        "seed_output": seed_output.relative_to(ROOT).as_posix(),
        "candidate_count": int(result["candidate_count"]),
        "candidate_selected_dir": candidate_dir.relative_to(ROOT).as_posix(),
        "selected_predictions": result["selected_predictions"],
        "selected_family": selected["family"],
        "selected_representation": selected["representation"],
        "mean_component_auroc": float(selected["mean_component_auroc"]),
        "mean_component_balanced_accuracy": float(selected["mean_component_balanced_accuracy"]),
        "exact_set_accuracy": float(selected["exact_set_accuracy"]),
        "mean_brier_score": float(selected["mean_brier_score"]),
        "hamming_loss": float(selected["hamming_loss"]),
        "status": "completed",
    }


def _run_noace_classical_one_seed(*, seed: int, output: Path) -> dict[str, Any]:
    seed_output = output.with_name(f"{output.stem}_seed_{seed}{output.suffix}")
    seed_selected = seed_output.with_name(f"{seed_output.stem}_selected_predictions.npz")
    result = build_noace(
        random_state=seed,
        output=seed_output,
        selected_predictions=seed_selected,
    )
    selected = result["selected"]
    return {
        "seed": int(seed),
        "seed_output": seed_output.relative_to(ROOT).as_posix(),
        "candidate_count": int(result["candidate_count"]),
        "selected_predictions": seed_selected.relative_to(ROOT).as_posix(),
        "selected_family": selected["family"],
        "selected_representation": selected["representation"],
        "mean_component_auroc": float(selected["mean_component_auroc"]),
        "mean_component_balanced_accuracy": float(selected["mean_component_balanced_accuracy"]),
        "exact_set_accuracy": float(selected["exact_set_accuracy"]),
        "mean_brier_score": float(selected["mean_brier_score"]),
        "hamming_loss": float(selected["hamming_loss"]),
        "status": "completed",
    }


def run_noace_multiseed(
    *,
    seeds: tuple[int, ...],
    mode: str,
    output: Path = DEFAULT_OUTPUT,
    representations: tuple[str, ...] = DEFAULT_REPRESENTATIONS,
    candidate_root: Path = DEFAULT_CANDIDATE_SELECTED_DIR,
) -> dict[str, Any]:
    if not seeds:
        raise ValueError("At least one seed is required")
    if mode == "grid" and not representations:
        raise ValueError("At least one NOACE representation is required in grid mode")
    if mode not in {"classical", "grid"}:
        raise ValueError("mode must be classical or grid")

    runs: list[dict[str, Any]] = []
    for seed in seeds:
        if mode == "grid":
            run = _run_noace_grid_one_seed(
                seed=seed,
                output=output,
                representations=representations,
                candidate_root=candidate_root,
            )
        else:
            run = _run_noace_classical_one_seed(seed=seed, output=output)
        runs.append(run)

    champion = max(
        runs,
        key=lambda row: (
            row["mean_component_auroc"],
            row["exact_set_accuracy"],
            -row["mean_brier_score"],
            row["seed"],
        ),
    )

    summary_fields = (
        "mean_component_auroc",
        "mean_component_balanced_accuracy",
        "exact_set_accuracy",
        "mean_brier_score",
        "hamming_loss",
    )
    aggregates = {
        field: {
            "mean": float(np.mean([row[field] for row in runs])),
            "std": float(np.std([row[field] for row in runs], ddof=0)),
            "min": float(np.min([row[field] for row in runs])),
            "max": float(np.max([row[field] for row in runs])),
        }
        for field in summary_fields
    }

    summary: dict[str, Any] = {
        "seed_count": len(runs),
        "seeds": list(seeds),
        "mode": mode,
        "mode_params": {"representations": list(representations)} if mode == "grid" else {},
        "champion": champion,
        "champion_selection": {
            "metric_order": [
                "mean_component_auroc",
                "exact_set_accuracy",
                "-mean_brier_score",
                "seed",
            ]
        },
        "aggregates": aggregates,
        "runs": runs,
    }

    stage = "hanoi_hust_source_noace_multiseed_classical"
    if mode == "grid":
        stage = "hanoi_hust_source_noace_multiseed_grid"
        representation_summary: dict[str, dict[str, float | int]] = {}
        by_rep: dict[str, list[float]] = {}
        for row in runs:
            by_rep.setdefault(row["selected_representation"], []).append(
                row["mean_component_auroc"]
            )
        for rep, values in sorted(by_rep.items()):
            representation_summary[rep] = {
                "count": len(values),
                "mean_auroc": float(np.mean(values)),
                "std_auroc": float(np.std(values, ddof=0)),
                "wins": int(sum(1 for row in runs if row["selected_representation"] == rep)),
            }
        summary["representation_summary"] = representation_summary

    summary.update(
        {
            "stage": stage,
            "schema_version": 1,
            "status": "completed",
            "cache": {
                "path": "artifacts/hanoi_hust/source_features.npz",
                "sha256": _sha256(ROOT / "artifacts" / "hanoi_hust" / "source_features.npz"),
            },
            "provenance": {
                "builder": "src/run_hanoi_hust_source_noace_multiseed.py",
                "requested_mode": mode,
                "requested_representations": list(representations),
            },
        }
    )

    _atomic_json(output, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="*", default=list(DEFAULT_SEEDS))
    parser.add_argument("--mode", choices=("classical", "grid"), default="grid")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--representations",
        type=str,
        nargs="*",
        default=list(DEFAULT_REPRESENTATIONS),
    )
    parser.add_argument(
        "--candidate-root",
        type=Path,
        default=DEFAULT_CANDIDATE_SELECTED_DIR / "noace_multiseed",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run_noace_multiseed(
                seeds=tuple(args.seeds),
                mode=args.mode,
                output=args.output,
                representations=tuple(args.representations),
                candidate_root=args.candidate_root,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
