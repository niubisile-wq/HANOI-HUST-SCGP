"""Run HANOI HUST NOACE-Deep baselines across multiple random seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from run_hanoi_hust_source_noace_deep import (
    DEFAULT_OUTPUT,
    build_noace_deep,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MULTISEED_OUTPUT = ROOT / "results" / "analysis" / "hanoi_hust_noace_deep_multiseed.json"
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


def _run_one_seed(*, seed: int, output: Path) -> dict[str, Any]:
    seed_output = output.with_name(f"{output.stem}_seed_{seed}{output.suffix}")
    seed_selected = seed_output.with_name(f"{seed_output.stem}_selected_predictions.npz")
    result = build_noace_deep(
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


def run_noace_deep_multiseed(
    *,
    seeds: tuple[int, ...],
    output: Path = DEFAULT_MULTISEED_OUTPUT,
) -> dict[str, Any]:
    if not seeds:
        raise ValueError("At least one seed is required")

    runs = [_run_one_seed(seed=seed, output=output) for seed in seeds]
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
        "stage": "hanoi_hust_noace_deep_multiseed",
        "schema_version": 1,
        "status": "completed",
        "cache": {
            "path": "artifacts/hanoi_hust/source_features.npz",
            "sha256": _sha256(ROOT / "artifacts" / "hanoi_hust" / "source_features.npz"),
        },
        "provenance": {
            "builder": "src/run_hanoi_hust_source_noace_deep_multiseed.py",
        },
    }
    _atomic_json(output, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="*", default=list(DEFAULT_SEEDS))
    parser.add_argument("--output", type=Path, default=DEFAULT_MULTISEED_OUTPUT)
    args = parser.parse_args()
    print(
        json.dumps(
            run_noace_deep_multiseed(seeds=tuple(args.seeds), output=args.output),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
