"""Run frozen HANOI HUST source-only baselines across multiple seeds."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from run_hanoi_hust_source_baselines import build_result


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results" / "development" / "hanoi_hust_source_multiseed.json"
DEFAULT_SEEDS = (42, 7, 21, 84, 168)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix(".writing")
    staging.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(staging, path)


def run_multiseed(
    *,
    seeds: tuple[int, ...],
    output: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    if not seeds:
        raise ValueError("At least one seed is required")
    runs = []
    for seed in seeds:
        seed_output = output.with_name(f"{output.stem}_seed_{seed}{output.suffix}")
        seed_selected = seed_output.with_name(
            f"{seed_output.stem}_selected_predictions.npz"
        )
        result = build_result(
            random_state=seed,
            output=seed_output,
            selected_predictions=seed_selected,
        )
        selected = result["selected"]
        runs.append(
            {
                "seed": seed,
                "output": seed_output.relative_to(ROOT).as_posix(),
                "selected_predictions": seed_selected.relative_to(ROOT).as_posix(),
                "selected_family": selected["family"],
                "selected_representation": selected["representation"],
                "selected_hyperparameter": selected["hyperparameter"],
                "mean_component_auroc": selected["mean_component_auroc"],
                "mean_component_balanced_accuracy": selected[
                    "mean_component_balanced_accuracy"
                ],
                "exact_set_accuracy": selected["exact_set_accuracy"],
                "mean_brier_score": selected["mean_brier_score"],
                "hamming_loss": selected["hamming_loss"],
                "status": "completed",
            }
        )

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
    summary = {
        "stage": "hanoi_hust_source_multiseed",
        "schema_version": 1,
        "status": "completed",
        "seed_count": len(runs),
        "seeds": list(seeds),
        "champion": champion,
        "aggregates": aggregates,
        "runs": runs,
        "provenance": {
            "builder": "src/run_hanoi_hust_source_multiseed.py",
        },
    }
    _atomic_json(output, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", type=int, nargs="*", default=list(DEFAULT_SEEDS))
    args = parser.parse_args()
    print(
        json.dumps(
            run_multiseed(seeds=tuple(args.seeds), output=args.output),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
