"""Summarize HANOI HUST NOACE grid runs."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results" / "analysis" / "hanoi_hust_noace_grid.json"
OUTPUT_JSON = ROOT / "results" / "analysis" / "hanoi_hust_noace_grid_ablation.json"
OUTPUT_MD = ROOT / "results" / "analysis" / "hanoi_hust_noace_grid_ablation.md"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix(".writing")
    staging.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(staging, path)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix(".writing")
    staging.write_text(text, encoding="utf-8")
    os.replace(staging, path)


def _best_by(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        rows,
        key=lambda row: (
            row["mean_component_auroc"],
            row["exact_set_accuracy"],
            -row["mean_brier_score"],
            -row["mean_component_balanced_accuracy"],
            row["feature_dimension"],
            row["representation"],
            str(row["hyperparameter"]),
        ),
    )


def build_noace_grid_report(*, input_path: Path = DEFAULT_INPUT) -> dict[str, Any]:
    input_path = input_path.resolve()
    if not input_path.is_absolute():
        input_path = (Path.cwd() / input_path).resolve()
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if payload.get("status") != "completed":
        raise RuntimeError("NOACE grid result is not completed")
    if payload.get("stage") != "hanoi_hust_noace_grid":
        raise RuntimeError("Expected hanoi_hust_noace_grid stage")

    rows = payload["candidate_table"]
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("NOACE grid has no candidate rows")
    if len(rows) != payload.get("candidate_count"):
        raise RuntimeError("NOACE grid candidate count does not match")

    by_view: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_view[row["representation"]].append(row)

    view_summary = {view: _best_by(items) for view, items in sorted(by_view.items())}
    selected = payload["selected"]
    dominant_view = max(
        view_summary.items(),
        key=lambda item: (
            item[1]["mean_component_auroc"],
            item[1]["exact_set_accuracy"],
            -item[1]["mean_brier_score"],
            -item[1]["mean_component_balanced_accuracy"],
            item[1]["feature_dimension"],
        ),
    )[0]

    # rank candidate rows by score, tie breaking by Brier and model size
    candidate_ranking = sorted(
        rows,
        key=lambda row: (
            -row["mean_component_auroc"],
            -row["exact_set_accuracy"],
            row["mean_brier_score"],
            -row["mean_component_balanced_accuracy"],
            -row["feature_dimension"],
            row["representation"],
        ),
    )

    report = {
        "stage": "hanoi_hust_noace_grid_ablation",
        "schema_version": 1,
        "status": "completed",
        "input": str(input_path.relative_to(ROOT).as_posix())
        if input_path.is_relative_to(ROOT)
        else input_path.as_posix(),
        "selected": selected,
        "dominant_view": dominant_view,
        "view_summary": {
            key: {
                "mean_component_auroc": float(value["mean_component_auroc"]),
                "exact_set_accuracy": float(value["exact_set_accuracy"]),
                "mean_brier_score": float(value["mean_brier_score"]),
                "mean_component_balanced_accuracy": float(
                    value["mean_component_balanced_accuracy"]
                ),
                "feature_dimension": int(value["feature_dimension"]),
                "family": value["family"],
            }
            for key, value in view_summary.items()
        },
        "candidate_ranking": [
            {
                "family": row["family"],
                "representation": row["representation"],
                "mean_component_auroc": float(row["mean_component_auroc"]),
                "exact_set_accuracy": float(row["exact_set_accuracy"]),
                "mean_brier_score": float(row["mean_brier_score"]),
                "mean_component_balanced_accuracy": float(
                    row["mean_component_balanced_accuracy"]
                ),
                "feature_dimension": int(row["feature_dimension"]),
            }
            for row in candidate_ranking
        ],
        "candidate_count": int(len(rows)),
        "performance_gap_vs_dominant_view": {
            "dominant_view": dominant_view,
            "selected_minus_dominant": float(
                selected["mean_component_auroc"] - view_summary[dominant_view]["mean_component_auroc"]
            ),
            "selected_minus_fixed_log_power": float(
                selected["mean_component_auroc"] - view_summary["fixed_log_power"]["mean_component_auroc"]
            ) if "fixed_log_power" in view_summary else None,
            "selected_minus_statistics": float(
                selected["mean_component_auroc"] - view_summary["statistics"]["mean_component_auroc"]
            ) if "statistics" in view_summary else None,
        },
    }

    _atomic_json(OUTPUT_JSON, report)

    table_rows = [
        "| View | Family | AUROC | Exact-set | Brier | Mean CBA |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in report["candidate_ranking"]:
        table_rows.append(
            "| {representation} | {family} | {mean_component_auroc:.6f} | "
            "{exact_set_accuracy:.6f} | {mean_brier_score:.6f} | "
            "{mean_component_balanced_accuracy:.6f} |".format(**row)
        )

    markdown = [
        "# HANOI HUST NOACE Grid Ablation",
        "",
        f"- input: `{report['input']}`",
        f"- selected: `{selected['family']}` / `{selected['representation']}`",
        f"- dominant view: `{dominant_view}`",
        f"- selected AUROC: `{selected['mean_component_auroc']:.6f}`",
        f"- selected exact-set: `{selected['exact_set_accuracy']:.6f}`",
        f"- selected Brier: `{selected['mean_brier_score']:.6f}`",
        "",
        "## Candidate Ranking",
        "",
        *table_rows,
        "",
        "## Gap to Dominant View",
        "",
        f"- AUROC gap (selected - dominant view): `{report['performance_gap_vs_dominant_view']['selected_minus_dominant']:.6f}`",
    ]

    if report["performance_gap_vs_dominant_view"]["selected_minus_fixed_log_power"] is not None:
        markdown.append(
            f"- AUROC gap (selected - fixed_log_power): `{report['performance_gap_vs_dominant_view']['selected_minus_fixed_log_power']:.6f}`"
        )
    if report["performance_gap_vs_dominant_view"]["selected_minus_statistics"] is not None:
        markdown.append(
            f"- AUROC gap (selected - statistics): `{report['performance_gap_vs_dominant_view']['selected_minus_statistics']:.6f}`"
        )
    _atomic_text(OUTPUT_MD, "\n".join(markdown))
    return report


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args()
    print(json.dumps(build_noace_grid_report(input_path=args.input), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
