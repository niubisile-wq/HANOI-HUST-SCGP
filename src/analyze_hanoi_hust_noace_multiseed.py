"""Summarize HANOI HUST NOACE multi-seed runs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results" / "analysis" / "hanoi_hust_source_noace_multiseed.json"
OUTPUT_JSON = ROOT / "results" / "analysis" / "hanoi_hust_noace_multiseed_summary.json"
OUTPUT_MD = ROOT / "results" / "analysis" / "hanoi_hust_noace_multiseed_summary.md"


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


def build_noace_multiseed_report(
    *,
    input_path: Path = DEFAULT_INPUT,
) -> dict[str, Any]:
    input_path = input_path.resolve()
    if not input_path.is_absolute():
        input_path = (Path.cwd() / input_path).resolve()
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if payload.get("status") != "completed":
        raise RuntimeError("NOACE multi-seed result is not completed")
    if not payload.get("stage", "").startswith("hanoi_hust_source_noace_multiseed_"):
        raise RuntimeError("Expected NOACE multi-seed stage")

    runs = payload["runs"]
    if not runs:
        raise RuntimeError("NOACE multi-seed contains no runs")
    mode = payload.get("mode", "unknown")
    seeds = [int(value) for value in payload.get("seeds", [])]

    by_rep: dict[str, list[dict[str, float]]] = {}
    for run in runs:
        rep = run.get("selected_representation", "unknown")
        by_rep.setdefault(rep, []).append(
            {
                "mean_component_auroc": float(run["mean_component_auroc"]),
                "exact_set_accuracy": float(run["exact_set_accuracy"]),
                "mean_brier_score": float(run["mean_brier_score"]),
                "seed": int(run["seed"]),
            }
        )

    representation_summary = {}
    for rep in sorted(by_rep):
        rep_rows = by_rep[rep]
        aurocs = [row["mean_component_auroc"] for row in rep_rows]
        representation_summary[rep] = {
            "wins": int(sum(1 for row in runs if row.get("selected_representation") == rep)),
            "count": len(rep_rows),
            "mean_auroc": float(np.mean(aurocs)),
            "std_auroc": float(np.std(aurocs, ddof=0)),
            "min_auroc": float(np.min(aurocs)),
            "max_auroc": float(np.max(aurocs)),
        }

    seed_order = sorted((int(v) for v in payload.get("seeds", [])))
    run_order = sorted(
        runs,
        key=lambda row: (
            -row["mean_component_auroc"],
            -row["exact_set_accuracy"],
            row["mean_brier_score"],
            row["seed"],
        ),
    )

    champion = payload["champion"]
    report = {
        "stage": "hanoi_hust_noace_multiseed_summary",
        "schema_version": 1,
        "status": "completed",
        "input": str(input_path.relative_to(ROOT).as_posix())
        if input_path.is_relative_to(ROOT)
        else input_path.as_posix(),
        "mode": mode,
        "seed_count": int(len(runs)),
        "seeds": seed_order,
        "aggregates": payload.get("aggregates", {}),
        "champion": champion,
        "winner_by_seed": [
            {"seed": int(row["seed"]), "selected_representation": row.get("selected_representation")}
            for row in run_order
        ],
        "representation_summary": representation_summary,
        "top_runs": run_order,
    }

    _atomic_json(OUTPUT_JSON, report)

    markdown = [
        "# HANOI HUST NOACE Multi-Seed Summary",
        "",
        f"- mode: `{mode}`",
        f"- seed count: `{report['seed_count']}`",
        f"- champion: `{champion['selected_family']} / {champion['selected_representation']}`",
        "",
        "## Aggregates (selected runs)",
        "",
        "| Metric | Mean | Std | Min | Max |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric in (
        "mean_component_auroc",
        "mean_component_balanced_accuracy",
        "exact_set_accuracy",
        "mean_brier_score",
        "hamming_loss",
    ):
        stats = report["aggregates"][metric]
        markdown.append(
            "| {metric} | {mean:.6f} | {std:.6f} | {min:.6f} | {max:.6f} |".format(
                metric=metric,
                **stats,
            )
        )

    markdown.extend(
        [
            "",
            "## Representation summary",
            "",
            "| Representation | Wins | Mean AUROC | Std AUROC | Min | Max |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for rep, summary in sorted(
        representation_summary.items(), key=lambda item: item[1]["wins"], reverse=True
    ):
        markdown.append(
            "| {rep} | {wins} | {mean_auroc:.6f} | {std_auroc:.6f} | {min_auroc:.6f} | {max_auroc:.6f} |".format(
                rep=rep, **summary
            )
        )

    markdown.extend(
        [
            "",
            "## Run ranking",
            "",
            "| Rank | Seed | Representation | AUROC | Exact-set | Brier |",
            "|---:|---:|---|---:|---:|---:|",
        ]
    )
    for index, row in enumerate(run_order, start=1):
        markdown.append(
            "| {rank} | {seed} | {selected_representation} | {mean_component_auroc:.6f} | {exact_set_accuracy:.6f} | {mean_brier_score:.6f} |".format(
                rank=index,
                seed=row["seed"],
                selected_representation=row["selected_representation"],
                mean_component_auroc=row["mean_component_auroc"],
                exact_set_accuracy=row["exact_set_accuracy"],
                mean_brier_score=row["mean_brier_score"],
            )
        )
    _atomic_text(OUTPUT_MD, "\n".join(markdown))
    return report


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args()
    print(
        json.dumps(
            build_noace_multiseed_report(input_path=args.input),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
