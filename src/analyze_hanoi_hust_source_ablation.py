"""Summarize the frozen Hanoi HUST source-baseline ablation table."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASELINE_RESULT = (
    ROOT / "results" / "development" / "hanoi_hust_source_baselines.json"
)
OUTPUT = ROOT / "results" / "analysis" / "hanoi_hust_source_ablation.json"
OUTPUT_MD = ROOT / "results" / "analysis" / "hanoi_hust_source_ablation.md"


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


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix(".writing")
    staging.write_text(text, encoding="utf-8")
    os.replace(staging, path)


def _best_by(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    return max(
        rows,
        key=lambda row: (
            row["mean_component_auroc"],
            row["exact_set_accuracy"],
            -row["mean_brier_score"],
            -row["feature_dimension"],
            row["family"],
            row["representation"],
            json.dumps(row["hyperparameter"], sort_keys=True),
        ),
    )


def build_report() -> dict[str, Any]:
    """Return an ablation report from the frozen source baseline table."""
    payload = json.loads(BASELINE_RESULT.read_text(encoding="utf-8"))
    if payload.get("status") != "completed":
        raise RuntimeError("HANOI HUST source baselines are not complete")
    rows = payload["candidate_table"]
    if len(rows) != payload["candidate_count"]:
        raise RuntimeError("HANOI HUST candidate table length changed")
    by_view: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_view[row["representation"]].append(row)
        by_family[row["family"]].append(row)
    view_summary = {
        view: _best_by(items, view)
        for view, items in sorted(by_view.items())
    }
    family_summary = {
        family: _best_by(items, family)
        for family, items in sorted(by_family.items())
    }
    selected = payload["selected"]
    dominant_view = max(
        view_summary.items(),
        key=lambda item: (
            item[1]["mean_component_auroc"],
            item[1]["exact_set_accuracy"],
            -item[1]["mean_brier_score"],
            -item[1]["feature_dimension"],
            item[0],
        ),
    )[0]
    dominant_family = max(
        family_summary.items(),
        key=lambda item: (
            item[1]["mean_component_auroc"],
            item[1]["exact_set_accuracy"],
            -item[1]["mean_brier_score"],
            -item[1]["feature_dimension"],
            item[0],
        ),
    )[0]
    report = {
        "stage": "hanoi_hust_source_ablation",
        "schema_version": 1,
        "status": "completed",
        "baseline_result": {
            "path": BASELINE_RESULT.relative_to(ROOT).as_posix(),
            "sha256": _sha256(BASELINE_RESULT),
        },
        "selected": selected,
        "dominant_view": dominant_view,
        "dominant_family": dominant_family,
        "view_summary": view_summary,
        "family_summary": family_summary,
        "view_ranking": sorted(
            view_summary.values(),
            key=lambda row: (
                -row["mean_component_auroc"],
                -row["exact_set_accuracy"],
                row["mean_brier_score"],
                row["feature_dimension"],
                row["family"],
            ),
        ),
        "family_ranking": sorted(
            family_summary.values(),
            key=lambda row: (
                -row["mean_component_auroc"],
                -row["exact_set_accuracy"],
                row["mean_brier_score"],
                row["feature_dimension"],
                row["family"],
            ),
        ),
        "key_findings": {
            "best_view_mean_auroc": float(view_summary[dominant_view]["mean_component_auroc"]),
            "best_family_mean_auroc": float(family_summary[dominant_family]["mean_component_auroc"]),
            "view_gap_vs_fixed": float(
                view_summary[dominant_view]["mean_component_auroc"]
                - view_summary["fixed_log_power"]["mean_component_auroc"]
            ),
            "view_gap_vs_statistics": float(
                view_summary[dominant_view]["mean_component_auroc"]
                - view_summary["statistics"]["mean_component_auroc"]
            ),
            "exact_set_gain_vs_fixed": float(
                view_summary[dominant_view]["exact_set_accuracy"]
                - view_summary["fixed_log_power"]["exact_set_accuracy"]
            ),
            "selected_gap_vs_best_view": float(
                selected["mean_component_auroc"]
                - view_summary[dominant_view]["mean_component_auroc"]
            ),
        },
        "provenance": {
            "git_commit": _git("rev-parse", "HEAD"),
            "builder": "src/analyze_hanoi_hust_source_ablation.py",
        },
    }
    _atomic_json(OUTPUT, report)
    markdown = [
        "# Hanoi HUST Source Ablation",
        "",
        f"- Selected model: `{selected['family']}` / `{selected['representation']}` / `{selected['hyperparameter']}`",
        f"- Best view: `{dominant_view}`",
        f"- Best family: `{dominant_family}`",
        "",
        "## View Ranking",
        "",
        "| View | Family | Hyperparameter | Mean AUROC | Exact-set | Mean CBA |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in report["view_ranking"]:
        markdown.append(
            "| {representation} | {family} | {hyperparameter} | {mean_component_auroc:.6f} | {exact_set_accuracy:.6f} | {mean_component_balanced_accuracy:.6f} |".format(
                **row
            )
        )
    markdown.extend(
        [
            "",
            "## Family Ranking",
            "",
            "| Family | View | Hyperparameter | Mean AUROC | Exact-set | Mean CBA |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for row in report["family_ranking"]:
        markdown.append(
            "| {family} | {representation} | {hyperparameter} | {mean_component_auroc:.6f} | {exact_set_accuracy:.6f} | {mean_component_balanced_accuracy:.6f} |".format(
                **row
            )
        )
    markdown.extend(
        [
            "",
            "## Key Findings",
            "",
            f"- Best view gain over `fixed_log_power`: {report['key_findings']['view_gap_vs_fixed']:.6f}",
            f"- Best view gain over `statistics`: {report['key_findings']['view_gap_vs_statistics']:.6f}",
            f"- Exact-set gain over `fixed_log_power`: {report['key_findings']['exact_set_gain_vs_fixed']:.6f}",
            f"- Gap between selected model and best-view score: {report['key_findings']['selected_gap_vs_best_view']:.6f}",
            "",
        ]
    )
    _atomic_text(OUTPUT_MD, "\n".join(markdown))
    return report


def main() -> None:
    print(json.dumps(build_report(), indent=2, ensure_ascii=False, default=_json_default))


if __name__ == "__main__":
    main()
