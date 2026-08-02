"""Mechanism-validation synthesis for the frozen HANOI HUST protocol.

This report collects the explicit mechanism-oriented evidence:

- identity probes from the refutation pack;
- full-factorial interaction residuals on the frozen source grid;
- deletion/occlusion-style leave-one-component-out evidence;
- worst-group behavior and label-shuffle contrast.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE_BASELINES = ROOT / "results" / "development" / "hanoi_hust_source_baselines.json"
REFUTATION_PACK = ROOT / "results" / "analysis" / "hanoi_hust_refutation_pack.json"
STATS_HARDENING = ROOT / "results" / "analysis" / "hanoi_hust_statistics_hardening.json"
OUTPUT_JSON = ROOT / "results" / "analysis" / "hanoi_hust_mechanism_validation.json"
OUTPUT_MD = ROOT / "results" / "analysis" / "hanoi_hust_mechanism_validation.md"


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
    raise TypeError(f"Unsupported value type: {type(value).__name__}")


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


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _interaction_residuals(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    families = sorted({row["family"] for row in rows})
    views = sorted({row["representation"] for row in rows})
    matrix = np.empty((len(families), len(views)), dtype=float)
    for i, family in enumerate(families):
        for j, view in enumerate(views):
            match = next(row for row in rows if row["family"] == family and row["representation"] == view)
            matrix[i, j] = float(match[metric])
    grand = float(matrix.mean())
    family_mean = matrix.mean(axis=1, keepdims=True)
    view_mean = matrix.mean(axis=0, keepdims=True)
    residual = matrix - family_mean - view_mean + grand
    strongest = np.unravel_index(np.argmax(residual), residual.shape)
    weakest = np.unravel_index(np.argmin(residual), residual.shape)
    return {
        "families": families,
        "views": views,
        "grand_mean": grand,
        "matrix": matrix.tolist(),
        "residual": residual.tolist(),
        "strongest_positive": {
            "family": families[int(strongest[0])],
            "view": views[int(strongest[1])],
            "residual": float(residual[strongest]),
        },
        "strongest_negative": {
            "family": families[int(weakest[0])],
            "view": views[int(weakest[1])],
            "residual": float(residual[weakest]),
        },
    }


def build_report() -> dict[str, Any]:
    baselines = _load_json(SOURCE_BASELINES)
    refutation = _load_json(REFUTATION_PACK)
    stats = _load_json(STATS_HARDENING)

    rows = baselines["candidate_table"]
    auroc_residual = _interaction_residuals(rows, "mean_component_auroc")
    exact_residual = _interaction_residuals(rows, "exact_set_accuracy")

    report = {
        "stage": "hanoi_hust_mechanism_validation",
        "schema_version": 1,
        "status": "completed",
        "source_baselines": {
            "path": SOURCE_BASELINES.relative_to(ROOT).as_posix(),
            "sha256": _sha256(SOURCE_BASELINES),
            "candidate_count": baselines["candidate_count"],
        },
        "refutation_pack": {
            "path": REFUTATION_PACK.relative_to(ROOT).as_posix(),
            "sha256": _sha256(REFUTATION_PACK),
            "identity_probes": refutation["identity_probes"],
            "label_shuffle": {
                "best_seed": max(refutation["label_shuffle"]["runs"], key=lambda row: row["mean_component_auroc"]),
                "aggregate_mean_auroc": refutation["label_shuffle"]["aggregates"]["mean_component_auroc"]["mean"],
            },
            "leave_one_bearing_out": {
                "worst_groups": refutation["leave_one_bearing_out"]["groups"],
            },
        },
        "interaction_residuals": {
            "mean_component_auroc": auroc_residual,
            "exact_set_accuracy": exact_residual,
        },
        "occlusion_and_deletion": {
            "leave_one_component_out": stats["leave_one_component_out"],
        },
        "provenance": {
            "git_commit": _git("rev-parse", "HEAD"),
            "builder": "src/analyze_hanoi_hust_mechanism_validation.py",
        },
    }
    _atomic_json(OUTPUT_JSON, report)

    lines = [
        "# HANOI HUST mechanism validation",
        "",
        "## Identity probes",
        "",
        f"- bearing_type_from_features_by_load_split accuracy: {refutation['identity_probes']['bearing_type_from_features_by_load_split']['accuracy']:.6f}",
        f"- load_from_features_by_bearing_split accuracy: {refutation['identity_probes']['load_from_features_by_bearing_split']['accuracy']:.6f}",
        "",
        "## Interaction residuals",
        "",
        "| Metric | Strongest positive | Residual | Strongest negative | Residual |",
        "|---|---|---:|---|---:|",
        "| mean_component_auroc | {pos_f} / {pos_v} | {pos_r:.6f} | {neg_f} / {neg_v} | {neg_r:.6f} |".format(
            pos_f=auroc_residual["strongest_positive"]["family"],
            pos_v=auroc_residual["strongest_positive"]["view"],
            pos_r=auroc_residual["strongest_positive"]["residual"],
            neg_f=auroc_residual["strongest_negative"]["family"],
            neg_v=auroc_residual["strongest_negative"]["view"],
            neg_r=auroc_residual["strongest_negative"]["residual"],
        ),
        "| exact_set_accuracy | {pos_f} / {pos_v} | {pos_r:.6f} | {neg_f} / {neg_v} | {neg_r:.6f} |".format(
            pos_f=exact_residual["strongest_positive"]["family"],
            pos_v=exact_residual["strongest_positive"]["view"],
            pos_r=exact_residual["strongest_positive"]["residual"],
            neg_f=exact_residual["strongest_negative"]["family"],
            neg_v=exact_residual["strongest_negative"]["view"],
            neg_r=exact_residual["strongest_negative"]["residual"],
        ),
        "",
        "## Deletion / occlusion-style checks",
        "",
        "| Dropped component | Kept components | Exact-set | Delta vs full (pp) |",
        "|---|---|---:|---:|",
    ]
    for dropped, values in stats["leave_one_component_out"].items():
        lines.append(
            "| {dropped} | {kept} | {exact_set_accuracy:.6f} | {delta_vs_full_pp:.3f} |".format(
                dropped=dropped,
                kept=", ".join(values["keep_components"]),
                exact_set_accuracy=values["exact_set_accuracy"],
                delta_vs_full_pp=values["delta_vs_full_pp"],
            )
        )
    _atomic_text(OUTPUT_MD, "\n".join(lines))
    return report


def main() -> None:
    print(json.dumps(build_report(), indent=2, ensure_ascii=False, default=_json_default))


if __name__ == "__main__":
    main()
