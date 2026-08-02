"""Rebuild and audit the G2 main table from frozen JSON/NPZ artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "g2" / "protocol_cell_metrics.json"
PAIRED = ROOT / "results" / "g2" / "primary_paired_split.json"
TABLE = ROOT / "results" / "g2" / "g2_main_table.csv"
AUDIT = ROOT / "results" / "g2" / "g2_main_table_audit.json"

CELLS = (
    ("record_grouped", "fixed_prespecified"),
    ("record_grouped", "nested_selection"),
    ("bearing_grouped", "fixed_prespecified"),
    ("bearing_grouped", "nested_selection"),
    ("window_random", "fixed_prespecified"),
    ("window_random", "nested_selection"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix(".writing")
    staging.write_text(text, encoding="utf-8")
    os.replace(staging, path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_text(
        path,
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
    )


def build_audit() -> dict[str, Any]:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    paired = json.loads(PAIRED.read_text(encoding="utf-8"))
    failures: list[str] = []
    rows: list[dict[str, Any]] = []
    for hierarchy, selection in CELLS:
        key = f"{hierarchy}__{selection}"
        cell = summary["cells"].get(key)
        if cell is None:
            failures.append(f"missing summary cell: {key}")
            continue
        if cell["split_count"] != 100:
            failures.append(f"split_count != 100: {key}")
        prediction = ROOT / cell["prediction_artifact"]["path"]
        if not prediction.is_file():
            failures.append(f"missing prediction artifact: {prediction}")
        else:
            actual = _sha256(prediction)
            if actual != cell["prediction_artifact"]["sha256"]:
                failures.append(f"prediction hash mismatch: {key}")
        rows.append({
            "hierarchy": hierarchy,
            "selection": selection,
            "inference_allowed": cell["inference_allowed"],
            "macro_auroc_mean": cell["metrics"]["mean_component_auroc"]["mean"],
            "macro_auroc_split_ci_low": cell["metrics"]["mean_component_auroc"]["split_sensitivity_interval"][0],
            "macro_auroc_split_ci_high": cell["metrics"]["mean_component_auroc"]["split_sensitivity_interval"][1],
            "exact_set_mean": cell["metrics"]["exact_set_accuracy"]["mean"],
            "selection_entropy": cell["normalized_selection_entropy"],
            "outer_regret_mean": cell["outer_test_regret"]["mean_outer_test_regret_macro_auroc"],
        })
    if paired.get("status") != "completed":
        failures.append("primary paired split is not completed")
    if paired.get("pre_specified_test_bearings") != ["N4", "N5", "I4", "O4", "B5"]:
        failures.append("pre-specified paired test-bearing list changed")

    fields = list(rows[0]) if rows else []
    table_lines = []
    if fields:
        table_lines.append(",".join(fields))
        for row in rows:
            table_lines.append(",".join(str(row[field]) for field in fields))
    _atomic_text(TABLE, "\n".join(table_lines) + "\n")
    result = {
        "stage": "hanoi_hust_as_g2_main_table_audit",
        "schema_version": 1,
        "status": "passed" if not failures else "failed",
        "summary_sha256": _sha256(SUMMARY),
        "paired_split_sha256": _sha256(PAIRED),
        "table_path": TABLE.relative_to(ROOT).as_posix(),
        "table_sha256": _sha256(TABLE),
        "cell_count": len(rows),
        "checks": {
            "all_six_cells_present": len(rows) == 6,
            "all_cells_have_100_splits": not any("split_count" in failure for failure in failures),
            "all_prediction_hashes_match": not any("hash mismatch" in failure for failure in failures),
            "paired_split_locked": not any("paired" in failure or "pre-specified" in failure for failure in failures),
            "table_rebuilt_from_json": True,
        },
        "failures": failures,
    }
    _atomic_json(AUDIT, result)
    return result


def main() -> None:
    result = build_audit()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
