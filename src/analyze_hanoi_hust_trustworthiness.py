"""Analyze calibration and worst-group behavior for HANOI HUST."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = (
    ROOT / "results" / "development" / "hanoi_hust_source_selected_predictions.npz"
)
OUTPUT_JSON = ROOT / "results" / "analysis" / "hanoi_hust_trustworthiness.json"
OUTPUT_MD = ROOT / "results" / "analysis" / "hanoi_hust_trustworthiness.md"


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


def _ece(probabilities: np.ndarray, labels: np.ndarray, *, bins: int = 10) -> float:
    probs = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    truth = np.asarray(labels, dtype=np.int8).reshape(-1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    total = len(probs)
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        if upper < 1.0:
            mask = (probs >= lower) & (probs < upper)
        else:
            mask = (probs >= lower) & (probs <= upper)
        if not np.any(mask):
            continue
        confidence = float(probs[mask].mean())
        accuracy = float(truth[mask].mean())
        ece += (mask.sum() / total) * abs(confidence - accuracy)
    return float(ece)


def _group_summary(
    groups: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    group_values = np.asarray(groups, dtype=str)
    row_correct = np.all(predictions == labels, axis=1)
    summaries: dict[str, Any] = {}
    for group in sorted(set(group_values.tolist())):
        mask = group_values == group
        per_group_probs = probabilities[mask]
        per_group_labels = labels[mask]
        per_group_preds = predictions[mask]
        summaries[group] = {
            "record_count": int(mask.sum()),
            "exact_set_accuracy": float(np.all(per_group_preds == per_group_labels, axis=1).mean()),
            "hamming_loss": float(np.mean(per_group_preds != per_group_labels)),
            "component_accuracy": {
                "inner": float(np.mean(per_group_preds[:, 0] == per_group_labels[:, 0])),
                "outer": float(np.mean(per_group_preds[:, 1] == per_group_labels[:, 1])),
                "ball": float(np.mean(per_group_preds[:, 2] == per_group_labels[:, 2])),
            },
            "component_ece": {
                "inner": _ece(per_group_probs[:, 0], per_group_labels[:, 0]),
                "outer": _ece(per_group_probs[:, 1], per_group_labels[:, 1]),
                "ball": _ece(per_group_probs[:, 2], per_group_labels[:, 2]),
            },
            "exact_set_error_rate": float(1.0 - row_correct[mask].mean()),
        }
    return summaries


def build_report() -> dict[str, Any]:
    with np.load(PREDICTIONS) as payload:
        probabilities = np.asarray(payload["probabilities"], dtype=np.float64)
        predictions = np.asarray(payload["predictions"], dtype=np.int8)
        targets = np.asarray(payload["targets"], dtype=np.int8)
        groups = np.asarray(payload["groups"], dtype=str)
        bearing_type = np.asarray(payload["bearing_type"], dtype=np.int16)
        load_w = np.asarray(payload["load_w"], dtype=np.int16)

    if probabilities.shape != predictions.shape or predictions.shape != targets.shape:
        raise RuntimeError("HANOI HUST prediction tensors changed")

    overall = {
        "record_count": int(len(predictions)),
        "exact_set_accuracy": float(np.all(predictions == targets, axis=1).mean()),
        "hamming_loss": float(np.mean(predictions != targets)),
        "component_accuracy": {
            "inner": float(np.mean(predictions[:, 0] == targets[:, 0])),
            "outer": float(np.mean(predictions[:, 1] == targets[:, 1])),
            "ball": float(np.mean(predictions[:, 2] == targets[:, 2])),
        },
        "component_ece": {
            "inner": _ece(probabilities[:, 0], targets[:, 0]),
            "outer": _ece(probabilities[:, 1], targets[:, 1]),
            "ball": _ece(probabilities[:, 2], targets[:, 2]),
        },
    }
    groups_summary = _group_summary(groups, predictions, targets, probabilities)
    worst_group = min(
        groups_summary.items(),
        key=lambda item: (item[1]["exact_set_accuracy"], item[1]["hamming_loss"], item[0]),
    )
    worst_group_by_accuracy = min(
        groups_summary.items(),
        key=lambda item: (item[1]["component_accuracy"]["inner"], item[0]),
    )
    report = {
        "stage": "hanoi_hust_trustworthiness",
        "schema_version": 1,
        "status": "completed",
        "artifact_path": PREDICTIONS.relative_to(ROOT).as_posix(),
        "overall": overall,
        "groups": groups_summary,
        "worst_group_exact_set": {
            "group": worst_group[0],
            "summary": worst_group[1],
        },
        "worst_group_inner_accuracy": {
            "group": worst_group_by_accuracy[0],
            "summary": worst_group_by_accuracy[1],
        },
        "bearing_types": sorted({int(x) for x in bearing_type.tolist()}),
        "loads_w": sorted({int(x) for x in load_w.tolist()}),
        "provenance": {
            "builder": "src/analyze_hanoi_hust_trustworthiness.py",
        },
    }
    _atomic_json(OUTPUT_JSON, report)

    top_groups = sorted(
        groups_summary.items(),
        key=lambda item: (item[1]["exact_set_accuracy"], item[1]["hamming_loss"], item[0]),
    )[:5]
    markdown = [
        "# HANOI HUST Trustworthiness Analysis",
        "",
        f"- artifact_path: `{report['artifact_path']}`",
        f"- record_count: `{overall['record_count']}`",
        f"- overall_exact_set_accuracy: `{overall['exact_set_accuracy']:.6f}`",
        f"- overall_hamming_loss: `{overall['hamming_loss']:.6f}`",
        "",
        "## Overall Calibration",
        "",
        "| Component | ECE |",
        "|---|---:|",
        f"| inner | {overall['component_ece']['inner']:.6f} |",
        f"| outer | {overall['component_ece']['outer']:.6f} |",
        f"| ball | {overall['component_ece']['ball']:.6f} |",
        "",
        "## Worst Groups By Exact-Set Accuracy",
        "",
        "| Group | Records | Exact-set | Hamming |",
        "|---|---:|---:|---:|",
    ]
    for group, summary in top_groups:
        markdown.append(
            f"| {group} | {summary['record_count']} | {summary['exact_set_accuracy']:.6f} | {summary['hamming_loss']:.6f} |"
        )
    markdown.extend(
        [
            "",
            "## Worst Group Highlight",
            "",
            f"- Worst exact-set group: `{worst_group[0]}`",
            f"- Worst exact-set accuracy: `{worst_group[1]['exact_set_accuracy']:.6f}`",
            f"- Worst inner-accuracy group: `{worst_group_by_accuracy[0]}`",
            "",
        ]
    )
    _atomic_text(OUTPUT_MD, "\n".join(markdown))
    return report


def main() -> None:
    print(json.dumps(build_report(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
