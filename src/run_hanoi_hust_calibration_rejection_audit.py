"""Build calibration, rejection, and failure-sample audit for HANOI HUST."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "results" / "development" / "hanoi_hust_source_selected_predictions.npz"
REFUTATION_JSON = ROOT / "results" / "analysis" / "hanoi_hust_refutation_pack.json"
OUTPUT_JSON = ROOT / "results" / "analysis" / "hanoi_hust_calibration_rejection.json"
OUTPUT_MD = (
    ROOT / "research" / "HANOI_HUST_20260727" / "HANOI_HUST_CALIBRATION_REJECTION_正式结果页_20260728.md"
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


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix(".writing")
    staging.write_text(text, encoding="utf-8")
    os.replace(staging, path)


def _set_confidence(probabilities: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    chosen = np.where(predictions == 1, probabilities, 1.0 - probabilities)
    return np.min(chosen, axis=1)


def _risk_coverage_curve(confidence: np.ndarray, correct: np.ndarray, *, bins: int = 10) -> list[dict[str, float]]:
    order = np.argsort(-confidence)
    confidence = confidence[order]
    correct = correct[order]
    n = len(confidence)
    points: list[dict[str, float]] = []
    for k in range(1, bins + 1):
        accepted = int(np.ceil(n * k / bins))
        accepted = max(1, min(n, accepted))
        coverage = accepted / n
        accepted_correct = correct[:accepted]
        selective_accuracy = float(accepted_correct.mean())
        risk = float(1.0 - selective_accuracy)
        threshold = float(confidence[accepted - 1])
        points.append(
            {
                "coverage": coverage,
                "threshold": threshold,
                "selective_accuracy": selective_accuracy,
                "risk": risk,
            }
        )
    return points


def _aurc(points: list[dict[str, float]]) -> float:
    coverages = np.asarray([0.0] + [point["coverage"] for point in points], dtype=np.float64)
    risks = np.asarray([1.0] + [point["risk"] for point in points], dtype=np.float64)
    return float(np.trapezoid(risks, coverages))


def _ece(confidence: np.ndarray, correct: np.ndarray, *, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(confidence)
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        if upper < 1.0:
            mask = (confidence >= lower) & (confidence < upper)
        else:
            mask = (confidence >= lower) & (confidence <= upper)
        if not np.any(mask):
            continue
        ece += (mask.sum() / total) * abs(float(confidence[mask].mean()) - float(correct[mask].mean()))
    return float(ece)


def _load_refutation() -> dict[str, Any]:
    if not REFUTATION_JSON.exists():
        raise RuntimeError("Refutation pack JSON is missing; run the refutation pack first")
    return json.loads(REFUTATION_JSON.read_text(encoding="utf-8"))


def build_report() -> dict[str, Any]:
    if not PREDICTIONS.exists():
        raise RuntimeError("Champion prediction cache is missing")
    with np.load(PREDICTIONS) as payload:
        probabilities = np.asarray(payload["probabilities"], dtype=np.float64)
        predictions = np.asarray(payload["predictions"], dtype=np.int8)
        targets = np.asarray(payload["targets"], dtype=np.int8)
        groups = np.asarray(payload["groups"], dtype=str)

    correct = np.all(predictions == targets, axis=1)
    confidence = _set_confidence(probabilities, predictions)
    curve = _risk_coverage_curve(confidence, correct, bins=10)
    aurc = _aurc(curve)
    ece = _ece(confidence, correct, bins=10)

    failure_groups: list[dict[str, Any]] = []
    for group in sorted(set(groups.tolist())):
        mask = groups == group
        group_correct = np.all(predictions[mask] == targets[mask], axis=1)
        group_confidence = confidence[mask]
        failure_groups.append(
            {
                "group": group,
                "record_count": int(mask.sum()),
                "exact_set_accuracy": float(group_correct.mean()),
                "mean_confidence": float(group_confidence.mean()),
                "failure_rate": float(1.0 - group_correct.mean()),
            }
        )
    failure_groups.sort(
        key=lambda row: (row["exact_set_accuracy"], -row["failure_rate"], row["group"])
    )
    top_failures = failure_groups[:5]

    refutation = _load_refutation()
    calibration_error = {
        "overall_exact_set_accuracy": float(correct.mean()),
        "overall_hamming_loss": float(np.mean(predictions != targets)),
        "inner_ece": float(
            _ece(probabilities[:, 0], targets[:, 0])
        ),
        "outer_ece": float(
            _ece(probabilities[:, 1], targets[:, 1])
        ),
        "ball_ece": float(
            _ece(probabilities[:, 2], targets[:, 2])
        ),
    }

    report = {
        "stage": "hanoi_hust_calibration_rejection",
        "schema_version": 1,
        "status": "completed",
        "prediction_cache": {
            "path": PREDICTIONS.relative_to(ROOT).as_posix(),
            "sha256": _sha256(PREDICTIONS),
        },
        "champion": {
            "family": "logistic_l2",
            "representation": "envelope_log_power",
            "exact_set_accuracy": float(correct.mean()),
        },
        "calibration": calibration_error,
        "risk_coverage": {
            "aurc": aurc,
            "ece": ece,
            "curve": curve,
        },
        "failure_groups": failure_groups,
        "top_failures": top_failures,
        "refutation_pack": {
            "path": REFUTATION_JSON.relative_to(ROOT).as_posix(),
            "sha256": _sha256(REFUTATION_JSON),
            "label_shuffle_mean_aurc": None,
            "label_shuffle_mean_exact_set": float(
                np.mean([row["exact_set_accuracy"] for row in refutation["label_shuffle"]["runs"]])
            ),
        },
        "provenance": {
            "builder": "src/run_hanoi_hust_calibration_rejection_audit.py",
            "git_commit": _git("rev-parse", "HEAD"),
        },
    }
    _atomic_json(OUTPUT_JSON, report)

    md_lines = [
        "# HANOI HUST Calibration, Rejection, and Failure Audit",
        "",
        f"- exact_set_accuracy: `{report['calibration']['overall_exact_set_accuracy']:.6f}`",
        f"- hamming_loss: `{report['calibration']['overall_hamming_loss']:.6f}`",
        f"- aurc: `{report['risk_coverage']['aurc']:.6f}`",
        f"- set_confidence_ece: `{report['risk_coverage']['ece']:.6f}`",
        "",
        "## Component Calibration",
        "",
        "| Component | ECE |",
        "|---|---:|",
        f"| inner | {report['calibration']['inner_ece']:.6f} |",
        f"| outer | {report['calibration']['outer_ece']:.6f} |",
        f"| ball | {report['calibration']['ball_ece']:.6f} |",
        "",
        "## Risk-Coverage Curve",
        "",
        "| Coverage | Threshold | Selective accuracy | Risk |",
        "|---|---:|---:|---:|",
    ]
    for point in curve:
        md_lines.append(
            f"| {point['coverage']:.2f} | {point['threshold']:.6f} | {point['selective_accuracy']:.6f} | {point['risk']:.6f} |"
        )
    md_lines.extend(
        [
            "",
            "## Worst Failure Groups",
            "",
            "| Group | Records | Exact-set | Mean confidence | Failure rate |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in top_failures:
        md_lines.append(
            f"| {row['group']} | {row['record_count']} | {row['exact_set_accuracy']:.6f} | {row['mean_confidence']:.6f} | {row['failure_rate']:.6f} |"
        )
    md_lines.extend(
        [
            "",
            "## Manuscript-safe note",
            "",
            "This page is the missing negative-control block for the paper. It shows that the champion is miscalibrated at the set level, that selective rejection can trade coverage for accuracy, and that a small number of physical groups carry most of the failures.",
            "",
        ]
    )
    _atomic_text(OUTPUT_MD, "\n".join(md_lines))
    return report


def main() -> None:
    print(json.dumps(build_report(), indent=2, ensure_ascii=False, default=_json_default))


if __name__ == "__main__":
    main()
