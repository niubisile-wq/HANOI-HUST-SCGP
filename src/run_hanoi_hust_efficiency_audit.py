"""Audit evaluation-time efficiency for the frozen HANOI HUST methods."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
import tracemalloc
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from electrical_fm.hanoi_hust_baselines import (
    Candidate,
    bearing_groups,
    component_targets,
    evaluate_candidate,
    representation_views,
)
from electrical_fm.hanoi_hust_noace import evaluate_noace_candidate


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "artifacts" / "hanoi_hust" / "source_features.npz"
METADATA = ROOT / "artifacts" / "hanoi_hust" / "source_features_metadata.json"
DEFAULT_OUTPUT = ROOT / "results" / "analysis" / "hanoi_hust_efficiency.json"
DEFAULT_MD = ROOT / "results" / "analysis" / "hanoi_hust_efficiency.md"


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _as_artifact_path(value: Path) -> str:
    resolved = value.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


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
        json.dumps(_jsonable(payload), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(staging, path)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix(".writing")
    staging.write_text(text, encoding="utf-8")
    os.replace(staging, path)


def _timed_repeat(function, repeats: int) -> dict[str, Any]:
    elapsed = []
    peaks = []
    result: dict[str, Any] | None = None
    for _ in range(repeats):
        tracemalloc.start()
        start = time.perf_counter()
        result = function()
        elapsed.append(time.perf_counter() - start)
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peaks.append(peak / (1024**2))
    assert result is not None
    return {
        "result": result,
        "elapsed_seconds": elapsed,
        "peak_tracemalloc_mib": peaks,
    }


def _describe(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p95": float(np.quantile(arr, 0.95)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _source_champion(random_state: int) -> Candidate:
    return Candidate(
        family="logistic_l2",
        representation="envelope_log_power",
        hyperparameter={"C": 10.0},
        build=lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=10.0,
                class_weight="balanced",
                max_iter=5_000,
                random_state=random_state,
            ),
        ),
    )


def _noace_deep(random_state: int) -> Candidate:
    return Candidate(
        family="mlp_deep",
        representation="statistics",
        hyperparameter={
            "hidden_layer_sizes": (64, 64, 32),
            "alpha": 1e-4,
            "learning_rate_init": 1e-3,
            "early_stopping": True,
        },
        build=lambda: make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(64, 64, 32),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                batch_size="auto",
                learning_rate_init=1e-3,
                max_iter=800,
                early_stopping=True,
                n_iter_no_change=30,
                validation_fraction=0.2,
                shuffle=True,
                random_state=random_state,
            ),
        ),
    )


def build_efficiency_audit(
    *,
    random_state: int,
    repeats: int,
    output: Path,
    markdown: Path,
) -> dict[str, Any]:
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    if metadata.get("status") != "source_features_complete_compound_sealed":
        raise RuntimeError("HANOI HUST source cache is not complete")
    if _sha256(CACHE) != metadata.get("cache", {}).get("sha256"):
        raise RuntimeError("HANOI HUST source cache hash changed")

    with np.load(CACHE, allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}

    states = arrays["state"].astype(str)
    groups = bearing_groups(states, arrays["bearing_type"])
    targets = component_targets(states)
    views = representation_views(arrays)

    method_specs = [
        {
            "key": "source_champion",
            "label": "Source champion",
            "kind": "supervised",
            "representation": "envelope_log_power",
            "candidate": _source_champion(random_state),
        },
        {
            "key": "noace_classical",
            "label": "NOACE classical",
            "kind": "noace",
            "representation": "all",
            "candidate": None,
        },
        {
            "key": "noace_deep",
            "label": "NOACE deep",
            "kind": "supervised",
            "representation": "statistics",
            "candidate": _noace_deep(random_state),
        },
    ]

    method_rows: list[dict[str, Any]] = []
    for spec in method_specs:
        base_view = views[spec["representation"]]

        def run_once() -> dict[str, Any]:
            if spec["kind"] == "noace":
                return evaluate_noace_candidate(
                    features=base_view,
                    states=states,
                    bearing_type=arrays["bearing_type"],
                    load_w=arrays["load_w"],
                    groups=groups,
                    random_seed=random_state,
                    representation=spec["representation"],
                )
            assert spec["candidate"] is not None
            return evaluate_candidate(
                base_view,
                targets,
                groups,
                spec["candidate"],
            )

        timed = _timed_repeat(run_once, repeats)
        clean_result = timed["result"]
        method_rows.append(
            {
                "method_key": spec["key"],
                "method_label": spec["label"],
                "kind": spec["kind"],
                "representation": spec["representation"],
                "candidate": spec["candidate"].hyperparameter if spec["candidate"] else {},
                "feature_dimension": int(base_view.shape[1]),
                "repeats": repeats,
                "evaluation_seconds": _describe(timed["elapsed_seconds"]),
                "peak_tracemalloc_mib": _describe(timed["peak_tracemalloc_mib"]),
                "metrics": {
                    key: clean_result[key]
                    for key in (
                        "mean_component_auroc",
                        "mean_component_balanced_accuracy",
                        "exact_set_accuracy",
                        "mean_brier_score",
                        "hamming_loss",
                    )
                },
            }
        )

    payload: dict[str, Any] = {
        "stage": "hanoi_hust_efficiency_audit",
        "schema_version": 1,
        "status": "completed",
        "cache": {
            "path": CACHE.relative_to(ROOT).as_posix(),
            "sha256": _sha256(CACHE),
        },
        "metadata": {
            "path": METADATA.relative_to(ROOT).as_posix(),
            "sha256": _sha256(METADATA),
        },
        "record_count": int(len(states)),
        "bearing_count": int(len(np.unique(groups))),
        "methods": method_rows,
        "provenance": {
            "git_commit": _git("rev-parse", "HEAD"),
            "builder": "src/run_hanoi_hust_efficiency_audit.py",
            "random_state": int(random_state),
            "repeats": int(repeats),
        },
    }
    _atomic_json(output, payload)

    lines = [
        "# HANOI HUST Efficiency Audit",
        "",
        "Frozen source-cache evaluation time and Python-level peak allocation.",
        "",
        "| Method | Rep. | Mean AUROC | Exact-set | Eval median (s) | Eval p95 (s) | Peak tracemalloc MiB |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in method_rows:
        metrics = row["metrics"]
        elapsed = row["evaluation_seconds"]
        peak = row["peak_tracemalloc_mib"]
        lines.append(
            f"| {row['method_label']} | {row['representation']} | "
            f"{metrics['mean_component_auroc']:.6f} | {metrics['exact_set_accuracy']:.6f} | "
            f"{elapsed['median']:.4f} | {elapsed['p95']:.4f} | {peak['median']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This is a protocol-level efficiency audit, not a GPU kernel benchmark.",
            "- It compares the clean source champion, classical NOACE, and NOACE-Deep under identical frozen access.",
            "- The result is intended for the paper's complexity and cost discussion.",
        ]
    )
    _atomic_text(markdown, "\n".join(lines))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-state", type=int, default=20_260_727)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()
    print(
        json.dumps(
            build_efficiency_audit(
                random_state=args.random_state,
                repeats=args.repeats,
                output=args.output,
                markdown=args.markdown,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
