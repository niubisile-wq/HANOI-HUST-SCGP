"""Evaluate fixed HANOI HUST methods under additive Gaussian noise stress."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

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
DEFAULT_OUTPUT = ROOT / "results" / "analysis" / "hanoi_hust_noise_robustness.json"
DEFAULT_MD = ROOT / "results" / "analysis" / "hanoi_hust_noise_robustness.md"

SNR_LEVELS = [20, 10, 5, 0, -5]


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


def _noise_perturb(
    values: np.ndarray,
    *,
    snr_db: int | None,
    seed: int,
) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    if snr_db is None:
        return x.copy()
    rng = np.random.default_rng(seed)
    signal_rms = np.sqrt(np.mean(np.square(x), axis=1, keepdims=True, dtype=np.float64))
    noise_scale = signal_rms / np.sqrt(10.0 ** (float(snr_db) / 10.0))
    noise = rng.normal(0.0, 1.0, size=x.shape).astype(np.float32)
    noise *= noise_scale.astype(np.float32)
    return x + noise


def _fixed_source_champion(random_state: int) -> Candidate:
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


def _fixed_deep_champion(random_state: int) -> Candidate:
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


def _summarize(clean: dict[str, Any], stressed: dict[str, Any]) -> dict[str, Any]:
    return {
        "mean_component_auroc": stressed["mean_component_auroc"],
        "mean_component_balanced_accuracy": stressed["mean_component_balanced_accuracy"],
        "exact_set_accuracy": stressed["exact_set_accuracy"],
        "mean_brier_score": stressed["mean_brier_score"],
        "hamming_loss": stressed["hamming_loss"],
        "delta_mean_component_auroc": (
            None
            if clean["mean_component_auroc"] is None
            or stressed["mean_component_auroc"] is None
            else float(stressed["mean_component_auroc"] - clean["mean_component_auroc"])
        ),
        "delta_mean_component_balanced_accuracy": float(
            stressed["mean_component_balanced_accuracy"]
            - clean["mean_component_balanced_accuracy"]
        ),
        "delta_exact_set_accuracy": float(
            stressed["exact_set_accuracy"] - clean["exact_set_accuracy"]
        ),
        "delta_mean_brier_score": float(
            stressed["mean_brier_score"] - clean["mean_brier_score"]
        ),
        "delta_hamming_loss": float(stressed["hamming_loss"] - clean["hamming_loss"]),
    }


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    kind: str
    representation: str
    candidate: Candidate | None = None


def build_noise_robustness(
    *,
    random_state: int,
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
        MethodSpec(
            key="source_champion",
            label="Source champion",
            kind="supervised",
            representation="envelope_log_power",
            candidate=_fixed_source_champion(random_state),
        ),
        MethodSpec(
            key="noace_classical",
            label="NOACE classical",
            kind="noace",
            representation="all",
        ),
        MethodSpec(
            key="noace_deep",
            label="NOACE deep",
            kind="supervised",
            representation="statistics",
            candidate=_fixed_deep_champion(random_state),
        ),
    ]

    rows: list[dict[str, Any]] = []
    for method_index, spec in enumerate(method_specs):
        base_view = views[spec.representation]
        clean_result: dict[str, Any]
        if spec.kind == "noace":
            clean_result = evaluate_noace_candidate(
                features=base_view,
                states=states,
                bearing_type=arrays["bearing_type"],
                load_w=arrays["load_w"],
                groups=groups,
                random_seed=random_state,
                representation=spec.representation,
            )
        else:
            assert spec.candidate is not None
            clean_result = evaluate_candidate(
                base_view,
                targets,
                groups,
                spec.candidate,
            )

        for snr_db in [None, *SNR_LEVELS]:
            noisy_view = _noise_perturb(
                base_view,
                snr_db=snr_db,
                seed=random_state + method_index * 100 + (0 if snr_db is None else int(snr_db) + 50),
            )
            if spec.kind == "noace":
                stressed_result = evaluate_noace_candidate(
                    features=noisy_view,
                    states=states,
                    bearing_type=arrays["bearing_type"],
                    load_w=arrays["load_w"],
                    groups=groups,
                    random_seed=random_state,
                    representation=spec.representation,
                )
            else:
                assert spec.candidate is not None
                stressed_result = evaluate_candidate(
                    noisy_view,
                    targets,
                    groups,
                    spec.candidate,
                )
            rows.append(
                {
                    "method_key": spec.key,
                    "method_label": spec.label,
                    "representation": spec.representation,
                    "condition": "clean" if snr_db is None else f"{snr_db}dB",
                    "snr_db": snr_db,
                    "metrics": {
                        key: stressed_result[key]
                        for key in (
                            "mean_component_auroc",
                            "mean_component_balanced_accuracy",
                            "exact_set_accuracy",
                            "mean_brier_score",
                            "hamming_loss",
                        )
                    },
                    "delta_vs_clean": _summarize(
                        {
                            key: clean_result[key]
                            for key in (
                                "mean_component_auroc",
                                "mean_component_balanced_accuracy",
                                "exact_set_accuracy",
                                "mean_brier_score",
                                "hamming_loss",
                            )
                        },
                        {
                            key: stressed_result[key]
                            for key in (
                                "mean_component_auroc",
                                "mean_component_balanced_accuracy",
                                "exact_set_accuracy",
                                "mean_brier_score",
                                "hamming_loss",
                            )
                        },
                    ),
                }
            )

    payload: dict[str, Any] = {
        "stage": "hanoi_hust_noise_robustness",
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
        "snr_levels_db": SNR_LEVELS,
        "rows": rows,
        "provenance": {
            "git_commit": _git("rev-parse", "HEAD"),
            "builder": "src/run_hanoi_hust_noise_robustness.py",
            "random_state": int(random_state),
        },
    }
    _atomic_json(output, payload)

    method_rows: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        method_rows.setdefault(row["method_key"], []).append(row)

    lines = [
        "# HANOI HUST Noise Robustness",
        "",
        "This is a feature-space Gaussian noise stress test on the frozen source cache.",
        "",
        "| Method | Condition | Mean AUROC | Exact-set | Brier | Delta AUROC | Delta Exact-set |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for spec in method_specs:
        for row in method_rows[spec.key]:
            metrics = row["metrics"]
            delta = row["delta_vs_clean"]
            delta_auroc = (
                ""
                if delta["delta_mean_component_auroc"] is None
                else f"{delta['delta_mean_component_auroc']:.6f}"
            )
            lines.append(
                f"| {spec.label} | {row['condition']} | {metrics['mean_component_auroc']:.6f} | "
                f"{metrics['exact_set_accuracy']:.6f} | {metrics['mean_brier_score']:.6f} | "
                f"{delta_auroc} | "
                f"{delta['delta_exact_set_accuracy']:.6f} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The curve shows how each fixed method degrades as additive feature noise increases.",
            "- This is a stress test, not a new method-search result.",
            "- The clean row is the frozen baseline; the noisy rows quantify sensitivity.",
        ]
    )
    _atomic_text(markdown, "\n".join(lines))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-state", type=int, default=20_260_727)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()
    print(
        json.dumps(
            build_noise_robustness(
                random_state=args.random_state,
                output=args.output,
                markdown=args.markdown,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
