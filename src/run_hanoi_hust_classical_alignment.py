"""Build classical baseline alignment results for HANOI HUST."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

from electrical_fm.hanoi_hust_baselines import (
    Candidate,
    bearing_groups,
    component_targets,
    evaluate_candidate,
    evaluate_candidates,
    representation_views,
    _EmpiricalPriorClassifier,
)


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "artifacts" / "hanoi_hust" / "source_features.npz"
METADATA = ROOT / "artifacts" / "hanoi_hust" / "source_features_metadata.json"
OUTPUT_JSON = ROOT / "results" / "analysis" / "hanoi_hust_classical_alignment.json"
OUTPUT_MD = (
    ROOT / "research" / "HANOI_HUST_20260727" / "HANOI_HUST_CLASSICAL_ALIGNMENT_正式结果页_20260728.md"
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


def _load_arrays() -> dict[str, np.ndarray]:
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    if metadata.get("status") != "source_features_complete_compound_sealed":
        raise RuntimeError("HANOI HUST source cache is not complete")
    if _sha256(CACHE) != metadata.get("cache", {}).get("sha256"):
        raise RuntimeError("HANOI HUST source cache hash changed")
    with np.load(CACHE, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def _candidate_factories(random_state: int) -> list[tuple[str, Candidate]]:
    return [
        (
            "rf",
            Candidate(
                family="rf",
                representation="all",
                hyperparameter={"n_estimators": 500, "max_features": "sqrt"},
                build=lambda: RandomForestClassifier(
                    n_estimators=500,
                    max_features="sqrt",
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ),
        (
            "linear_svm",
            Candidate(
                family="linear_svm",
                representation="all",
                hyperparameter={"C": 1.0},
                build=lambda: CalibratedClassifierCV(
                    make_pipeline(
                        StandardScaler(),
                        LinearSVC(C=1.0, class_weight="balanced", random_state=random_state),
                    ),
                    ensemble=False,
                    cv=3,
                ),
            ),
        ),
        (
            "rbf_svm",
            Candidate(
                family="rbf_svm",
                representation="all",
                hyperparameter={"C": 10.0, "gamma": "scale"},
                build=lambda: CalibratedClassifierCV(
                    make_pipeline(
                        StandardScaler(),
                        SVC(
                            C=10.0,
                            gamma="scale",
                            kernel="rbf",
                            class_weight="balanced",
                            probability=False,
                            random_state=random_state,
                        ),
                    ),
                    ensemble=False,
                    cv=3,
                ),
            ),
        ),
        (
            "extra_trees",
            Candidate(
                family="extra_trees",
                representation="all",
                hyperparameter={"min_samples_leaf": 4},
                build=lambda: ExtraTreesClassifier(
                    n_estimators=500,
                    min_samples_leaf=4,
                    max_features="sqrt",
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ),
        (
            "empirical_prior",
            Candidate(
                family="empirical_prior",
                representation="all",
                hyperparameter={},
                build=lambda: _EmpiricalPriorClassifier(),
            ),
        ),
        (
            "logistic_l2",
            Candidate(
                family="logistic_l2",
                representation="all",
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
            ),
        ),
    ]


def build_report(*, random_state: int = 20_260_728) -> dict[str, Any]:
    arrays = _load_arrays()
    views = representation_views(arrays)
    states = arrays["state"].astype(str)
    targets = component_targets(states)
    groups = bearing_groups(states, arrays["bearing_type"])

    rows: list[dict[str, Any]] = []
    for family_name, candidate in _candidate_factories(random_state):
        for view_name, features in views.items():
            screened = Candidate(
                family=family_name,
                representation=view_name,
                hyperparameter=candidate.hyperparameter,
                build=candidate.build,
            )
            row = evaluate_candidate(features, targets, groups, screened)
            rows.append(row)

    rows.sort(
        key=lambda row: (
            -row["mean_component_auroc"],
            -row["exact_set_accuracy"],
            row["mean_brier_score"],
            row["feature_dimension"],
            row["family"],
            row["representation"],
            json.dumps(row["hyperparameter"], sort_keys=True),
        )
    )

    family_summary: dict[str, dict[str, Any]] = {}
    for row in rows:
        family = row["family"]
        if family not in family_summary:
            family_summary[family] = {
                "family": family,
                "representation": row["representation"],
                "hyperparameter": row["hyperparameter"],
                "mean_component_auroc": float(row["mean_component_auroc"]),
                "mean_component_balanced_accuracy": float(row["mean_component_balanced_accuracy"]),
                "mean_component_aupr": float(row["mean_component_aupr"]),
                "mean_brier_score": float(row["mean_brier_score"]),
                "mean_component_macro_f1": float(row["mean_component_macro_f1"]),
                "exact_set_accuracy": float(row["exact_set_accuracy"]),
                "hamming_loss": float(row["hamming_loss"]),
            }

    report = {
        "stage": "hanoi_hust_classical_alignment",
        "schema_version": 1,
        "status": "completed",
        "cache": {
            "path": CACHE.relative_to(ROOT).as_posix(),
            "sha256": _sha256(CACHE),
        },
        "records": int(len(states)),
        "bearing_count": int(len(np.unique(groups))),
        "rows": [
            {
                key: value
                for key, value in row.items()
                if key not in {"probabilities", "predictions", "targets"}
            }
            for row in rows
        ],
        "family_summary": family_summary,
        "champion": family_summary[rows[0]["family"]],
        "provenance": {
            "builder": "src/run_hanoi_hust_classical_alignment.py",
            "git_commit": _git("rev-parse", "HEAD"),
            "random_state": int(random_state),
        },
    }
    _atomic_json(OUTPUT_JSON, report)

    md_lines = [
        "# HANOI HUST Classical Alignment Formal Results Page",
        "",
        "## Family Champions",
        "",
        "| Family | Representation | Mean AUROC | Mean CBA | Mean AUPR | Mean Brier | Mean Macro F1 | Exact-set | Hamming |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for family in ("rf", "linear_svm", "rbf_svm", "extra_trees", "empirical_prior", "logistic_l2"):
        row = family_summary.get(family)
        if row is None:
            continue
        md_lines.append(
            "| {family} | {representation} | {mean_component_auroc:.6f} | {mean_component_balanced_accuracy:.6f} | {mean_component_aupr:.6f} | {mean_brier_score:.6f} | {mean_component_macro_f1:.6f} | {exact_set_accuracy:.6f} | {hamming_loss:.6f} |".format(
                **row
            )
        )
    md_lines.extend(
        [
            "",
            "## Manuscript-safe note",
            "",
            "This page aligns the classical comparator families under the frozen source cache. It is the direct evidence page for the Stage 2 alignment step, but it should not be treated as the final WDCNN reproduction, which still requires a separate deep-learning pipeline.",
            "",
        ]
    )
    _atomic_text(OUTPUT_MD, "\n".join(md_lines))
    return report


def main() -> None:
    print(json.dumps(build_report(), indent=2, ensure_ascii=False, default=_json_default))


if __name__ == "__main__":
    main()
