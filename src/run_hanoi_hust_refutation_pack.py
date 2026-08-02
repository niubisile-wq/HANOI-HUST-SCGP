"""Build HANOI HUST negative-control / refutation results."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut

from electrical_fm.hanoi_hust_baselines import (
    Candidate,
    bearing_groups,
    component_targets,
    evaluate_candidate,
    evaluate_candidates,
    representation_views,
)


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "artifacts" / "hanoi_hust" / "source_features.npz"
METADATA = ROOT / "artifacts" / "hanoi_hust" / "source_features_metadata.json"
OUTPUT_JSON = ROOT / "results" / "analysis" / "hanoi_hust_refutation_pack.json"
OUTPUT_MD = (
    ROOT / "research" / "HANOI_HUST_20260727" / "HANOI_HUST_REFUTATION_PACK_正式结果页_20260728.md"
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


def _load_arrays() -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    if metadata.get("status") != "source_features_complete_compound_sealed":
        raise RuntimeError("HANOI HUST source cache is not complete")
    if _sha256(CACHE) != metadata.get("cache", {}).get("sha256"):
        raise RuntimeError("HANOI HUST source cache hash changed")
    with np.load(CACHE, allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    return arrays, metadata


def _candidate_from_row(row: dict[str, Any], *, random_state: int) -> Candidate:
    family = row["family"]
    hyperparameter = row["hyperparameter"]
    if family == "logistic_l2":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        c_value = float(hyperparameter["C"])
        return Candidate(
            family=family,
            representation=row["representation"],
            hyperparameter=hyperparameter,
            build=lambda: make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    max_iter=5_000,
                    random_state=random_state,
                ),
            ),
        )
    if family == "extra_trees":
        leaf = int(hyperparameter["min_samples_leaf"])
        return Candidate(
            family=family,
            representation=row["representation"],
            hyperparameter=hyperparameter,
            build=lambda: ExtraTreesClassifier(
                n_estimators=500,
                min_samples_leaf=leaf,
                max_features="sqrt",
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
            ),
        )
    if family == "empirical_prior":
        from electrical_fm.hanoi_hust_baselines import _EmpiricalPriorClassifier

        return Candidate(
            family=family,
            representation=row["representation"],
            hyperparameter=hyperparameter,
            build=lambda: _EmpiricalPriorClassifier(),
        )
    if family == "rbf_svm":
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC

        c_value = float(hyperparameter["C"])
        gamma = str(hyperparameter["gamma"])
        return Candidate(
            family=family,
            representation=row["representation"],
            hyperparameter=hyperparameter,
            build=lambda: CalibratedClassifierCV(
                make_pipeline(
                    StandardScaler(),
                    SVC(
                        C=c_value,
                        gamma=gamma,
                        kernel="rbf",
                        class_weight="balanced",
                        probability=False,
                        random_state=random_state,
                    ),
                ),
                ensemble=False,
            ),
        )
    raise RuntimeError(f"Unsupported champion family: {family}")


def _champion_candidate(
    features_by_view: dict[str, np.ndarray],
    targets: np.ndarray,
    groups: np.ndarray,
    *,
    random_state: int,
) -> tuple[Candidate, dict[str, Any]]:
    ranked = evaluate_candidates(
        features_by_view,
        targets,
        groups,
        random_state=random_state,
    )
    champion_row = ranked[0]
    candidate = _candidate_from_row(champion_row, random_state=random_state)
    return candidate, champion_row


def _shuffle_targets(targets: np.ndarray, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shuffled = np.empty_like(targets)
    for head in range(targets.shape[1]):
        permutation = rng.permutation(len(targets))
        shuffled[:, head] = targets[permutation, head]
    return shuffled


def _probe_leave_one_group_out(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    random_state: int,
) -> dict[str, Any]:
    splitter = LeaveOneGroupOut()
    predicted = np.empty_like(labels)
    fold_count = 0
    for train_index, test_index in splitter.split(features, labels, groups):
        fold_count += 1
        model = ExtraTreesClassifier(
            n_estimators=400,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(features[train_index], labels[train_index])
        predicted[test_index] = model.predict(features[test_index])
    return {
        "fold_count": fold_count,
        "accuracy": float(accuracy_score(labels, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "macro_f1": float(f1_score(labels, predicted, average="macro")),
    }


def build_report(
    *,
    random_state: int = 20_260_728,
    shuffle_seeds: tuple[int, ...] = (20_260_728, 42, 84, 168, 31415),
) -> dict[str, Any]:
    arrays, metadata = _load_arrays()
    views = representation_views(arrays)
    states = arrays["state"].astype(str)
    targets = component_targets(states)
    groups = bearing_groups(states, arrays["bearing_type"])
    champion_candidate, champion_row = _champion_candidate(
        views,
        targets,
        groups,
        random_state=random_state,
    )

    champion_features = views[champion_row["representation"]]
    shuffled_runs: list[dict[str, Any]] = []
    for seed in shuffle_seeds:
        shuffled_targets = _shuffle_targets(targets, seed=seed)
        result = evaluate_candidate(
            champion_features,
            shuffled_targets,
            groups,
            champion_candidate,
        )
        shuffled_runs.append(
            {
                "seed": int(seed),
                "mean_component_auroc": float(result["mean_component_auroc"]),
                "mean_component_balanced_accuracy": float(
                    result["mean_component_balanced_accuracy"]
                ),
                "mean_component_aupr": float(result["mean_component_aupr"]),
                "mean_brier_score": float(result["mean_brier_score"]),
                "mean_component_macro_f1": float(result["mean_component_macro_f1"]),
                "exact_set_accuracy": float(result["exact_set_accuracy"]),
                "hamming_loss": float(result["hamming_loss"]),
            }
        )

    def _agg(field: str) -> dict[str, float]:
        values = np.asarray([row[field] for row in shuffled_runs], dtype=np.float64)
        return {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "min": float(values.min()),
            "max": float(values.max()),
        }

    bearing_type_probe = _probe_leave_one_group_out(
        champion_features,
        arrays["bearing_type"].astype(np.int64),
        arrays["load_w"].astype(np.int64),
        random_state=random_state,
    )
    load_probe = _probe_leave_one_group_out(
        champion_features,
        arrays["load_w"].astype(np.int64),
        arrays["bearing_type"].astype(np.int64),
        random_state=random_state,
    )

    lobo_groups = json.loads(
        json.dumps(
            {
                group: {
                    "record_count": int(len(np.where(groups == group)[0])),
                    "exact_set_accuracy": float(
                        np.mean(
                            np.all(
                                champion_row["predictions"][groups == group]
                                == champion_row["targets"][groups == group],
                                axis=1,
                            )
                        )
                    ),
                    "hamming_loss": float(
                        np.mean(
                            champion_row["predictions"][groups == group]
                            != champion_row["targets"][groups == group]
                        )
                    ),
                }
                for group in sorted(set(groups.tolist()))
            },
            ensure_ascii=False,
            default=_json_default,
        )
    )
    worst_groups = sorted(
        lobo_groups.items(),
        key=lambda item: (item[1]["exact_set_accuracy"], item[1]["hamming_loss"], item[0]),
    )[:5]

    report = {
        "stage": "hanoi_hust_refutation_pack",
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
        "champion": {
            key: value
            for key, value in champion_row.items()
            if key not in {"probabilities", "predictions", "targets"}
        },
        "label_shuffle": {
            "seeds": list(shuffle_seeds),
            "runs": shuffled_runs,
            "aggregates": {
                field: _agg(field)
                for field in (
                    "mean_component_auroc",
                    "mean_component_balanced_accuracy",
                    "mean_component_aupr",
                    "mean_brier_score",
                    "mean_component_macro_f1",
                    "exact_set_accuracy",
                    "hamming_loss",
                )
            },
        },
        "identity_probes": {
            "bearing_type_from_features_by_load_split": bearing_type_probe,
            "load_from_features_by_bearing_split": load_probe,
        },
        "leave_one_bearing_out": {
            "group_count": int(len(lobo_groups)),
            "overall": {
                "exact_set_accuracy": float(
                    np.mean(np.all(champion_row["predictions"] == champion_row["targets"], axis=1))
                ),
                "hamming_loss": float(
                    np.mean(champion_row["predictions"] != champion_row["targets"])
                ),
            },
            "groups": lobo_groups,
            "worst_groups": worst_groups,
        },
        "provenance": {
            "builder": "src/run_hanoi_hust_refutation_pack.py",
            "git_commit": _git("rev-parse", "HEAD"),
            "random_state": int(random_state),
        },
    }
    _atomic_json(OUTPUT_JSON, report)

    md_lines = [
        "# HANOI HUST Refutation Pack",
        "",
        f"- champion_family: `{champion_row['family']}`",
        f"- champion_representation: `{champion_row['representation']}`",
        f"- champion_exact_set_accuracy: `{champion_row['exact_set_accuracy']:.6f}`",
        f"- champion_mean_auroc: `{champion_row['mean_component_auroc']:.6f}`",
        "",
        "## Label Shuffle",
        "",
        "| Seed | Mean AUROC | Mean CBA | AUPR | Brier | Macro F1 | Exact-set | Hamming |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in shuffled_runs:
        md_lines.append(
            "| {seed} | {mean_component_auroc:.6f} | {mean_component_balanced_accuracy:.6f} | {mean_component_aupr:.6f} | {mean_brier_score:.6f} | {mean_component_macro_f1:.6f} | {exact_set_accuracy:.6f} | {hamming_loss:.6f} |".format(
                **row
            )
        )
    md_lines.extend(
        [
            "",
            "## Identity Probes",
            "",
            "| Probe | Accuracy | Balanced Accuracy | Macro F1 | Folds |",
            "|---|---:|---:|---:|---:|",
            f"| bearing_type from features (load-held-out) | {bearing_type_probe['accuracy']:.6f} | {bearing_type_probe['balanced_accuracy']:.6f} | {bearing_type_probe['macro_f1']:.6f} | {bearing_type_probe['fold_count']} |",
            f"| load from features (bearing-held-out) | {load_probe['accuracy']:.6f} | {load_probe['balanced_accuracy']:.6f} | {load_probe['macro_f1']:.6f} | {load_probe['fold_count']} |",
            "",
            "## Leave-One-Bearing-Out",
            "",
            f"- overall_exact_set_accuracy: `{report['leave_one_bearing_out']['overall']['exact_set_accuracy']:.6f}`",
            f"- overall_hamming_loss: `{report['leave_one_bearing_out']['overall']['hamming_loss']:.6f}`",
            f"- group_count: `{report['leave_one_bearing_out']['group_count']}`",
            "",
            "| Group | Records | Exact-set | Hamming |",
            "|---|---:|---:|---:|",
        ]
    )
    for group, summary in worst_groups:
        md_lines.append(
            f"| {group} | {summary['record_count']} | {summary['exact_set_accuracy']:.6f} | {summary['hamming_loss']:.6f} |"
        )
    md_lines.extend(
        [
            "",
            "## Manuscript-safe note",
            "",
            "The refutation pack does not replace the main source-side champion. It is a negative-control block that shows how the frozen protocol behaves when labels are destroyed, when nuisance identity is probed, and when each bearing group is held out explicitly.",
            "",
        ]
    )
    _atomic_text(OUTPUT_MD, "\n".join(md_lines))
    return report


def main() -> None:
    print(json.dumps(build_report(), indent=2, ensure_ascii=False, default=_json_default))


if __name__ == "__main__":
    main()
