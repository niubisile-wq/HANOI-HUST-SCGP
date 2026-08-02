"""Statistical hardening audit for the frozen HANOI HUST source protocol.

This report consolidates the plan-mandated statistical layer:

- physical-bearing cluster bootstrap confidence intervals;
- paired comparisons against frozen comparator tracks;
- Holm correction across the paired tests;
- effect size reporting;
- worst-group reporting.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_JSON = ROOT / "results" / "analysis" / "hanoi_hust_statistics_hardening.json"
OUTPUT_MD = ROOT / "results" / "analysis" / "hanoi_hust_statistics_hardening.md"

CHAMPION = ROOT / "results" / "development" / "hanoi_hust_source_selected_predictions.npz"
COMPARATORS = {
    "noace_classical": ROOT / "results" / "analysis" / "hanoi_hust_noace_classical_selected_predictions.npz",
    "noace_physics": ROOT / "results" / "analysis" / "hanoi_hust_noace_physics_selected_predictions.npz",
    "noace_deep": ROOT / "results" / "analysis" / "hanoi_hust_noace_deep_selected_predictions.npz",
}

BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260728


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


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    arrays = np.load(path, allow_pickle=True)
    required = {"predictions", "targets"}
    missing = required.difference(arrays.files)
    if missing:
        raise RuntimeError(f"{path.name} is missing required arrays: {sorted(missing)}")
    return {name: arrays[name] for name in arrays.files}


def _group_array(arrays: dict[str, np.ndarray]) -> np.ndarray:
    if "groups" in arrays:
        return arrays["groups"]
    if "group_ids" in arrays:
        return arrays["group_ids"]
    raise RuntimeError("Prediction cache does not contain groups or group_ids")


def _exact_set_correct(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    return np.all(predictions == targets, axis=1).astype(np.int8)


def _subset_exact_set_correct(
    predictions: np.ndarray,
    targets: np.ndarray,
    keep_components: list[int],
) -> np.ndarray:
    return np.all(predictions[:, keep_components] == targets[:, keep_components], axis=1).astype(np.int8)


def _group_index(groups: np.ndarray) -> tuple[np.ndarray, list[str]]:
    unique = np.unique(groups.astype(str))
    group_to_idx = {group: idx for idx, group in enumerate(unique)}
    index = np.array([group_to_idx[str(group)] for group in groups], dtype=int)
    return index, list(unique)


def _ordered_unique(values: np.ndarray) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values.astype(str):
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _row_group_labels(arrays: dict[str, np.ndarray]) -> np.ndarray:
    predictions = arrays["predictions"]
    if "groups" in arrays and len(arrays["groups"]) == predictions.shape[0]:
        return arrays["groups"].astype(str)
    if "group_ids" in arrays:
        ordered = _ordered_unique(arrays["group_ids"])
        if len(ordered) == predictions.shape[0]:
            return np.asarray(ordered, dtype=str)
    if "groups" in arrays:
        ordered = _ordered_unique(arrays["groups"])
        if len(ordered) == predictions.shape[0]:
            return np.asarray(ordered, dtype=str)
    raise RuntimeError("Unable to infer row-level group labels for prediction cache")


def _group_means(
    values: np.ndarray,
    groups: np.ndarray,
) -> dict[str, float]:
    means: dict[str, float] = {}
    for group in np.unique(groups.astype(str)):
        mask = groups.astype(str) == group
        means[group] = float(values[mask].mean())
    return means


def _cluster_bootstrap_ci(
    values_by_group: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n_groups = values_by_group.shape[0]
    draws = np.empty(replicates, dtype=float)
    for i in range(replicates):
        sampled = rng.integers(0, n_groups, size=n_groups)
        draws[i] = float(values_by_group[sampled].mean())
    return {
        "mean": float(values_by_group.mean()),
        "lower": float(np.quantile(draws, 0.025)),
        "upper": float(np.quantile(draws, 0.975)),
    }


def _holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    m = len(ordered)
    for i, (name, p_value) in enumerate(ordered):
        scaled = (m - i) * p_value
        running = max(running, scaled)
        adjusted[name] = min(1.0, running)
    return adjusted


def _paired_stats(champion_groups: np.ndarray, comparator_groups: np.ndarray) -> dict[str, Any]:
    deltas = champion_groups - comparator_groups
    wilcoxon_result = wilcoxon(deltas, zero_method="wilcox", alternative="two-sided", mode="auto")
    std = float(deltas.std(ddof=1)) if deltas.size > 1 else float("nan")
    cohen_dz = float(deltas.mean() / std) if std not in (0.0, float("nan")) and np.isfinite(std) and std > 0 else float("nan")
    bootstrap = _cluster_bootstrap_ci(
        deltas,
        replicates=BOOTSTRAP_REPLICATES,
        seed=BOOTSTRAP_SEED,
    )
    return {
        "mean_difference": float(deltas.mean()),
        "mean_difference_pp": float(deltas.mean() * 100.0),
        "std_difference": std,
        "cohen_dz": cohen_dz,
        "wilcoxon_statistic": float(wilcoxon_result.statistic),
        "wilcoxon_pvalue": float(wilcoxon_result.pvalue),
        "bootstrap_95_ci": {
            "lower": bootstrap["lower"],
            "upper": bootstrap["upper"],
        },
    }


def build_report() -> dict[str, Any]:
    champion_arrays = _load_npz(CHAMPION)
    comparator_arrays = {name: _load_npz(path) for name, path in COMPARATORS.items()}

    champion_groups = _row_group_labels(champion_arrays)
    champion_correct = _exact_set_correct(champion_arrays["predictions"], champion_arrays["targets"])
    champion_group_means_map = _group_means(champion_correct, champion_groups)
    group_labels = sorted(champion_group_means_map)
    champion_group_means = np.array([champion_group_means_map[group] for group in group_labels], dtype=float)

    component_names = ["inner", "outer", "ball"]
    leave_one_component_out: dict[str, Any] = {}
    for idx, dropped in enumerate(component_names):
        keep = [i for i in range(3) if i != idx]
        subset_correct = _subset_exact_set_correct(
            champion_arrays["predictions"],
            champion_arrays["targets"],
            keep,
        )
        subset_group_map = _group_means(subset_correct, champion_groups)
        subset_group_means = np.array([subset_group_map[group] for group in group_labels], dtype=float)
        subset_bootstrap = _cluster_bootstrap_ci(
            subset_group_means,
            replicates=BOOTSTRAP_REPLICATES,
            seed=BOOTSTRAP_SEED + idx + 1,
        )
        leave_one_component_out[dropped] = {
            "keep_components": [component_names[i] for i in keep],
            "exact_set_accuracy": float(subset_correct.mean()),
            "delta_vs_full_pp": float((subset_correct.mean() - champion_correct.mean()) * 100.0),
            "group_bootstrap_95_ci": subset_bootstrap,
        }

    group_audit = [
        {
            "group": group,
            "record_count": int((champion_groups == group).sum()),
            "exact_set_accuracy": float(champion_group_means[idx]),
        }
        for idx, group in enumerate(group_labels)
    ]
    group_audit_sorted = sorted(group_audit, key=lambda row: (row["exact_set_accuracy"], row["group"]))

    champion_bootstrap = _cluster_bootstrap_ci(
        champion_group_means,
        replicates=BOOTSTRAP_REPLICATES,
        seed=BOOTSTRAP_SEED,
    )

    comparisons: dict[str, Any] = {}
    raw_pvalues: dict[str, float] = {}
    for name, arrays in comparator_arrays.items():
        groups = _row_group_labels(arrays)
        correct = _exact_set_correct(arrays["predictions"], arrays["targets"])
        group_means_map = _group_means(correct, groups)
        missing_groups = set(group_labels).difference(group_means_map)
        if missing_groups:
            raise RuntimeError(f"{name} is missing groups: {sorted(missing_groups)}")
        group_means = np.array([group_means_map[group] for group in group_labels], dtype=float)
        stats = _paired_stats(champion_group_means, group_means)
        comparisons[name] = {
            "exact_set_accuracy": float(correct.mean()),
            "group_mean_exact_set_accuracy": float(group_means.mean()),
            "group_level": stats,
        }
        raw_pvalues[name] = stats["wilcoxon_pvalue"]

    adjusted = _holm_adjust(raw_pvalues)
    for name, value in adjusted.items():
        comparisons[name]["group_level"]["holm_adjusted_pvalue"] = float(value)

    report = {
        "stage": "hanoi_hust_statistics_hardening",
        "schema_version": 1,
        "status": "completed",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "prediction_cache": {
            "path": CHAMPION.relative_to(ROOT).as_posix(),
            "sha256": _sha256(CHAMPION),
        },
        "champion": {
            "family": "logistic_l2",
            "representation": "envelope_log_power",
            "exact_set_accuracy": float(champion_correct.mean()),
            "group_bootstrap_95_ci": champion_bootstrap,
            "worst_groups": group_audit_sorted[:5],
        },
        "leave_one_component_out": leave_one_component_out,
        "comparators": comparisons,
        "provenance": {
            "git_commit": _git("rev-parse", "HEAD"),
            "builder": "src/analyze_hanoi_hust_statistics_hardening.py",
        },
    }
    _atomic_json(OUTPUT_JSON, report)

    lines = [
        "# HANOI HUST statistical hardening",
        "",
        f"- Bootstrap replicates: `{BOOTSTRAP_REPLICATES}`",
        f"- Bootstrap seed: `{BOOTSTRAP_SEED}`",
        f"- Champion exact-set accuracy: `{report['champion']['exact_set_accuracy']:.6f}`",
        f"- Champion group bootstrap 95% CI: [{report['champion']['group_bootstrap_95_ci']['lower']:.6f}, {report['champion']['group_bootstrap_95_ci']['upper']:.6f}]",
        "",
        "## Leave-one-component-out",
        "",
        "| Dropped component | Kept components | Exact-set | Delta vs full (pp) | 95% CI (exact-set) |",
        "|---|---|---:|---:|---:|",
    ]
    for dropped, values in leave_one_component_out.items():
        ci = values["group_bootstrap_95_ci"]
        lines.append(
            "| {dropped} | {kept} | {exact_set_accuracy:.6f} | {delta_vs_full_pp:.3f} | [{lower:.6f}, {upper:.6f}] |".format(
                dropped=dropped,
                kept=", ".join(values["keep_components"]),
                exact_set_accuracy=values["exact_set_accuracy"],
                delta_vs_full_pp=values["delta_vs_full_pp"],
                lower=ci["lower"],
                upper=ci["upper"],
            )
        )
    lines.extend(
        [
            "## Worst groups",
            "",
            "| Group | Records | Exact-set accuracy |",
            "|---|---:|---:|",
        ]
    )
    for row in group_audit_sorted[:5]:
        lines.append(
            f"| {row['group']} | {row['record_count']} | {row['exact_set_accuracy']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Paired comparisons",
            "",
            "| Comparator | Exact-set | Mean diff (pp) | Wilcoxon p | Holm p | Cohen's d_z | 95% CI (diff) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, stats in comparisons.items():
        lvl = stats["group_level"]
        ci = lvl["bootstrap_95_ci"]
        lines.append(
            "| {name} | {exact_set_accuracy:.6f} | {mean_difference_pp:.3f} | {wilcoxon_pvalue:.6g} | {holm_adjusted_pvalue:.6g} | {cohen_dz:.4f} | [{lower:.6f}, {upper:.6f}] |".format(
                name=name,
                exact_set_accuracy=stats["exact_set_accuracy"],
                mean_difference_pp=lvl["mean_difference_pp"],
                wilcoxon_pvalue=lvl["wilcoxon_pvalue"],
                holm_adjusted_pvalue=lvl["holm_adjusted_pvalue"],
                cohen_dz=lvl["cohen_dz"],
                lower=ci["lower"],
                upper=ci["upper"],
            )
        )
    _atomic_text(OUTPUT_MD, "\n".join(lines))
    return report


def main() -> None:
    print(json.dumps(build_report(), indent=2, ensure_ascii=False, default=_json_default))


if __name__ == "__main__":
    main()
