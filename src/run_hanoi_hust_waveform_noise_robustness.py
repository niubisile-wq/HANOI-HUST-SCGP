"""Evaluate HANOI HUST methods under waveform-level perturbation stress."""

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
from scipy.signal import lfilter
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from download_hanoi_hust import ARCHIVE
from electrical_fm.hanoi_hust import HanoiHustContract, contract_to_json
from electrical_fm.hanoi_hust_baselines import (
    Candidate,
    bearing_groups,
    component_targets,
    evaluate_candidate,
)
from electrical_fm.hanoi_hust_features import WINDOW_OFFSETS, record_feature_blocks
from electrical_fm.hanoi_hust_io import (
    parse_source_record,
    verify_source_authorization,
)
from electrical_fm.hanoi_hust_noace import evaluate_noace_candidate


ROOT = Path(__file__).resolve().parents[1]
SOURCE_FREEZE = ROOT / "research" / "HANOI_HUST_SOURCE_ACCESS_FREEZE.json"
PREACCESS_FREEZE = ROOT / "research" / "HANOI_HUST_V3_PREACCESS_FREEZE.json"
OUTPUT_JSON = ROOT / "results" / "analysis" / "hanoi_hust_waveform_noise_robustness.json"
OUTPUT_MD = ROOT / "results" / "analysis" / "hanoi_hust_waveform_noise_robustness.md"
RESULT_PAGE = (
    ROOT
    / "research"
    / "HANOI_HUST_20260727"
    / "HANOI_HUST_WAVEFORM_NOISE_ROBUSTNESS_正式结果页_20260728.md"
)

AWGN_SNR_LEVELS = [20, 10, 5, 0, -5, -10]
COLOR_ALPHA_LEVELS = [0.0, 0.25, 0.5, 0.75, 0.9]
IMPULSE_RATE_LEVELS = [0.0, 0.01, 0.02, 0.05, 0.1]
CLIP_LEVELS = [1.0, 0.9, 0.8, 0.7, 0.6]
RESAMPLE_LEVELS = [1.0, 0.9, 0.8, 0.75, 0.65]
DROPOUT_LEVELS = [0.0, 0.05, 0.1, 0.2, 0.3]
SHIFT_LEVELS = [0.0, 0.05, 0.1, 0.15, 0.2]


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


def _load_contracts() -> tuple[list[HanoiHustContract], dict[str, Any]]:
    source_freeze = verify_source_authorization(
        SOURCE_FREEZE,
        repo_root=ROOT,
    )
    preaccess = json.loads(PREACCESS_FREEZE.read_text(encoding="utf-8"))
    contracts = [
        HanoiHustContract(
            archive_index=row["archive_index"],
            path=row["path"],
            filename=row["filename"],
            uncompressed_bytes=row["uncompressed_bytes"],
            compressed_bytes=row["compressed_bytes"],
            crc32=int(row["crc32"], 16),
            compression_method=row["compression_method"],
            local_header_offset=row["local_header_offset"],
            state=row["state"],
            components=tuple(row["components"]),
            bearing_type=row["bearing_type"],
            bearing_id=row["bearing_id"],
            load_w=row["load_w"],
            access_role=row["access_role"],
        )
        for row in preaccess.get("file_contracts", [])
        if row.get("access_role") == "source"
    ]
    if len(contracts) != 57:
        raise RuntimeError("HANOI HUST source contract count changed")
    return contracts, source_freeze


def _load_source_records() -> list[dict[str, Any]]:
    contracts, source_freeze = _load_contracts()
    rows: list[dict[str, Any]] = []
    for contract in contracts:
        signal, schema = parse_source_record(
            ARCHIVE,
            contract,
            source_freeze,
        )
        rows.append(
            {
                "signal": signal,
                "schema": schema,
                "contract": contract,
            }
        )
    return rows


def _feature_view_from_waveform(signal: np.ndarray, *, offset: int) -> dict[str, np.ndarray]:
    features, _starts = record_feature_blocks(signal, offset=offset)
    return {
        name: np.asarray(values, dtype=np.float32).reshape(-1)
        for name, values in features.items()
    }


def _representation_views_from_waveform(signal: np.ndarray) -> dict[str, np.ndarray]:
    per_name: dict[str, list[np.ndarray]] = {
        "statistics": [],
        "fixed_log_power": [],
        "envelope_log_power": [],
    }
    for offset in WINDOW_OFFSETS:
        blocks = _feature_view_from_waveform(signal, offset=offset)
        for name in per_name:
            per_name[name].append(blocks[name])
    views = {
        name: np.concatenate(chunks).astype(np.float32, copy=False)
        for name, chunks in per_name.items()
    }
    views["all"] = np.concatenate(
        [views["statistics"], views["fixed_log_power"], views["envelope_log_power"]]
    ).astype(np.float32, copy=False)
    return views


def _stack_views(rows: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    view_names = ("statistics", "fixed_log_power", "envelope_log_power", "all")
    stacked: dict[str, list[np.ndarray]] = {name: [] for name in view_names}
    for row in rows:
        for name in view_names:
            stacked[name].append(np.asarray(row[name], dtype=np.float32))
    return {
        name: np.asarray(items, dtype=np.float32)
        for name, items in stacked.items()
    }


def _awgn(signal: np.ndarray, *, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    values = np.asarray(signal, dtype=np.float32)
    signal_rms = float(np.sqrt(np.mean(np.square(values), dtype=np.float64)))
    if signal_rms <= 0:
        return values.copy()
    noise = rng.normal(0.0, 1.0, size=values.shape).astype(np.float32)
    noise_rms = float(np.sqrt(np.mean(np.square(noise), dtype=np.float64)))
    if noise_rms <= 0:
        return values.copy()
    desired_noise_rms = signal_rms / np.sqrt(10.0 ** (snr_db / 10.0))
    return values + noise * np.float32(desired_noise_rms / noise_rms)


def _colored_noise(signal: np.ndarray, *, alpha: float, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    values = np.asarray(signal, dtype=np.float32)
    signal_rms = float(np.sqrt(np.mean(np.square(values), dtype=np.float64)))
    if signal_rms <= 0:
        return values.copy()
    white = rng.normal(0.0, 1.0, size=values.shape).astype(np.float32)
    filtered = lfilter([1.0], [1.0, -float(alpha)], white).astype(np.float32)
    filtered -= filtered.mean()
    noise_rms = float(np.sqrt(np.mean(np.square(filtered), dtype=np.float64)))
    if noise_rms <= 0:
        return values.copy()
    desired_noise_rms = signal_rms / np.sqrt(10.0 ** (snr_db / 10.0))
    return values + filtered * np.float32(desired_noise_rms / noise_rms)


def _impulsive_noise(signal: np.ndarray, *, rate: float, scale: float, rng: np.random.Generator) -> np.ndarray:
    values = np.asarray(signal, dtype=np.float32)
    if rate <= 0:
        return values.copy()
    count = max(1, int(round(values.size * rate)))
    indices = rng.choice(values.size, size=count, replace=False)
    spikes = rng.laplace(0.0, scale, size=count).astype(np.float32)
    stressed = values.copy()
    stressed[indices] = stressed[indices] + spikes
    return stressed


def _clipping(signal: np.ndarray, *, clip_ratio: float) -> np.ndarray:
    values = np.asarray(signal, dtype=np.float32)
    limit = float(np.quantile(np.abs(values), min(max(clip_ratio, 0.05), 1.0)))
    if limit <= 0:
        return values.copy()
    return np.clip(values, -limit, limit)


def _resample(signal: np.ndarray, *, factor: float) -> np.ndarray:
    values = np.asarray(signal, dtype=np.float32)
    if factor >= 1.0:
        return values.copy()
    length = values.size
    reduced = max(64, int(round(length * factor)))
    base = np.linspace(0.0, 1.0, length, dtype=np.float64)
    compressed = np.interp(
        np.linspace(0.0, 1.0, reduced, dtype=np.float64),
        base,
        values.astype(np.float64),
    )
    restored = np.interp(
        base,
        np.linspace(0.0, 1.0, reduced, dtype=np.float64),
        compressed,
    )
    return restored.astype(np.float32)


def _dropout(signal: np.ndarray, *, fraction: float, rng: np.random.Generator) -> np.ndarray:
    values = np.asarray(signal, dtype=np.float32)
    if fraction <= 0:
        return values.copy()
    length = values.size
    span = max(1, int(round(length * fraction)))
    start = int(rng.integers(0, max(1, length - span + 1)))
    stressed = values.copy()
    stressed[start : start + span] = 0.0
    return stressed


def _time_shift(signal: np.ndarray, *, fraction: float) -> np.ndarray:
    values = np.asarray(signal, dtype=np.float32)
    if fraction <= 0:
        return values.copy()
    shift = int(round(values.size * fraction))
    return np.roll(values, shift).astype(np.float32)


def _perturb_waveform(
    signal: np.ndarray,
    *,
    mode: str,
    severity: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if mode == "clean":
        return np.asarray(signal, dtype=np.float32).copy()
    if mode == "awgn":
        return _awgn(signal, snr_db=severity, rng=rng)
    if mode == "colored_noise":
        return _colored_noise(signal, alpha=severity, snr_db=5.0, rng=rng)
    if mode == "impulsive_noise":
        return _impulsive_noise(signal, rate=severity, scale=float(np.std(signal)) * 6.0, rng=rng)
    if mode == "clipping":
        return _clipping(signal, clip_ratio=severity)
    if mode == "resampling":
        return _resample(signal, factor=severity)
    if mode == "dropout":
        return _dropout(signal, fraction=severity, rng=rng)
    if mode == "time_shift":
        return _time_shift(signal, fraction=severity)
    raise ValueError(f"Unknown waveform perturbation mode: {mode}")


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


def _build_views_for_record(signal: np.ndarray) -> dict[str, np.ndarray]:
    return _representation_views_from_waveform(signal)


def build_waveform_noise_robustness(
    *,
    random_state: int,
    output: Path,
    markdown: Path,
) -> dict[str, Any]:
    metadata = json.loads((ROOT / "artifacts" / "hanoi_hust" / "source_features_metadata.json").read_text(encoding="utf-8"))
    if metadata.get("status") != "source_features_complete_compound_sealed":
        raise RuntimeError("HANOI HUST source cache is not complete")
    if _sha256(ARCHIVE) != metadata.get("archive", {}).get("sha256"):
        raise RuntimeError("HANOI HUST archive hash changed")

    rows = _load_source_records()
    states = np.asarray([row["contract"].state for row in rows], dtype=str)
    groups = bearing_groups(states, np.asarray([row["contract"].bearing_type for row in rows], dtype=np.int16))
    targets = component_targets(states)

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
            key="mlp_comparator",
            label="MLP comparator",
            kind="supervised",
            representation="statistics",
            candidate=_fixed_deep_champion(random_state),
        ),
    ]

    stress_specs: list[tuple[str, list[float]]] = [
        ("awgn", AWGN_SNR_LEVELS),
        ("colored_noise", COLOR_ALPHA_LEVELS),
        ("impulsive_noise", IMPULSE_RATE_LEVELS),
        ("clipping", CLIP_LEVELS),
        ("resampling", RESAMPLE_LEVELS),
        ("dropout", DROPOUT_LEVELS),
        ("time_shift", SHIFT_LEVELS),
    ]

    clean_views = {
        name: _stack_views([_build_views_for_record(row["signal"]) for row in rows])[name]
        for name in ("statistics", "fixed_log_power", "envelope_log_power", "all")
    }

    clean_results: dict[str, dict[str, Any]] = {}
    for spec in method_specs:
        base_view = clean_views[spec.representation]
        if spec.kind == "noace":
            clean_result = evaluate_noace_candidate(
                features=base_view,
                states=states,
                bearing_type=np.asarray([row["contract"].bearing_type for row in rows], dtype=np.int16),
                load_w=np.asarray([row["contract"].load_w for row in rows], dtype=np.int16),
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
        clean_results[spec.key] = clean_result

    baseline_path = ROOT / "results" / "development" / "hanoi_hust_source_baselines.json"
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        expected = baseline["selected"]
        observed = clean_results["source_champion"]
        for key in ("mean_component_auroc", "mean_component_balanced_accuracy", "exact_set_accuracy", "mean_brier_score", "hamming_loss"):
            if not np.isclose(float(observed[key]), float(expected[key]), atol=1e-12, rtol=0.0):
                raise RuntimeError(f"Clean waveform pipeline drifted for {key}")

    rows_out: list[dict[str, Any]] = []
    for spec in method_specs:
        clean_result = clean_results[spec.key]
        rows_out.append(
            {
                "method_key": spec.key,
                "method_label": spec.label,
                "perturbation_mode": "clean",
                "severity": None,
                "severity_label": "clean",
                "summary": _summarize(clean_result, clean_result),
            }
        )

    bearing_type_array = np.asarray([row["contract"].bearing_type for row in rows], dtype=np.int16)
    load_w_array = np.asarray([row["contract"].load_w for row in rows], dtype=np.int16)

    for mode_index, (mode, severities) in enumerate(stress_specs, start=1):
        for severity_index, severity in enumerate(severities):
            perturbed_rows: list[dict[str, np.ndarray]] = []
            for record_index, row in enumerate(rows):
                signal = row["signal"]
                stressed_signal = _perturb_waveform(
                    signal,
                    mode=mode,
                    severity=float(severity),
                    seed=random_state
                    + mode_index * 1_000
                    + severity_index * 100
                    + record_index,
                )
                perturbed_rows.append(_build_views_for_record(stressed_signal))
            stressed_views = _stack_views(perturbed_rows)
            for spec in method_specs:
                base_view = stressed_views[spec.representation]
                if spec.kind == "noace":
                    stressed_result = evaluate_noace_candidate(
                        features=base_view,
                        states=states,
                        bearing_type=bearing_type_array,
                        load_w=load_w_array,
                        groups=groups,
                        random_seed=random_state,
                        representation=spec.representation,
                    )
                else:
                    assert spec.candidate is not None
                    stressed_result = evaluate_candidate(
                        base_view,
                        targets,
                        groups,
                        spec.candidate,
                    )
                clean_result = clean_results[spec.key]
                rows_out.append(
                    {
                        "method_key": spec.key,
                        "method_label": spec.label,
                        "perturbation_mode": mode,
                        "severity": float(severity),
                        "severity_label": str(severity),
                        "summary": _summarize(clean_result, stressed_result),
                    }
                )

    report = {
        "stage": "hanoi_hust_waveform_noise_robustness",
        "schema_version": 1,
        "status": "completed",
        "archive": {
            "path": ARCHIVE.relative_to(ROOT).as_posix(),
            "sha256": _sha256(ARCHIVE),
        },
        "record_count": int(len(rows)),
        "bearing_count": int(len(np.unique(groups))),
        "methods": [
            {
                "key": spec.key,
                "label": spec.label,
                "kind": spec.kind,
                "representation": spec.representation,
                "hyperparameter": spec.candidate.hyperparameter if spec.candidate else {},
            }
            for spec in method_specs
        ],
        "stress_specs": [
            {"mode": mode, "levels": levels}
            for mode, levels in stress_specs
        ],
        "clean_results": {
            key: {
                k: v
                for k, v in result.items()
                if k not in {"probabilities", "predictions", "targets"}
            }
            for key, result in clean_results.items()
        },
        "rows": rows_out,
        "provenance": {
            "git_commit": _git("rev-parse", "HEAD"),
            "builder": "src/run_hanoi_hust_waveform_noise_robustness.py",
            "random_state": int(random_state),
            "source_records_opened": int(len(rows)),
            "compound_records_opened": 0,
        },
    }
    _atomic_json(output, report)

    md_lines = [
        "# HANOI HUST Waveform-Level Noise Robustness",
        "",
        "This result page replaces feature-space-only noise with waveform-level perturbation before feature extraction.",
        "",
        "## Clean baseline",
        "",
        "| Method | Mean AUROC | Mean CBA | Exact-set | Mean Brier | Hamming |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for spec in method_specs:
        clean = report["clean_results"][spec.key]
        md_lines.append(
            f"| {spec.label} | `{clean['mean_component_auroc']}` | `{clean['mean_component_balanced_accuracy']}` | `{clean['exact_set_accuracy']}` | `{clean['mean_brier_score']}` | `{clean['hamming_loss']}` |"
        )
    md_lines.extend(["", "## Waveform stress summary", ""])
    for spec in method_specs:
        md_lines.extend(
            [
                f"### {spec.label}",
                "",
                "| Mode | Severity | Mean AUROC | Exact-set | Mean Brier | Delta AUROC vs clean |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        clean = report["clean_results"][spec.key]
        for row in report["rows"]:
            if row["method_key"] != spec.key or row["perturbation_mode"] == "clean":
                continue
            summary = row["summary"]
            md_lines.append(
                f"| {row['perturbation_mode']} | {row['severity_label']} | `{summary['mean_component_auroc']}` | `{summary['exact_set_accuracy']}` | `{summary['mean_brier_score']}` | `{summary['delta_mean_component_auroc']}` |"
            )
        md_lines.append("")
    md_lines.extend(
        [
            "## Manuscript-safe note",
            "",
            "This page is the waveform-level pressure test that the plan asked for. It should be cited as a robustness block, not as a new main method result.",
            "",
        ]
    )
    _atomic_text(markdown, "\n".join(md_lines))
    _atomic_text(RESULT_PAGE, "\n".join(md_lines))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-state", type=int, default=20_260_728)
    parser.add_argument("--output", type=Path, default=OUTPUT_JSON)
    parser.add_argument("--markdown", type=Path, default=OUTPUT_MD)
    args = parser.parse_args()
    print(json.dumps(build_waveform_noise_robustness(
        random_state=args.random_state,
        output=args.output,
        markdown=args.markdown,
    ), indent=2, ensure_ascii=False, default=_jsonable))


if __name__ == "__main__":
    main()
