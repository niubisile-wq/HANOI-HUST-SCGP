"""Run the sealed HANOI HUST compound confirmation on frozen source models."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from download_hanoi_hust import ARCHIVE
from electrical_fm.hanoi_hust import HanoiHustContract, contract_to_json
from electrical_fm.hanoi_hust_baselines import bearing_groups, representation_views
from electrical_fm.hanoi_hust_features import WINDOW_OFFSETS, record_feature_blocks
from electrical_fm.hanoi_hust_io import (
    parse_compound_record,
    parse_source_record,
)
from electrical_fm.hanoi_hust_noace import (
    NOACEFit,
    _fit_nuisance,
    _predict_nuisance,
    _prepare_view,
    _select_view,
    _subset_states,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_FREEZE = ROOT / "research" / "HANOI_HUST_SOURCE_ACCESS_FREEZE.json"
SOURCE_CACHE = ROOT / "artifacts" / "hanoi_hust" / "source_features.npz"
SOURCE_METADATA = ROOT / "artifacts" / "hanoi_hust" / "source_features_metadata.json"
SOURCE_BASELINES = ROOT / "results" / "development" / "hanoi_hust_source_baselines.json"
SOURCE_NOACE = ROOT / "results" / "analysis" / "hanoi_hust_noace_classical.json"
SOURCE_NOACE_DEEP = ROOT / "results" / "analysis" / "hanoi_hust_noace_deep.json"
SOURCE_NOACE_PHYSICS = ROOT / "results" / "analysis" / "hanoi_hust_noace_physics.json"
SOURCE_REGISTRY = ROOT / "results" / "audits" / "hanoi_hust_source_registry.json"
PREACCESS_FREEZE = ROOT / "research" / "HANOI_HUST_V3_PREACCESS_FREEZE.json"
ARCHIVE_FREEZE = ROOT / "research" / "HANOI_HUST_V3_ARCHIVE_FREEZE.json"
OUTPUT_JSON = ROOT / "results" / "confirmation" / "hanoi_hust_compound_confirmation.json"
OUTPUT_MD = ROOT / "results" / "confirmation" / "hanoi_hust_compound_confirmation.md"
COMPOUND_CACHE = ROOT / "artifacts" / "hanoi_hust" / "compound_confirmation_features.npz"
COMPOUND_METADATA = ROOT / "artifacts" / "hanoi_hust" / "compound_confirmation_features_metadata.json"


@dataclass(frozen=True)
class ConfirmedModel:
    name: str
    family: str
    representation: str
    hyperparameter: dict[str, Any]
    probabilities: np.ndarray
    predictions: np.ndarray
    record_metrics: dict[str, Any]
    bearing_metrics: dict[str, Any]


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
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(staging, path)


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix(".writing")
    with staging.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(staging, path)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported Hanoi HUST JSON value: {type(value).__name__}")


def _load_freeze_contracts() -> tuple[dict[str, Any], tuple[HanoiHustContract, ...]]:
    preaccess = json.loads(PREACCESS_FREEZE.read_text(encoding="utf-8"))
    contracts = tuple(
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
        for row in preaccess["file_contracts"]
    )
    return preaccess, contracts


def _load_source_cache() -> dict[str, np.ndarray]:
    metadata = json.loads(SOURCE_METADATA.read_text(encoding="utf-8"))
    if metadata.get("status") != "source_features_complete_compound_sealed":
        raise RuntimeError("HANOI HUST source cache is not complete")
    if _sha256(SOURCE_CACHE) != metadata.get("cache", {}).get("sha256"):
        raise RuntimeError("HANOI HUST source cache hash changed")
    with np.load(SOURCE_CACHE, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def _load_compound_cache() -> dict[str, np.ndarray]:
    if COMPOUND_CACHE.is_file() and COMPOUND_METADATA.is_file():
        metadata = json.loads(COMPOUND_METADATA.read_text(encoding="utf-8"))
        if metadata.get("status") == "compound_confirmation_features_complete":
            if _sha256(COMPOUND_CACHE) != metadata.get("cache", {}).get("sha256"):
                raise RuntimeError("HANOI HUST compound cache hash changed")
            with np.load(COMPOUND_CACHE, allow_pickle=False) as payload:
                return {name: np.asarray(payload[name]) for name in payload.files}
    preaccess, contracts = _load_freeze_contracts()
    source_freeze = json.loads(SOURCE_FREEZE.read_text(encoding="utf-8"))
    records = [
        contract for contract in contracts if contract.access_role == "compound"
    ]
    rows: dict[str, list[list[np.ndarray]]] = {
        "statistics": [],
        "fixed_log_power": [],
        "envelope_log_power": [],
    }
    schemas: list[dict[str, Any]] = []
    window_starts: list[list[list[int]]] = []
    for contract in records:
        signal, schema = parse_compound_record(ARCHIVE, contract, source_freeze)
        record_views = {name: [] for name in rows}
        starts_by_offset = []
        for offset in WINDOW_OFFSETS:
            features, starts = record_feature_blocks(signal, offset=offset)
            for name, values in features.items():
                record_views[name].append(values)
            starts_by_offset.append(starts.tolist())
        for name in rows:
            rows[name].append(record_views[name])
        schemas.append(schema)
        window_starts.append(starts_by_offset)
    arrays = {
        name: np.asarray(values, dtype=np.float32)
        for name, values in rows.items()
    }
    labels = {
        "contract_index": np.asarray(
            [row.archive_index for row in records], dtype=np.int16
        ),
        "bearing_type": np.asarray([row.bearing_type for row in records], dtype=np.int16),
        "load_w": np.asarray([row.load_w for row in records], dtype=np.int16),
        "state": np.asarray([row.state for row in records]),
        "components": np.asarray(["+".join(row.components) for row in records]),
        "bearing_id": np.asarray([row.bearing_id for row in records]),
    }
    _atomic_npz(COMPOUND_CACHE, {**arrays, **labels})
    payload = {
        "stage": "hanoi_hust_compound_confirmation_features",
        "schema_version": 1,
        "status": "compound_confirmation_features_complete",
        "source_freeze": {
            "path": SOURCE_FREEZE.relative_to(ROOT).as_posix(),
            "sha256": _sha256(SOURCE_FREEZE),
        },
        "preaccess_freeze": {
            "path": PREACCESS_FREEZE.relative_to(ROOT).as_posix(),
            "sha256": _sha256(PREACCESS_FREEZE),
        },
        "archive_freeze": {
            "path": ARCHIVE_FREEZE.relative_to(ROOT).as_posix(),
            "sha256": _sha256(ARCHIVE_FREEZE),
        },
        "record_count": len(records),
        "contracts": [contract_to_json(row) for row in records],
        "window_offsets": list(WINDOW_OFFSETS),
        "window_starts": window_starts,
        "block_shapes": {name: list(values.shape) for name, values in arrays.items()},
        "parser_schemas": schemas,
        "cache": {
            "path": COMPOUND_CACHE.relative_to(ROOT).as_posix(),
            "sha256": _sha256(COMPOUND_CACHE),
        },
        "information_boundary": {
            "source_numeric_files_opened": 0,
            "compound_numeric_files_opened": len(records),
            "transient_numeric_files_opened": 0,
            "window_level_independent_samples": 0,
        },
        "provenance": {
            "git_commit": _git("rev-parse", "HEAD"),
            "builder": "src/run_hanoi_hust_compound_confirmation.py",
        },
    }
    _atomic_json(COMPOUND_METADATA, payload)
    return {**arrays, **labels}


def _component_targets_from_components(components: np.ndarray) -> np.ndarray:
    values = np.asarray(components, dtype=str)
    targets = np.zeros((len(values), 3), dtype=np.int8)
    for index, item in enumerate(values):
        parts = set(str(item).split("+"))
        targets[index, 0] = int("inner" in parts)
        targets[index, 1] = int("outer" in parts)
        targets[index, 2] = int("ball" in parts)
    return targets


def _component_metrics(
    truth: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, Any]:
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        f1_score,
        roc_auc_score,
    )

    truth = np.asarray(truth, dtype=np.int8)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.int8)
    balanced = [
        float(balanced_accuracy_score(truth[:, idx], predictions[:, idx]))
        for idx in range(3)
    ]
    auroc = [
        float(roc_auc_score(truth[:, idx], probabilities[:, idx]))
        for idx in range(3)
    ]
    aupr = [
        float(average_precision_score(truth[:, idx], probabilities[:, idx]))
        for idx in range(3)
    ]
    brier = [
        float(brier_score_loss(truth[:, idx], probabilities[:, idx]))
        for idx in range(3)
    ]
    macro_f1 = [
        float(
            f1_score(
                truth[:, idx],
                predictions[:, idx],
                average="macro",
                labels=(0, 1),
                zero_division=0,
            )
        )
        for idx in range(3)
    ]
    exact = np.all(truth == predictions, axis=1)
    return {
        "balanced_accuracy_by_component": dict(
            zip(("inner", "outer", "ball"), balanced, strict=True)
        ),
        "mean_component_balanced_accuracy": float(np.mean(balanced)),
        "auroc_by_component": dict(
            zip(("inner", "outer", "ball"), auroc, strict=True)
        ),
        "mean_component_auroc": float(np.mean(auroc)),
        "aupr_by_component": dict(
            zip(("inner", "outer", "ball"), aupr, strict=True)
        ),
        "mean_component_aupr": float(np.mean(aupr)),
        "brier_by_component": dict(
            zip(("inner", "outer", "ball"), brier, strict=True)
        ),
        "mean_brier_score": float(np.mean(brier)),
        "macro_f1_by_component": dict(
            zip(("inner", "outer", "ball"), macro_f1, strict=True)
        ),
        "mean_component_macro_f1": float(np.mean(macro_f1)),
        "exact_set_accuracy": float(np.mean(exact)),
        "hamming_loss": float(np.mean(truth != predictions)),
    }


def _aggregate_by_bearing(
    *,
    group_ids: np.ndarray,
    truth: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    groups = np.asarray(group_ids, dtype=str)
    unique_groups, inverse = np.unique(groups, return_inverse=True)
    agg_truth = []
    agg_probs = []
    for group_index in range(len(unique_groups)):
        mask = inverse == group_index
        agg_truth.append(np.asarray(truth[mask][0], dtype=np.int8))
        agg_probs.append(np.asarray(probabilities[mask].mean(axis=0), dtype=np.float64))
    agg_truth_arr = np.asarray(agg_truth, dtype=np.int8)
    agg_probs_arr = np.asarray(agg_probs, dtype=np.float64)
    agg_pred_arr = (agg_probs_arr >= 0.5).astype(np.int8)
    return unique_groups, agg_truth_arr, agg_probs_arr, agg_pred_arr


def _fit_logistic(source_views: dict[str, np.ndarray], source_targets: np.ndarray):
    features = representation_views(source_views)["envelope_log_power"]
    models = []
    for head in range(source_targets.shape[1]):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=10.0,
                class_weight="balanced",
                max_iter=5_000,
                random_state=20_260_727,
            ),
        )
        model.fit(features, source_targets[:, head])
        models.append(model)
    return tuple(models)


def _fit_mlp(source_views: dict[str, np.ndarray], source_targets: np.ndarray):
    features = representation_views(source_views)["statistics"]
    models = []
    for head in range(source_targets.shape[1]):
        model = make_pipeline(
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
                random_state=20_260_727,
            ),
        )
        model.fit(features, source_targets[:, head])
        models.append(model)
    return tuple(models)


def _fit_noace(
    source_arrays: dict[str, np.ndarray],
) -> tuple[NOACEFit, tuple[int, ...], tuple[int, ...], np.ndarray, np.ndarray]:
    values = _prepare_view(_select_view(source_arrays, representation="all"))
    source_states = source_arrays["state"].astype(str)
    residual_train, coeff, bearing_categories, load_categories = _fit_nuisance(
        values,
        bearing_type=source_arrays["bearing_type"],
        load_w=source_arrays["load_w"],
    )
    healthy, inner, outer, ball = _subset_states(source_states)
    healthy_mean = residual_train[healthy].mean(axis=0)
    inner_mean = residual_train[inner].mean(axis=0)
    outer_mean = residual_train[outer].mean(axis=0)
    ball_mean = residual_train[ball].mean(axis=0)
    singleton_effects = [
        healthy_mean,
        inner_mean - healthy_mean,
        outer_mean - healthy_mean,
        ball_mean - healthy_mean,
    ]
    subset_signatures = list(__import__("itertools").product([False, True], repeat=3))
    subset_means = np.empty(
        (len(subset_signatures), residual_train.shape[1]),
        dtype=np.float64,
    )
    for idx, signature in enumerate(subset_signatures):
        prototype = healthy_mean.copy()
        for comp_index, active in enumerate(signature):
            if active:
                prototype += singleton_effects[comp_index + 1]
        subset_means[idx] = prototype
    train_variance = np.var(residual_train, axis=0)
    train_variance = np.where(
        train_variance > 0,
        train_variance,
        np.finfo(np.float64).eps,
    )
    subset_inv_variance = np.full(
        (len(subset_signatures), residual_train.shape[1]),
        1.0 / train_variance,
        dtype=np.float64,
    )
    fit = NOACEFit(
        subset_signatures=subset_signatures,
        subset_means=subset_means,
        subset_inv_variance=subset_inv_variance,
        temperature=1.0,
    )
    return fit, bearing_categories, load_categories, coeff, residual_train


def _predict_noace(
    fit: NOACEFit,
    *,
    source_arrays: dict[str, np.ndarray],
    compound_arrays: dict[str, np.ndarray],
    coeff: np.ndarray,
    bearing_categories: tuple[int, ...],
    load_categories: tuple[int, ...],
) -> np.ndarray:
    compound_values = _prepare_view(_select_view(compound_arrays, representation="all"))
    residual_compound = compound_values - _predict_nuisance(
        coeff,
        bearing_categories=bearing_categories,
        load_categories=load_categories,
        bearing_test=compound_arrays["bearing_type"],
        load_test=compound_arrays["load_w"],
        x_test_count=len(compound_values),
    )
    return fit.component_probabilities(residual_compound)


def _predict_multilabel(model: Any, features: np.ndarray) -> np.ndarray:
    probabilities = []
    for head in range(3):
        head_model = model[head]
        probabilities.append(head_model.predict_proba(features)[:, 1])
    return np.column_stack(probabilities)


def _fit_and_predict_source_models(
    source_arrays: dict[str, np.ndarray],
    compound_arrays: dict[str, np.ndarray],
) -> list[ConfirmedModel]:
    source_targets = _component_targets_from_components(
        source_arrays["components"]
    )
    compound_targets = _component_targets_from_components(
        compound_arrays["components"]
    )
    compound_group_ids = np.asarray(compound_arrays["bearing_id"], dtype=str)
    compound_views = representation_views(compound_arrays)

    confirmed: list[ConfirmedModel] = []

    logistic = _fit_logistic(source_arrays, source_targets)
    logistic_probs = _predict_multilabel(logistic, compound_views["envelope_log_power"])
    logistic_pred = (logistic_probs >= 0.5).astype(np.int8)
    _, logistic_bear_truth, logistic_bear_probs, logistic_bear_pred = _aggregate_by_bearing(
        group_ids=compound_group_ids,
        truth=compound_targets,
        probabilities=logistic_probs,
    )
    confirmed.append(
        ConfirmedModel(
            name="source_logistic_l2",
            family="logistic_l2",
            representation="envelope_log_power",
            hyperparameter={"C": 10.0, "class_weight": "balanced"},
            probabilities=logistic_probs,
            predictions=logistic_pred,
            record_metrics=_component_metrics(
                compound_targets,
                logistic_probs,
                logistic_pred,
            ),
            bearing_metrics=_component_metrics(
                logistic_bear_truth,
                logistic_bear_probs,
                logistic_bear_pred,
            ),
        )
    )

    mlp = _fit_mlp(source_arrays, source_targets)
    mlp_probs = _predict_multilabel(mlp, compound_views["statistics"])
    mlp_pred = (mlp_probs >= 0.5).astype(np.int8)
    _, mlp_bear_truth, mlp_bear_probs, mlp_bear_pred = _aggregate_by_bearing(
        group_ids=compound_group_ids,
        truth=compound_targets,
        probabilities=mlp_probs,
    )
    confirmed.append(
        ConfirmedModel(
            name="source_mlp_deep",
            family="mlp_deep",
            representation="statistics",
            hyperparameter={
                "hidden_layer_sizes": [64, 64, 32],
                "alpha": 1e-4,
                "learning_rate_init": 1e-3,
                "early_stopping": True,
            },
            probabilities=mlp_probs,
            predictions=mlp_pred,
            record_metrics=_component_metrics(
                compound_targets,
                mlp_probs,
                mlp_pred,
            ),
            bearing_metrics=_component_metrics(
                mlp_bear_truth,
                mlp_bear_probs,
                mlp_bear_pred,
            ),
        )
    )

    noace_fit, bearing_categories, load_categories, coeff, _ = _fit_noace(
        source_arrays
    )
    noace_probs = _predict_noace(
        noace_fit,
        source_arrays=source_arrays,
        compound_arrays=compound_arrays,
        coeff=coeff,
        bearing_categories=bearing_categories,
        load_categories=load_categories,
    )
    noace_pred = (noace_probs >= 0.5).astype(np.int8)
    _, noace_bear_truth, noace_bear_probs, noace_bear_pred = _aggregate_by_bearing(
        group_ids=compound_group_ids,
        truth=compound_targets,
        probabilities=noace_probs,
    )
    confirmed.append(
        ConfirmedModel(
            name="noace_classical",
            family="noace_classical",
            representation="all",
            hyperparameter={
                "nuisance_model": "linear_residual",
                "subset_prototypes": "additive_health_singleton",
                "variance": "diagonal",
                "temperature": 1.0,
            },
            probabilities=noace_probs,
            predictions=noace_pred,
            record_metrics=_component_metrics(
                compound_targets,
                noace_probs,
                noace_pred,
            ),
            bearing_metrics=_component_metrics(
                noace_bear_truth,
                noace_bear_probs,
                noace_bear_pred,
            ),
        )
    )

    confirmed.sort(
        key=lambda row: (
            -row.bearing_metrics["mean_component_auroc"],
            -row.bearing_metrics["exact_set_accuracy"],
            row.bearing_metrics["mean_brier_score"],
            row.family,
            row.representation,
        )
    )
    return confirmed


def _render_markdown(payload: dict[str, Any]) -> str:
    main = payload["primary_confirmation"]
    rows = payload["candidate_table"]
    lines = [
        "# HANOI HUST sealed compound confirmation",
        "",
        f"Status: `{payload['status']}`",
        "",
        "## Main result",
        "",
        f"- Model: `{main['name']}`",
        f"- Family: `{main['family']}`",
        f"- Representation: `{main['representation']}`",
        f"- Mean component AUROC: `{main['bearing_metrics']['mean_component_auroc']}`",
        f"- Exact-set accuracy: `{main['bearing_metrics']['exact_set_accuracy']}`",
        f"- Mean Brier score: `{main['bearing_metrics']['mean_brier_score']}`",
        "",
        "## Comparators",
        "",
        "| Model | Mean AUROC | Exact-set | Mean Brier |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        metrics = row["bearing_metrics"]
        lines.append(
            f"| `{row['name']}` | `{metrics['mean_component_auroc']}` | "
            f"`{metrics['exact_set_accuracy']}` | `{metrics['mean_brier_score']}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The compound partition is now evaluated directly, not inferred from source-side boundaries.",
            "- The primary confirmation result is fixed by the frozen source choice and scored on the 14 sealed compound bearings.",
            "- Comparators are fitted only on the frozen source partition and never tuned on compound labels.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_confirmation() -> dict[str, Any]:
    preaccess, contracts = _load_freeze_contracts()
    source_freeze = json.loads(SOURCE_FREEZE.read_text(encoding="utf-8"))
    archive_freeze = json.loads(ARCHIVE_FREEZE.read_text(encoding="utf-8"))
    if (
        source_freeze.get("status") != "source_numeric_access_frozen"
        or source_freeze.get("authorization", {}).get("source_numeric_parse") is not True
        or archive_freeze.get("status")
        != "opaque_archive_frozen_source_access_not_yet_authorized"
    ):
        raise RuntimeError("HANOI HUST source-side freeze prerequisites failed")

    source_arrays = _load_source_cache()
    compound_arrays = _load_compound_cache()
    source_targets = _component_targets_from_components(source_arrays["components"])
    compound_targets = _component_targets_from_components(
        compound_arrays["components"]
    )

    source_group_ids = bearing_groups(
        source_arrays["state"].astype(str),
        source_arrays["bearing_type"],
    )
    compound_group_ids = np.asarray(compound_arrays["bearing_id"], dtype=str)

    confirmed = _fit_and_predict_source_models(source_arrays, compound_arrays)
    primary = next(model for model in confirmed if model.name == "noace_classical")

    _atomic_npz(
        COMPOUND_CACHE.with_name("compound_confirmation_selected_predictions.npz"),
        {
            "probabilities": primary.probabilities.astype(np.float32),
            "predictions": primary.predictions.astype(np.int8),
            "targets": compound_targets.astype(np.int8),
            "bearing_id": compound_group_ids.astype("U"),
            "bearing_type": compound_arrays["bearing_type"].astype(np.int16),
            "load_w": compound_arrays["load_w"].astype(np.int16),
            "contract_index": compound_arrays["contract_index"].astype(np.int16),
        },
    )
    # Preserve every sealed model's record-level predictions for bearing-level
    # uncertainty audits; no labels are used beyond the frozen confirmation step.
    for model in confirmed:
        safe_name = model.name.replace("-", "_")
        _atomic_npz(
            COMPOUND_CACHE.with_name(f"compound_confirmation_{safe_name}_predictions.npz"),
            {
                "probabilities": model.probabilities.astype(np.float32),
                "predictions": model.predictions.astype(np.int8),
                "targets": compound_targets.astype(np.int8),
                "bearing_id": compound_group_ids.astype("U"),
                "contract_index": compound_arrays["contract_index"].astype(np.int16),
            },
        )

    payload = {
        "stage": "hanoi_hust_compound_confirmation",
        "schema_version": 1,
        "status": "completed",
        "source_freeze": {
            "path": SOURCE_FREEZE.relative_to(ROOT).as_posix(),
            "sha256": _sha256(SOURCE_FREEZE),
        },
        "preaccess_freeze": {
            "path": PREACCESS_FREEZE.relative_to(ROOT).as_posix(),
            "sha256": _sha256(PREACCESS_FREEZE),
        },
        "archive_freeze": {
            "path": ARCHIVE_FREEZE.relative_to(ROOT).as_posix(),
            "sha256": _sha256(ARCHIVE_FREEZE),
        },
        "source_registry": {
            "path": SOURCE_REGISTRY.relative_to(ROOT).as_posix(),
            "sha256": _sha256(SOURCE_REGISTRY),
        },
        "source_reference": {
            "source_baselines": json.loads(SOURCE_BASELINES.read_text(encoding="utf-8")),
            "source_noace": json.loads(SOURCE_NOACE.read_text(encoding="utf-8")),
            "source_noace_deep": json.loads(SOURCE_NOACE_DEEP.read_text(encoding="utf-8")),
            "source_noace_physics": json.loads(SOURCE_NOACE_PHYSICS.read_text(encoding="utf-8")),
        },
        "source_partition": {
            "records": int(len(source_arrays["state"])),
            "bearings": int(len(np.unique(source_group_ids))),
        },
        "compound_partition": {
            "records": int(len(compound_arrays["state"])),
            "bearings": int(len(np.unique(compound_group_ids))),
        },
        "primary_confirmation": {
            "name": primary.name,
            "family": primary.family,
            "representation": primary.representation,
            "hyperparameter": primary.hyperparameter,
            "record_metrics": primary.record_metrics,
            "bearing_metrics": primary.bearing_metrics,
            "selected_predictions": "results/confirmation/compound_confirmation_selected_predictions.npz",
            "all_model_predictions": [
                f"artifacts/hanoi_hust/compound_confirmation_{model.name.replace('-', '_')}_predictions.npz"
                for model in confirmed
            ],
        },
        "candidate_table": [
            {
                "name": row.name,
                "family": row.family,
                "representation": row.representation,
                "hyperparameter": row.hyperparameter,
                "record_metrics": row.record_metrics,
                "bearing_metrics": row.bearing_metrics,
            }
            for row in confirmed
        ],
        "frozen_controls": {
            "source_numeric_files_opened": int(len(source_arrays["state"])),
            "compound_numeric_files_opened": int(len(compound_arrays["state"])),
            "transient_numeric_files_opened": 0,
        },
        "provenance": {
            "git_commit": _git("rev-parse", "HEAD"),
            "builder": "src/run_hanoi_hust_compound_confirmation.py",
        },
    }
    _atomic_json(OUTPUT_JSON, payload)
    OUTPUT_MD.write_text(_render_markdown(payload), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    print(
        json.dumps(
            build_confirmation(),
            indent=2,
            ensure_ascii=False,
            default=_json_default,
        )
    )


if __name__ == "__main__":
    main()
