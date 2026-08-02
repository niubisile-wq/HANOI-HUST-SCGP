"""Run a source-only MiniROCKET baseline on raw HANOI windows."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from sktime.classification.kernel_based import RocketClassifier

from electrical_fm.hanoi_hust_baselines import (
    aggregate_group_predictions,
    component_targets,
    compute_multilabel_metrics,
)


ROOT = Path(__file__).resolve().parents[1]
WAVEFORM_CACHE = ROOT / "artifacts" / "hanoi_hust_window" / "source_window_waveforms.npy"
WAVEFORM_METADATA = ROOT / "artifacts" / "hanoi_hust_window" / "source_window_waveforms_metadata.json"
MANIFESTS = ROOT / "results" / "g2" / "split_manifests" / "hanoi_hust_g2_manifests.json"
OUTPUT = ROOT / "results" / "g3" / "hanoi_hust_minirocket_bearing_grouped.json"
PREDICTIONS = ROOT / "results" / "g3" / "unit_level_predictions" / "hanoi_hust_minirocket_bearing_grouped.npz"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix(".writing")
    staging.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(staging, path)


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix(".writing")
    with staging.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(staging, path)


def run(*, count: int = 10, num_kernels: int = 10_000, downsample: int = 1) -> dict[str, Any]:
    metadata = json.loads(WAVEFORM_METADATA.read_text(encoding="utf-8"))
    waveforms = np.load(WAVEFORM_CACHE, mmap_mode="r", allow_pickle=False)
    if waveforms.shape[0] != metadata["window_count"] or waveforms.shape[1] != metadata["window_samples"]:
        raise RuntimeError("MiniROCKET waveform cache shape changed")
    row_metadata = metadata.get("metadata", {})
    states = np.asarray(row_metadata["state"], dtype=str)
    bearings = np.asarray(row_metadata["bearing_id"], dtype=str)
    labels = component_targets(states)
    manifests = json.loads(MANIFESTS.read_text(encoding="utf-8"))["bearing_grouped"][:count]
    rows: list[dict[str, Any]] = []
    flat_split: list[int] = []
    flat_groups: list[str] = []
    flat_targets: list[np.ndarray] = []
    flat_probabilities: list[np.ndarray] = []
    flat_predictions: list[np.ndarray] = []
    for split_id, manifest in enumerate(manifests):
        train_bearings = set(manifest["train_groups"])
        test_bearings = set(manifest["test_groups"])
        train_windows = np.flatnonzero(np.isin(bearings, list(train_bearings)))
        test_windows = np.flatnonzero(np.isin(bearings, list(test_bearings)))
        x_train = np.asarray(waveforms[train_windows, ::downsample], dtype=np.float32)[:, None, :]
        x_test = np.asarray(waveforms[test_windows, ::downsample], dtype=np.float32)[:, None, :]
        probabilities = np.zeros((len(test_windows), labels.shape[1]), dtype=np.float64)
        for head in range(labels.shape[1]):
            classifier = RocketClassifier(
                num_kernels=num_kernels,
                rocket_transform="minirocket",
                use_multivariate="no",
                n_jobs=1,
                random_state=20_260_730 + split_id,
            )
            classifier.fit(x_train, labels[train_windows, head])
            probabilities[:, head] = classifier.predict_proba(x_test)[:, 1]
        grouped = aggregate_group_predictions(labels[test_windows], probabilities, bearings[test_windows])
        metrics = compute_multilabel_metrics(grouped["targets"], grouped["probabilities"], grouped["predictions"])
        rows.append({"split_id": split_id, "train_window_count": int(len(train_windows)), "test_window_count": int(len(test_windows)), "metrics": metrics})
        flat_split.extend([split_id] * len(grouped["groups"]))
        flat_groups.extend(grouped["groups"].tolist())
        flat_targets.append(grouped["targets"])
        flat_probabilities.append(grouped["probabilities"])
        flat_predictions.append(grouped["predictions"])
    _atomic_npz(PREDICTIONS, {
        "split_id": np.asarray(flat_split, dtype=np.int16),
        "groups": np.asarray(flat_groups, dtype="U"),
        "targets": np.concatenate(flat_targets, axis=0).astype(np.int8),
        "probabilities": np.concatenate(flat_probabilities, axis=0).astype(np.float32),
        "predictions": np.concatenate(flat_predictions, axis=0).astype(np.int8),
    })
    metric_names = ("mean_component_auroc", "mean_component_aupr", "mean_component_balanced_accuracy", "mean_component_macro_f1", "mean_brier_score", "exact_set_accuracy", "hamming_loss")
    summary = {name: {"mean": float(np.mean([row["metrics"][name] for row in rows])), "std": float(np.std([row["metrics"][name] for row in rows], ddof=1)) if len(rows) > 1 else 0.0} for name in metric_names}
    result = {
        "stage": "hanoi_hust_as_g3_minirocket",
        "schema_version": 1,
        "status": "completed",
        "method_level": "R2",
        "information_budget": "I0_source_only",
        "independent_metric_unit": "physical_bearing",
        "candidate": {"family": "MiniROCKET", "num_kernels": num_kernels, "rocket_transform": "minirocket", "input": "raw_one_channel_window", "downsample": downsample},
        "waveform_cache": {"path": WAVEFORM_CACHE.relative_to(ROOT).as_posix(), "sha256": _sha256(WAVEFORM_CACHE)},
        "manifest_sha256": _sha256(MANIFESTS),
        "split_count": len(rows),
        "summary": summary,
        "splits": rows,
        "predictions": {"path": PREDICTIONS.relative_to(ROOT).as_posix(), "sha256": _sha256(PREDICTIONS)},
    }
    _atomic_json(OUTPUT, result)
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--num-kernels", type=int, default=10_000)
    parser.add_argument("--downsample", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(run(count=args.count, num_kernels=args.num_kernels, downsample=args.downsample), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
