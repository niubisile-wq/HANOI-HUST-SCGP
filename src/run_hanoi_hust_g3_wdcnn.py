"""Run a compact, source-only WDCNN-style baseline on HANOI-HUST windows."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from electrical_fm.hanoi_hust_baselines import (
    aggregate_group_predictions,
    component_targets,
    compute_multilabel_metrics,
)

ROOT = Path(__file__).resolve().parents[1]
WAVEFORM_CACHE = ROOT / "artifacts" / "hanoi_hust_window" / "source_window_waveforms.npy"
WAVEFORM_METADATA = ROOT / "artifacts" / "hanoi_hust_window" / "source_window_waveforms_metadata.json"
MANIFESTS = ROOT / "results" / "g2" / "split_manifests" / "hanoi_hust_g2_manifests.json"
OUTPUT = ROOT / "results" / "g3" / "hanoi_hust_wdcnn_bearing_grouped.json"
PREDICTIONS = ROOT / "results" / "g3" / "unit_level_predictions" / "hanoi_hust_wdcnn_bearing_grouped.npz"


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


class WDCNN(nn.Module):
    def __init__(self, outputs: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=31, stride=2, padding=15),
            nn.BatchNorm1d(8), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(64, outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x).squeeze(-1))


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _prepare(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values[:, ::64], dtype=np.float32)
    values = values - values.mean(axis=1, keepdims=True)
    scale = values.std(axis=1, keepdims=True)
    return values / np.maximum(scale, 1e-6)


def _fit_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, seed: int, epochs: int) -> np.ndarray:
    _seed(seed)
    model = WDCNN().cpu()
    positives = y_train.sum(axis=0)
    negatives = len(y_train) - positives
    pos_weight = torch.tensor(negatives / np.maximum(positives, 1), dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train = TensorDataset(torch.from_numpy(x_train[:, None, :]), torch.from_numpy(y_train.astype(np.float32)))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(train, batch_size=256, shuffle=True, generator=generator, num_workers=0)
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(x_test[:, None, :]))
        return torch.sigmoid(logits).numpy().astype(np.float64)


def run(*, count: int = 10, epochs: int = 12) -> dict[str, Any]:
    metadata = json.loads(WAVEFORM_METADATA.read_text(encoding="utf-8"))
    row_metadata = metadata["metadata"]
    waveforms = np.load(WAVEFORM_CACHE, mmap_mode="r", allow_pickle=False)
    if waveforms.shape != (metadata["window_count"], metadata["window_samples"]):
        raise RuntimeError("WDCNN waveform cache shape changed")
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
        probabilities = _fit_predict(
            _prepare(np.asarray(waveforms[train_windows])),
            labels[train_windows],
            _prepare(np.asarray(waveforms[test_windows])),
            20_260_730 + split_id,
            epochs,
        )
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
        "stage": "hanoi_hust_as_g3_wdcnn", "schema_version": 1, "status": "completed", "method_level": "R2",
        "information_budget": "I0_source_only", "independent_metric_unit": "physical_bearing",
        "candidate": {"family": "WDCNN_style", "epochs": epochs, "downsample": 64, "input": "raw_one_channel_window", "optimizer": "AdamW", "batch_size": 256},
        "waveform_cache": {"path": WAVEFORM_CACHE.relative_to(ROOT).as_posix(), "sha256": _sha256(WAVEFORM_CACHE)},
        "manifest_sha256": _sha256(MANIFESTS), "split_count": len(rows), "summary": summary, "splits": rows,
        "predictions": {"path": PREDICTIONS.relative_to(ROOT).as_posix(), "sha256": _sha256(PREDICTIONS)},
    }
    _atomic_json(OUTPUT, result)
    return result


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()
    print(json.dumps(run(count=args.count, epochs=args.epochs), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
