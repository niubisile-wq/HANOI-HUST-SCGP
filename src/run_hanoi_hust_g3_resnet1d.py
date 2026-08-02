"""Run a compact source-only ResNet1D baseline on HANOI-HUST windows."""

from __future__ import annotations

import hashlib, json, os, random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from electrical_fm.hanoi_hust_baselines import aggregate_group_predictions, component_targets, compute_multilabel_metrics

ROOT = Path(__file__).resolve().parents[1]
WAVEFORM_CACHE = ROOT / "artifacts/hanoi_hust_window/source_window_waveforms.npy"
WAVEFORM_METADATA = ROOT / "artifacts/hanoi_hust_window/source_window_waveforms_metadata.json"
MANIFESTS = ROOT / "results/g2/split_manifests/hanoi_hust_g2_manifests.json"
OUTPUT = ROOT / "results/g3/hanoi_hust_resnet1d_bearing_grouped.json"
PREDICTIONS = ROOT / "results/g3/unit_level_predictions/hanoi_hust_resnet1d_bearing_grouped.npz"

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""): h.update(block)
    return h.hexdigest()

def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); staging = path.with_suffix(".writing")
    staging.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"); os.replace(staging, path)

def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); staging = path.with_suffix(".writing")
    with staging.open("wb") as f: np.savez_compressed(f, **arrays)
    os.replace(staging, path)

class Block(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(cin, cout, 7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(cout); self.conv2 = nn.Conv1d(cout, cout, 7, padding=3, bias=False); self.bn2 = nn.BatchNorm1d(cout)
        self.act = nn.ReLU(); self.skip = nn.Identity() if cin == cout and stride == 1 else nn.Sequential(nn.Conv1d(cin, cout, 1, stride=stride, bias=False), nn.BatchNorm1d(cout))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.bn1(self.conv1(x))); y = self.bn2(self.conv2(y)); return self.act(y + self.skip(x))

class ResNet1D(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 8, 31, stride=2, padding=15, bias=False), nn.BatchNorm1d(8), nn.ReLU(), nn.MaxPool1d(2),
            Block(8, 8), Block(8, 16, 2), Block(16, 16), Block(16, 32, 2), Block(32, 32), nn.AdaptiveAvgPool1d(1)
        )
        self.head = nn.Linear(32, 3)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(x).squeeze(-1))

def _seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.use_deterministic_algorithms(True, warn_only=True)

def _prepare(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values[:, ::64], dtype=np.float32); values -= values.mean(axis=1, keepdims=True); return values / np.maximum(values.std(axis=1, keepdims=True), 1e-6)

def _fit_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, seed: int, epochs: int) -> np.ndarray:
    _seed(seed); model = ResNet1D(); pos = y_train.sum(axis=0); neg = len(y_train) - pos
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg / np.maximum(pos, 1), dtype=torch.float32)); opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    ds = TensorDataset(torch.from_numpy(x_train[:, None, :]), torch.from_numpy(y_train.astype(np.float32))); gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(ds, batch_size=256, shuffle=True, generator=gen, num_workers=0); model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True); loss = loss_fn(model(xb), yb); loss.backward(); opt.step()
    model.eval();
    with torch.no_grad(): return torch.sigmoid(model(torch.from_numpy(x_test[:, None, :]))).numpy().astype(np.float64)

def run(*, count: int = 10, epochs: int = 3) -> dict[str, Any]:
    metadata = json.loads(WAVEFORM_METADATA.read_text(encoding="utf-8")); row = metadata["metadata"]; waveforms = np.load(WAVEFORM_CACHE, mmap_mode="r", allow_pickle=False)
    states = np.asarray(row["state"], dtype=str); bearings = np.asarray(row["bearing_id"], dtype=str); labels = component_targets(states)
    manifests = json.loads(MANIFESTS.read_text(encoding="utf-8"))["bearing_grouped"][:count]; rows=[]; fs=[]; fg=[]; ft=[]; fp=[]; fy=[]
    for split_id, manifest in enumerate(manifests):
        train = np.flatnonzero(np.isin(bearings, manifest["train_groups"])); test = np.flatnonzero(np.isin(bearings, manifest["test_groups"]))
        probs = _fit_predict(_prepare(np.asarray(waveforms[train])), labels[train], _prepare(np.asarray(waveforms[test])), 20260730 + split_id, epochs)
        grouped = aggregate_group_predictions(labels[test], probs, bearings[test]); metrics = compute_multilabel_metrics(grouped["targets"], grouped["probabilities"], grouped["predictions"])
        rows.append({"split_id": split_id, "train_window_count": int(len(train)), "test_window_count": int(len(test)), "metrics": metrics}); fs.extend([split_id]*len(grouped["groups"])); fg.extend(grouped["groups"].tolist()); ft.append(grouped["targets"]); fp.append(grouped["probabilities"]); fy.append(grouped["predictions"])
    _atomic_npz(PREDICTIONS, {"split_id":np.asarray(fs,dtype=np.int16),"groups":np.asarray(fg,dtype="U"),"targets":np.concatenate(ft).astype(np.int8),"probabilities":np.concatenate(fp).astype(np.float32),"predictions":np.concatenate(fy).astype(np.int8)})
    names=("mean_component_auroc","mean_component_aupr","mean_component_balanced_accuracy","mean_component_macro_f1","mean_brier_score","exact_set_accuracy","hamming_loss")
    summary={n:{"mean":float(np.mean([r["metrics"][n] for r in rows])),"std":float(np.std([r["metrics"][n] for r in rows],ddof=1)) if len(rows)>1 else 0.0} for n in names}
    result={"stage":"hanoi_hust_as_g3_resnet1d","schema_version":1,"status":"completed","method_level":"R2","information_budget":"I0_source_only","independent_metric_unit":"physical_bearing","candidate":{"family":"ResNet1D","epochs":epochs,"downsample":64,"batch_size":256,"input":"raw_one_channel_window"},"waveform_cache":{"path":WAVEFORM_CACHE.relative_to(ROOT).as_posix(),"sha256":_sha256(WAVEFORM_CACHE)},"manifest_sha256":_sha256(MANIFESTS),"split_count":len(rows),"summary":summary,"splits":rows,"predictions":{"path":PREDICTIONS.relative_to(ROOT).as_posix(),"sha256":_sha256(PREDICTIONS)}}
    _atomic_json(OUTPUT,result); return result

if __name__ == "__main__":
    import argparse
    p=argparse.ArgumentParser(); p.add_argument("--count",type=int,default=10); p.add_argument("--epochs",type=int,default=3); a=p.parse_args(); print(json.dumps(run(count=a.count,epochs=a.epochs),indent=2,ensure_ascii=False))
