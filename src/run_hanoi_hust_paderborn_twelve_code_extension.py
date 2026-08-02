"""Pre-access-frozen HANOI-to-Paderborn extension on 12 new bearing codes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly
from scipy.stats import beta
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from electrical_fm.hanoi_hust_features import record_feature_blocks


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts" / "hanoi_hust" / "source_features.npz"
REGISTRY = ROOT / "artifacts" / "paderborn_twelve_code_extension_registry.json"
EXT_ROOT = ROOT.parent / "external_lockbox_paderborn_extension_20260802" / "extracted"
OUT = ROOT / "results" / "g3" / "hanoi_hust_paderborn_twelve_code_extension.json"
CACHE = ROOT / "results" / "g3" / "hanoi_hust_paderborn_twelve_code_extension_features.npz"

LABELS = {
    "K002": np.array([0, 0, 0], dtype=np.int8),
    "K003": np.array([0, 0, 0], dtype=np.int8),
    "K004": np.array([0, 0, 0], dtype=np.int8),
    "K005": np.array([0, 0, 0], dtype=np.int8),
    "KA04": np.array([0, 1, 0], dtype=np.int8),
    "KA15": np.array([0, 1, 0], dtype=np.int8),
    "KA16": np.array([0, 1, 0], dtype=np.int8),
    "KA22": np.array([0, 1, 0], dtype=np.int8),
    "KI04": np.array([1, 0, 0], dtype=np.int8),
    "KI14": np.array([1, 0, 0], dtype=np.int8),
    "KI16": np.array([1, 0, 0], dtype=np.int8),
    "KI18": np.array([1, 0, 0], dtype=np.int8),
}
COMPONENTS = ("inner", "outer", "ball")
BOOTSTRAP_SEED = 20260802
BOOTSTRAP_RESAMPLES = 10_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def vibration(path: Path) -> np.ndarray:
    payload = loadmat(path, squeeze_me=True, struct_as_record=False)
    struct = next(value for key, value in payload.items() if not key.startswith("__"))
    candidates = []
    for channel in np.ravel(struct.Y):
        if getattr(channel, "Name", "") == "vibration_1":
            candidates.append(np.asarray(channel.Data, dtype=np.float64).reshape(-1))
    if len(candidates) != 1:
        raise RuntimeError(f"vibration_1 is not unique in {path}")
    return candidates[0]


def feature_blocks(path: Path) -> dict[str, np.ndarray]:
    signal = vibration(path)
    if signal.size < 200_000:
        raise RuntimeError(f"signal too short: {path} ({signal.size})")
    signal = np.asarray(resample_poly(signal, 4, 5), dtype=np.float32)
    blocks = []
    for offset in (0, 12_800, 25_600):
        block, _ = record_feature_blocks(signal, offset=offset)
        blocks.append(block)
    return {
        name: np.concatenate([block[name] for block in blocks]).astype(np.float32)
        for name in ("statistics", "fixed_log_power", "envelope_log_power")
    }


def clopper_pearson(successes: int, total: int, alpha: float = 0.05) -> list[float]:
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, total - successes + 1))
    upper = 1.0 if successes == total else float(beta.ppf(1 - alpha / 2, successes + 1, total - successes))
    return [lower, upper]


def stratified_auroc_interval(
    targets: np.ndarray, probabilities: np.ndarray, rng: np.random.Generator
) -> list[float]:
    negative = np.flatnonzero(targets == 0)
    positive = np.flatnonzero(targets == 1)
    draws = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for index in range(BOOTSTRAP_RESAMPLES):
        sample = np.concatenate(
            [
                rng.choice(negative, size=negative.size, replace=True),
                rng.choice(positive, size=positive.size, replace=True),
            ]
        )
        draws[index] = roc_auc_score(targets[sample], probabilities[sample])
    return np.quantile(draws, [0.025, 0.975]).astype(float).tolist()


def main() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    registered = registry["boundary_set"]["bearing_codes"]
    registered_codes = registered["healthy"] + registered["outer_race_damage"] + registered["inner_race_damage"]
    if registered_codes != list(LABELS):
        raise RuntimeError("registry order or code identity differs from the frozen script")
    if not registry["access_authorization"]["numeric_access_authorized"]:
        raise RuntimeError("numeric access is not authorized in the frozen registry")
    if sha256(SOURCE) != registry["source_model"]["source_cache_sha256"]:
        raise RuntimeError("source feature cache hash differs from the frozen registry")

    files: list[tuple[str, Path]] = []
    for code in LABELS:
        paths = sorted((EXT_ROOT / code).rglob("*.mat"))
        if len(paths) != 80:
            raise RuntimeError(f"{code}: expected 80 MAT files, found {len(paths)}")
        files.extend((code, path) for path in paths)

    names = ("statistics", "fixed_log_power", "envelope_log_power")
    external = {name: [] for name in names}
    codes: list[str] = []
    filenames: list[str] = []
    for code, path in files:
        blocks = feature_blocks(path)
        for name in names:
            external[name].append(blocks[name])
        codes.append(code)
        filenames.append(path.name)
    arrays = {name: np.stack(external[name]).astype(np.float32) for name in names}
    targets = np.stack([LABELS[code] for code in codes])
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE,
        **arrays,
        codes=np.asarray(codes),
        names=np.asarray(filenames),
        targets=targets,
    )

    with np.load(SOURCE, allow_pickle=False) as source:
        source_x = np.asarray(source["envelope_log_power"], dtype=np.float32).reshape(
            len(source["state"]), -1
        )
        source_y = np.stack(
            [
                np.array([state == "I", state == "O", state == "B"], dtype=np.int8)
                for state in source["state"].astype(str)
            ]
        )

    scaler = StandardScaler().fit(source_x)
    scaled_source = scaler.transform(source_x)
    scaled_external = scaler.transform(arrays["envelope_log_power"])
    probabilities = np.zeros((len(targets), 3), dtype=np.float64)
    for component in range(3):
        model = LogisticRegression(
            C=10.0,
            class_weight="balanced",
            max_iter=5000,
            random_state=20260730,
        )
        model.fit(scaled_source, source_y[:, component])
        probabilities[:, component] = model.predict_proba(scaled_external)[:, 1]

    hard = (probabilities >= 0.5).astype(np.int8)
    unit_rows = []
    code_array = np.asarray(codes)
    for code in LABELS:
        mask = code_array == code
        probability = probabilities[mask].mean(axis=0)
        target = LABELS[code]
        prediction = (probability >= 0.5).astype(np.int8)
        unit_rows.append(
            {
                "bearing_code": code,
                "record_count": int(mask.sum()),
                "target": target.tolist(),
                "probability": probability.tolist(),
                "prediction": prediction.tolist(),
                "exact_set": bool(np.all(target == prediction)),
            }
        )

    unit_probability = np.stack([row["probability"] for row in unit_rows])
    unit_target = np.stack([row["target"] for row in unit_rows])
    unit_prediction = np.stack([row["prediction"] for row in unit_rows])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    metrics: dict[str, object] = {}
    for component in (0, 1):
        target = unit_target[:, component]
        probability = unit_probability[:, component]
        prediction = unit_prediction[:, component]
        metrics[COMPONENTS[component]] = {
            "auroc": float(roc_auc_score(target, probability)),
            "auroc_stratified_bootstrap_95_ci": stratified_auroc_interval(
                target, probability, rng
            ),
            "aupr": float(average_precision_score(target, probability)),
            "balanced_accuracy": float(balanced_accuracy_score(target, prediction)),
            "brier": float(brier_score_loss(target, probability)),
            "macro_f1": float(
                f1_score(target, prediction, average="macro", zero_division=0)
            ),
        }

    exact = np.all(unit_target == unit_prediction, axis=1)
    successes = int(exact.sum())
    class_rows = {}
    for label, prefix in (("healthy", "K"), ("outer", "KA"), ("inner", "KI")):
        if prefix == "K":
            mask = np.array([row["bearing_code"].startswith("K0") for row in unit_rows])
        else:
            mask = np.array([row["bearing_code"].startswith(prefix) for row in unit_rows])
        class_rows[label] = {
            "correct": int(exact[mask].sum()),
            "total": int(mask.sum()),
        }
    metrics["exact_set_accuracy"] = successes / len(unit_rows)
    metrics["exact_set_count"] = [successes, len(unit_rows)]
    metrics["exact_set_clopper_pearson_95_ci"] = clopper_pearson(
        successes, len(unit_rows)
    )
    metrics["exact_set_by_class"] = class_rows
    metrics["hamming_loss"] = float(np.mean(unit_target != unit_prediction))

    archive_root = EXT_ROOT.parent / "archives"
    archive_hashes = {
        f"{code}.rar": sha256(archive_root / f"{code}.rar") for code in LABELS
    }
    output = {
        "stage": "hanoi_hust_paderborn_twelve_code_extension",
        "schema_version": 1,
        "status": "completed",
        "role": "pre_access_frozen_disjoint_external_extension",
        "information_budget": "I0_source_only",
        "independent_metric_unit": "physical_bearing_code",
        "registry": {
            "path": REGISTRY.relative_to(ROOT).as_posix(),
            "sha256": sha256(REGISTRY),
        },
        "source_model": {
            "cache_sha256": sha256(SOURCE),
            "representation": "envelope_log_power",
            "classifier": "logistic_l2",
            "C": 10.0,
            "scaler": "HANOI_source_only",
            "threshold": 0.5,
        },
        "external_extension": {
            "source": registry["source"],
            "license": registry["license"],
            "archive_sha256": archive_hashes,
            "bearing_count": len(LABELS),
            "record_count": len(targets),
            "disjoint_from_three_code_boundary": True,
        },
        "external_feature_cache": {
            "path": CACHE.relative_to(ROOT).as_posix(),
            "sha256": sha256(CACHE),
        },
        "metrics": metrics,
        "unit_rows": unit_rows,
    }
    OUT.write_text(
        json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
