"""Audit the 12-code Paderborn extension from frozen derived evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "artifacts" / "paderborn_twelve_code_extension_registry.json"
RESULT = ROOT / "results" / "g3" / "hanoi_hust_paderborn_twelve_code_extension.json"
CACHE = ROOT / "results" / "g3" / "hanoi_hust_paderborn_twelve_code_extension_features.npz"
SOURCE = ROOT / "artifacts" / "hanoi_hust" / "source_features.npz"
RUNNER = ROOT / "src" / "run_hanoi_hust_paderborn_twelve_code_extension.py"
AUDIT = ROOT / "results" / "g3" / "hanoi_hust_paderborn_twelve_code_extension_audit.json"
RAW_ARCHIVES = ROOT.parent / "external_lockbox_paderborn_extension_20260802" / "archives"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    groups = registry["boundary_set"]["bearing_codes"]
    expected_codes = groups["healthy"] + groups["outer_race_damage"] + groups["inner_race_damage"]
    expected_targets = {
        **{code: [0, 0, 0] for code in groups["healthy"]},
        **{code: [0, 1, 0] for code in groups["outer_race_damage"]},
        **{code: [1, 0, 0] for code in groups["inner_race_damage"]},
    }
    checks: dict[str, bool] = {
        "registry_hash_matches_result": sha256(REGISTRY) == result["registry"]["sha256"],
        "runner_hash_matches_registry": sha256(RUNNER)
        == registry["execution"]["script_sha256"],
        "source_hash_matches_registry": sha256(SOURCE)
        == registry["source_model"]["source_cache_sha256"],
        "feature_cache_hash_matches_result": sha256(CACHE)
        == result["external_feature_cache"]["sha256"],
        "twelve_disjoint_codes_registered": len(expected_codes) == len(set(expected_codes)) == 12,
        "external_tuning_prohibited": registry["external_tuning_permitted"] is False,
    }

    with np.load(CACHE, allow_pickle=False) as external:
        codes = external["codes"].astype(str)
        targets = external["targets"].astype(np.int8)
        external_x = np.asarray(external["envelope_log_power"], dtype=np.float32)
    checks["cache_has_960_records"] = len(codes) == 960
    checks["cache_has_80_records_per_code"] = all(
        int(np.sum(codes == code)) == 80 for code in expected_codes
    )
    checks["cache_targets_match_registry"] = all(
        np.all(targets[codes == code] == np.asarray(expected_targets[code], dtype=np.int8))
        for code in expected_codes
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
    source_z = scaler.transform(source_x)
    external_z = scaler.transform(external_x)
    probabilities = np.zeros((len(codes), 3), dtype=np.float64)
    for component in range(3):
        model = LogisticRegression(
            C=10.0,
            class_weight="balanced",
            max_iter=5000,
            random_state=20260730,
        )
        model.fit(source_z, source_y[:, component])
        probabilities[:, component] = model.predict_proba(external_z)[:, 1]

    rebuilt_rows = []
    for code in expected_codes:
        mask = codes == code
        probability = probabilities[mask].mean(axis=0)
        target = np.asarray(expected_targets[code], dtype=np.int8)
        prediction = (probability >= 0.5).astype(np.int8)
        rebuilt_rows.append(
            {
                "bearing_code": code,
                "target": target,
                "probability": probability,
                "prediction": prediction,
                "exact_set": bool(np.all(target == prediction)),
            }
        )
    reported_by_code = {row["bearing_code"]: row for row in result["unit_rows"]}
    checks["all_unit_probabilities_rebuilt"] = all(
        np.allclose(
            row["probability"],
            reported_by_code[row["bearing_code"]]["probability"],
            rtol=0,
            atol=1e-12,
        )
        for row in rebuilt_rows
    )
    checks["all_unit_predictions_rebuilt"] = all(
        row["prediction"].tolist()
        == reported_by_code[row["bearing_code"]]["prediction"]
        for row in rebuilt_rows
    )
    unit_targets = np.stack([row["target"] for row in rebuilt_rows])
    unit_probabilities = np.stack([row["probability"] for row in rebuilt_rows])
    unit_predictions = np.stack([row["prediction"] for row in rebuilt_rows])
    exact = np.all(unit_targets == unit_predictions, axis=1)
    rebuilt = {
        "inner_auroc": float(roc_auc_score(unit_targets[:, 0], unit_probabilities[:, 0])),
        "outer_auroc": float(roc_auc_score(unit_targets[:, 1], unit_probabilities[:, 1])),
        "exact_set_accuracy": float(exact.mean()),
        "exact_set_count": [int(exact.sum()), int(len(exact))],
        "hamming_loss": float(np.mean(unit_targets != unit_predictions)),
    }
    checks["headline_metrics_rebuilt"] = (
        rebuilt["inner_auroc"] == result["metrics"]["inner"]["auroc"]
        and rebuilt["outer_auroc"] == result["metrics"]["outer"]["auroc"]
        and rebuilt["exact_set_accuracy"] == result["metrics"]["exact_set_accuracy"]
        and rebuilt["exact_set_count"] == result["metrics"]["exact_set_count"]
        and rebuilt["hamming_loss"] == result["metrics"]["hamming_loss"]
    )

    raw_available = RAW_ARCHIVES.is_dir()
    raw_hashes_match = None
    if raw_available:
        raw_hashes_match = all(
            sha256(RAW_ARCHIVES / archive) == expected
            for archive, expected in result["external_extension"]["archive_sha256"].items()
        )
        checks["raw_archive_hashes_match_when_available"] = raw_hashes_match

    failures = [name for name, passed in checks.items() if not passed]
    audit = {
        "stage": "hanoi_hust_paderborn_twelve_code_extension_audit",
        "schema_version": 1,
        "status": "passed" if not failures else "failed",
        "result_sha256": sha256(RESULT),
        "registry_sha256": sha256(REGISTRY),
        "feature_cache_sha256": sha256(CACHE),
        "raw_archives_available": raw_available,
        "checks": checks,
        "rebuilt_metrics": rebuilt,
        "failures": failures,
    }
    AUDIT.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
