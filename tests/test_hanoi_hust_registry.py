from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "results" / "audits" / "hanoi_hust_source_registry.json"


def test_hanoi_hust_source_registry_matches_frozen_artifacts() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))

    assert registry["stage"] == "hanoi_hust_source_registry"
    assert registry["status"] == "source_registry_frozen"
    assert registry["dataset"]["record"] == "cbv7jyx4p9"
    assert registry["dataset"]["version"] == 3
    assert registry["dataset"]["record_count"] == 99
    assert registry["record_partition"]["source_bearings"] == 19
    assert registry["record_partition"]["compound_bearings_sealed"] == 14
    assert registry["source_freeze"]["sha256"] == (
        "f8d725de4ce30cb3c9e6046107cd26c1d3ed326b3617c43194144e29a22c8d4d"
    )
    assert registry["preregistration"]["sha256"] == (
        "fe759865824612fc085261df6ebc2299ebb8b52c00216ca14f794c2d7ad1d39c"
    )
    assert registry["baseline_result"]["sha256"] == (
        "de0b882a8f5f98b98d57ab47ce70cd46850d7ff31d44788ed40635afd857fe13"
    )
    assert registry["ablation_result"]["sha256"] == (
        "9d57df95ef70546d447e268c8706abdce59224231e046487f2b854d532a121d3"
    )
    assert registry["authorization"]["source_numeric_parse"] is True
    assert registry["authorization"]["compound_numeric_parse"] is False
    assert registry["source_baseline"]["selected_family"] == "logistic_l2"
    assert registry["source_baseline"]["selected_representation"] == "envelope_log_power"
    assert registry["source_baseline"]["mean_component_auroc"] == 0.8138447971781305
    assert registry["source_baseline"]["exact_set_accuracy"] == 0.543859649122807
    assert registry["ablation"]["dominant_view"] == "envelope_log_power"
    assert registry["ablation"]["dominant_family"] == "logistic_l2"
