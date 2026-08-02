from __future__ import annotations

from src.hanoi_hust_registry import (
    EXPECTED_STAGE,
    EXPECTED_STATUS,
    REGISTRY_PATH,
    load_registry,
    registry_summary,
)


def test_registry_loader_returns_expected_summary() -> None:
    registry = load_registry()
    summary = registry_summary()

    assert REGISTRY_PATH.exists()
    assert registry["stage"] == EXPECTED_STAGE
    assert registry["status"] == EXPECTED_STATUS
    assert summary == {
        "dataset": "cbv7jyx4p9",
        "version": 3,
        "source_partition": 19,
        "compound_partition": 14,
        "source_numeric_parse": True,
        "compound_numeric_parse": False,
        "selected_family": "logistic_l2",
        "selected_representation": "envelope_log_power",
        "mean_component_auroc": 0.8138447971781305,
        "mean_component_balanced_accuracy": 0.7427248677248678,
        "exact_set_accuracy": 0.543859649122807,
        "dominant_view": "envelope_log_power",
        "dominant_family": "logistic_l2",
    }
