"""Load the frozen HANOI HUST source registry."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "results" / "audits" / "hanoi_hust_source_registry.json"
EXPECTED_STAGE = "hanoi_hust_source_registry"
EXPECTED_STATUS = "source_registry_frozen"


def load_registry(path: Path = REGISTRY_PATH) -> dict[str, Any]:
    """Load and validate the frozen HANOI HUST source registry."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("stage") != EXPECTED_STAGE:
        raise ValueError("HANOI registry stage changed")
    if data.get("status") != EXPECTED_STATUS:
        raise ValueError("HANOI registry status changed")
    if data.get("authorization", {}).get("source_numeric_parse") is not True:
        raise ValueError("HANOI source numeric authorization changed")
    if data.get("authorization", {}).get("compound_numeric_parse") is not False:
        raise ValueError("HANOI compound numeric authorization changed")
    return data


def registry_summary(path: Path = REGISTRY_PATH) -> dict[str, Any]:
    """Return the minimal summary used by downstream experiment scripts."""
    data = load_registry(path)
    baseline = data["source_baseline"]
    return {
        "dataset": data["dataset"]["record"],
        "version": data["dataset"]["version"],
        "source_partition": data["record_partition"]["source_bearings"],
        "compound_partition": data["record_partition"]["compound_bearings_sealed"],
        "source_numeric_parse": data["authorization"]["source_numeric_parse"],
        "compound_numeric_parse": data["authorization"]["compound_numeric_parse"],
        "selected_family": baseline["selected_family"],
        "selected_representation": baseline["selected_representation"],
        "mean_component_auroc": baseline["mean_component_auroc"],
        "mean_component_balanced_accuracy": baseline[
            "mean_component_balanced_accuracy"
        ],
        "exact_set_accuracy": baseline["exact_set_accuracy"],
        "dominant_view": data["ablation"]["dominant_view"],
        "dominant_family": data["ablation"]["dominant_family"],
    }
