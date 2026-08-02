"""Load the HANOI HUST seed registry status record."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "results" / "audits" / "hanoi_hust_seed_registry_status.json"
EXPECTED_STAGE = "hanoi_hust_seed_registry_status"
EXPECTED_STATUSES = {"seed_registry_pending", "seed_registry_frozen"}


def load_seed_registry_status(path: Path = REGISTRY_PATH) -> dict[str, Any]:
    """Load and validate the seed-registry status record."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("stage") != EXPECTED_STAGE:
        raise ValueError("HANOI seed registry stage changed")
    if data.get("status") not in EXPECTED_STATUSES:
        raise ValueError("HANOI seed registry status changed")
    return data


def seed_registry_summary(path: Path = REGISTRY_PATH) -> dict[str, Any]:
    """Return the minimal summary used by downstream planning notes."""
    data = load_seed_registry_status(path)
    scope = data["scope"]
    return {
        "project": scope["project"],
        "dataset_record": scope["dataset_record"],
        "source_partition_bearings": scope["source_partition_bearings"],
        "compound_partition_bearings_sealed": scope["compound_partition_bearings_sealed"],
        "explicit_seed_information_present": data["explicit_seed_information_present"],
        "allowed_current_statement": data["allowed_current_statement"],
        "required_next_fields": list(data["required_next_fields"]),
    }
