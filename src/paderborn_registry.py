"""Load the frozen Paderborn confirmatory registry."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "results" / "audits" / "paderborn_confirmatory_registry.json"
EXPECTED_PROTOCOL_VERSION = "Table 10 confirmatory subset"


def load_registry(path: Path = REGISTRY_PATH) -> dict[str, Any]:
    """Load and validate the frozen Paderborn confirmatory registry."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("protocol", {}).get("version") != EXPECTED_PROTOCOL_VERSION:
        raise ValueError("Paderborn registry protocol version changed")
    if data.get("qualification", {}).get("status") != "qualified_conditional":
        raise ValueError("Paderborn registry status changed")
    return data


def registry_summary(path: Path = REGISTRY_PATH) -> dict[str, Any]:
    """Return the small subset of fields needed by experiment entrypoints."""
    data = load_registry(path)
    return {
        "protocol_version": data["protocol"]["version"],
        "qualification_status": data["qualification"]["status"],
        "numeric_access_authorized": data["qualification"][
            "numeric_access_authorized"
        ],
        "validation_status": data["validation"]["status"],
        "drift_codes": data["validation"]["drift_codes"],
    }
