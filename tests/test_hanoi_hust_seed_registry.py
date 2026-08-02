from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "results" / "audits" / "hanoi_hust_seed_registry_status.json"


def test_hanoi_hust_seed_registry_status_matches_current_state() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))

    assert registry["stage"] == "hanoi_hust_seed_registry_status"
    assert registry["status"] in {"seed_registry_pending", "seed_registry_frozen"}
    assert registry["scope"]["project"] == "HANOI_HUST"
    assert registry["scope"]["dataset_record"] == "cbv7jyx4p9"
    assert registry["scope"]["source_partition_bearings"] == 19
    assert registry["scope"]["compound_partition_bearings_sealed"] == 14
    if registry["status"] == "seed_registry_pending":
        assert registry["explicit_seed_information_present"] is False
        assert "seed_list" in registry["required_next_fields"]
        assert "commit_hash" in registry["required_next_fields"]
        assert "artifact_path" in registry["required_next_fields"]
    else:
        assert registry["explicit_seed_information_present"] is True
        assert registry["seed_list"] == [42, 7, 21, 84, 168]
        assert registry["repeat_count"] == 5
        assert registry["artifact_path"] == "results/development/hanoi_hust_source_multiseed.json"
