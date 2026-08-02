from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_extension_registry_has_twelve_balanced_disjoint_codes() -> None:
    extension = json.loads(
        (ROOT / "artifacts" / "paderborn_twelve_code_extension_registry.json").read_text(
            encoding="utf-8"
        )
    )
    pilot = json.loads(
        (ROOT / "artifacts" / "paderborn_three_code_boundary_registry.json").read_text(
            encoding="utf-8"
        )
    )
    groups = extension["boundary_set"]["bearing_codes"]
    codes = groups["healthy"] + groups["outer_race_damage"] + groups["inner_race_damage"]
    pilot_groups = pilot["boundary_set"]["bearing_codes"]
    pilot_codes = set(
        pilot_groups["healthy"]
        + pilot_groups["outer_race_damage"]
        + pilot_groups["inner_race_damage"]
    )
    assert [len(groups[name]) for name in groups] == [4, 4, 4]
    assert len(codes) == len(set(codes)) == 12
    assert not (set(codes) & pilot_codes)


def test_extension_registry_freezes_source_only_model() -> None:
    registry = json.loads(
        (ROOT / "artifacts" / "paderborn_twelve_code_extension_registry.json").read_text(
            encoding="utf-8"
        )
    )
    model = registry["source_model"]
    assert registry["information_budget"] == "I0_source_only"
    assert registry["access_authorization"]["numeric_access_authorized"] is True
    assert registry["external_tuning_permitted"] is False
    assert model["representation"] == "envelope_log_power"
    assert model["classifier"] == "logistic_l2"
    assert model["threshold"] == 0.5
