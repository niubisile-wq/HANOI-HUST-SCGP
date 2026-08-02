from __future__ import annotations

import numpy as np

from electrical_fm.hanoi_hust_baselines import (
    bearing_groups,
    component_targets,
    representation_views,
)


def test_component_targets_and_groups_are_consistent() -> None:
    states = np.asarray(["N", "I", "O", "B", "N"])
    bearing_type = np.asarray([4, 4, 5, 5, 6], dtype=np.int16)
    targets = component_targets(states)
    groups = bearing_groups(states, bearing_type)
    assert targets.shape == (5, 3)
    assert targets[0].tolist() == [0, 0, 0]
    assert targets[1].tolist() == [1, 0, 0]
    assert targets[2].tolist() == [0, 1, 0]
    assert targets[3].tolist() == [0, 0, 1]
    assert groups.tolist() == ["N4", "I4", "O5", "B5", "N6"]


def test_representation_views_concatenate_three_blocks() -> None:
    arrays = {
        "statistics": np.ones((2, 3, 28), dtype=np.float32),
        "fixed_log_power": np.ones((2, 3, 128), dtype=np.float32),
        "envelope_log_power": np.ones((2, 3, 128), dtype=np.float32),
    }
    views = representation_views(arrays)
    assert views["statistics"].shape == (2, 84)
    assert views["fixed_log_power"].shape == (2, 384)
    assert views["envelope_log_power"].shape == (2, 384)
    assert views["all"].shape == (2, 852)
