from __future__ import annotations

import numpy as np

from electrical_fm.hanoi_hust_features import (
    BLOCK_DIMENSIONS,
    WINDOW_OFFSETS,
    record_feature_blocks,
    window_starts,
)


def _signal() -> np.ndarray:
    time = np.arange(512_000, dtype=np.float32) / 51_200.0
    return np.sin(2 * np.pi * 37.0 * time) + 0.1 * np.cos(2 * np.pi * 91.0 * time)


def test_window_contract_never_crosses_records() -> None:
    starts = window_starts(512_000, offset=WINDOW_OFFSETS[1])
    assert len(starts) > 0
    assert np.all(starts >= 0)
    assert np.all(starts + 51_200 <= 512_000)


def test_record_feature_blocks_have_frozen_finite_dimensions() -> None:
    blocks, starts = record_feature_blocks(_signal(), offset=WINDOW_OFFSETS[0])
    assert len(starts) > 0
    assert set(blocks) == set(BLOCK_DIMENSIONS)
    for name, values in blocks.items():
        assert values.shape == (BLOCK_DIMENSIONS[name],)
        assert np.isfinite(values).all()
