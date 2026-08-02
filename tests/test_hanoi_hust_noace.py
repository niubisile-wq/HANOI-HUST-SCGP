from __future__ import annotations

import numpy as np
import pytest

from electrical_fm.hanoi_hust_noace import (
    _build_nuisance_design,
    _select_view,
    _fit_nuisance,
    _predict_nuisance,
    _prepare_view,
    _subset_states,
)


def test_prepare_view_averages_window_axis() -> None:
    features = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]],
        ],
        dtype=np.float32,
    )
    values = _prepare_view(features)
    assert values.shape == (2, 2)
    assert np.allclose(values[0], np.array([3.0, 4.0], dtype=np.float64))
    assert np.allclose(values[1], np.array([6.0, 8.0], dtype=np.float64))


def test_subset_states_maps_symbol_states_to_components() -> None:
    states = np.asarray(["N", "I", "O", "B", "N"], dtype=str)
    healthy, inner, outer, ball = _subset_states(states)
    assert healthy.tolist() == [True, False, False, False, True]
    assert inner.tolist() == [False, True, False, False, False]
    assert outer.tolist() == [False, False, True, False, False]
    assert ball.tolist() == [False, False, False, True, False]


def test_subset_states_rejects_missing_singleton_component() -> None:
    states = np.asarray(["N", "N", "B"], dtype=str)
    with pytest.raises(ValueError, match="NOACE requires singleton records for inner"):
        _subset_states(states)


def test_fit_and_predict_nuisance_rebuild_consistent_design() -> None:
    rng = np.random.default_rng(20260727)
    features = rng.normal(size=(8, 4))
    bearing_type = np.array([1, 2, 2, 3, 3, 3, 4, 4], dtype=np.int64)
    load_w = np.array([10, 10, 20, 10, 20, 20, 10, 20], dtype=np.int64)

    residual, coeff, bearing_categories, load_categories = _fit_nuisance(
        features=features,
        bearing_type=bearing_type,
        load_w=load_w,
    )
    assert residual.shape == features.shape
    assert coeff.shape[0] == 1 + len(bearing_categories) + len(load_categories)

    rebuilt = _predict_nuisance(
        coeff=coeff,
        bearing_categories=bearing_categories,
        load_categories=load_categories,
        bearing_test=np.array([1, 2, 4], dtype=np.int64),
        load_test=np.array([20, 20, 10], dtype=np.int64),
        x_test_count=3,
    )
    assert rebuilt.shape == (3, features.shape[1])


def test_build_nuisance_design_raises_unknown_categories() -> None:
    values = np.array([1, 2], dtype=np.int64)
    with pytest.raises(ValueError, match="Unknown nuisance category encountered"):
        _build_nuisance_design(values, categories=(1,))


def test_select_view_maps_features_by_name() -> None:
    arrays = {
        "statistics": np.zeros((2, 3, 4), dtype=np.float32),
        "fixed_log_power": np.ones((2, 3, 5), dtype=np.float32),
        "envelope_log_power": np.full((2, 3, 6), 2.0, dtype=np.float32),
    }
    statistics = _select_view(arrays, representation="statistics")
    fixed = _select_view(arrays, representation="fixed_log_power")
    envelope = _select_view(arrays, representation="envelope_log_power")
    all_view = _select_view(arrays, representation="all")
    assert statistics.shape == (2, 3, 4)
    assert fixed.shape == (2, 3, 5)
    assert envelope.shape == (2, 3, 6)
    assert all_view.shape == (2, 3, 15)
    assert np.allclose(_select_view(arrays, representation="statistics"), np.zeros((2, 3, 4)))
    assert np.allclose(_select_view(arrays, representation="fixed_log_power"), np.ones((2, 3, 5)))


def test_select_view_rejects_unknown_representation() -> None:
    arrays = {
        "statistics": np.zeros((1, 1, 1), dtype=np.float32),
        "fixed_log_power": np.zeros((1, 1, 1), dtype=np.float32),
        "envelope_log_power": np.zeros((1, 1, 1), dtype=np.float32),
    }
    with pytest.raises(ValueError, match="Unknown NOACE representation"):
        _select_view(arrays, representation="bogus")
