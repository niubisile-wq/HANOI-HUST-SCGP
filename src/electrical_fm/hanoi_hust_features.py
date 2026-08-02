"""Frozen one-dimensional features for HANOI HUST source development."""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import hilbert
from scipy.signal.windows import hann
from scipy.stats import kurtosis, skew


SAMPLE_RATE_HZ = 51_200
WINDOW_SAMPLES = SAMPLE_RATE_HZ
WINDOW_OFFSETS = (0, SAMPLE_RATE_HZ // 4, SAMPLE_RATE_HZ // 2)
BANDS = 64
FLOOR = 1e-20
STATISTICS_PER_WINDOW = 14
BLOCK_DIMENSIONS = {
    "statistics": STATISTICS_PER_WINDOW * 2,
    "fixed_log_power": BANDS * 2,
    "envelope_log_power": BANDS * 2,
}


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / max(denominator, FLOOR)


def window_starts(
    sample_count: int,
    *,
    offset: int,
) -> np.ndarray:
    if offset not in WINDOW_OFFSETS:
        raise ValueError("HANOI HUST window offset changed")
    starts = np.arange(
        offset,
        sample_count - WINDOW_SAMPLES + 1,
        WINDOW_SAMPLES,
        dtype=np.int32,
    )
    if len(starts) == 0 or np.any(starts < 0) or np.any(
        starts + WINDOW_SAMPLES > sample_count
    ):
        raise RuntimeError("HANOI HUST window extraction changed")
    return starts


def _statistics(values: np.ndarray, envelope: np.ndarray) -> np.ndarray:
    rms = float(np.sqrt(np.mean(np.square(values))))
    absolute_mean = float(np.mean(np.abs(values)))
    peak = float(np.max(np.abs(values)))
    clearance = float(np.mean(np.sqrt(np.abs(values))) ** 2)
    envelope_rms = float(np.sqrt(np.mean(np.square(envelope))))
    spread = float(np.std(values, ddof=0))
    if spread <= 1e-12:
        skewness = 0.0
        kurt = 0.0
    else:
        skewness = float(skew(values, bias=False))
        kurt = float(kurtosis(values, fisher=True, bias=False))
    output = np.asarray(
        [
            float(values.mean()),
            spread,
            rms,
            absolute_mean,
            float(values.min()),
            float(values.max()),
            peak,
            float(np.ptp(values)),
            skewness,
            kurt,
            _safe_ratio(peak, rms),
            _safe_ratio(peak, absolute_mean),
            _safe_ratio(rms, absolute_mean),
            _safe_ratio(float(envelope.max()), envelope_rms),
        ],
        dtype=np.float64,
    )
    if output.shape != (STATISTICS_PER_WINDOW,) or not np.isfinite(output).all():
        raise RuntimeError("HANOI HUST statistics are invalid")
    return output


def _band_sums(
    values: np.ndarray,
    frequencies_hz: np.ndarray,
    edges_hz: np.ndarray,
) -> np.ndarray:
    indices = np.searchsorted(frequencies_hz, edges_hz, side="left")
    indices[-1] = len(frequencies_hz)
    output = np.empty(
        (*values.shape[:-1], len(edges_hz) - 1),
        dtype=values.dtype,
    )
    for band, (lower, upper) in enumerate(
        zip(indices[:-1], indices[1:], strict=True)
    ):
        if upper <= lower:
            raise RuntimeError("HANOI HUST spectral band is empty")
        output[..., band] = values[..., lower:upper].sum(axis=-1)
    return output


def _window_blocks(signal: np.ndarray) -> dict[str, np.ndarray]:
    centered = signal.astype(np.float64, copy=False)
    centered = centered - centered.mean(axis=-1, keepdims=True)
    envelope = np.abs(hilbert(centered, axis=-1))
    taper = hann(WINDOW_SAMPLES, sym=False)
    spectra = np.fft.rfft(centered * taper, axis=-1)
    envelope_spectra = np.fft.rfft(
        (envelope - envelope.mean(axis=-1, keepdims=True)) * taper,
        axis=-1,
    )
    frequencies = np.fft.rfftfreq(
        WINDOW_SAMPLES,
        d=1.0 / SAMPLE_RATE_HZ,
    )
    edges_hz = np.linspace(0.0, SAMPLE_RATE_HZ / 2, BANDS + 1)
    power = np.square(np.abs(spectra))
    envelope_power = np.square(np.abs(envelope_spectra))
    fixed = np.log10(np.maximum(_band_sums(power, frequencies, edges_hz), FLOOR))
    envelope_log_power = np.log10(
        np.maximum(_band_sums(envelope_power, frequencies, edges_hz), FLOOR)
    )
    statistics = np.stack(
        [
            _statistics(window, envelope_window)
            for window, envelope_window in zip(centered, envelope, strict=True)
        ]
    )
    return {
        "statistics": statistics,
        "fixed_log_power": fixed.reshape(len(signal), -1),
        "envelope_log_power": envelope_log_power.reshape(len(signal), -1),
    }


def record_feature_blocks(
    signal: np.ndarray,
    *,
    offset: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Return median-IQR record blocks and the exact technical windows."""
    window_blocks, starts = window_feature_blocks(signal, offset=offset)
    record_blocks = {}
    for name, block in window_blocks.items():
        median = np.median(block, axis=0)
        quartiles = np.quantile(block, (0.25, 0.75), axis=0)
        aggregate = np.concatenate([median, quartiles[1] - quartiles[0]])
        aggregate = aggregate.astype(np.float32)
        if (
            aggregate.shape != (BLOCK_DIMENSIONS[name],)
            or not np.isfinite(aggregate).all()
        ):
            raise RuntimeError(f"HANOI HUST {name} block is invalid")
        record_blocks[name] = aggregate
    return record_blocks, starts


def window_feature_blocks(
    signal: np.ndarray,
    *,
    offset: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Return one feature row per technical window before record aggregation."""
    values = np.asarray(signal, dtype=np.float32).reshape(-1)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Expected one finite HANOI HUST waveform")
    starts = window_starts(values.size, offset=offset)
    windows = np.stack(
        [values[start : start + WINDOW_SAMPLES] for start in starts],
        axis=0,
    )
    window_blocks = _window_blocks(windows)
    for name, values in window_blocks.items():
        if values.ndim != 2 or values.shape[1] != BLOCK_DIMENSIONS[name] // 2 or not np.isfinite(values).all():
            raise RuntimeError(f"HANOI HUST window {name} block is invalid")
    return {name: values.astype(np.float32) for name, values in window_blocks.items()}, starts
