from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly


def robust_zscore_per_window_channel(
    windows: np.ndarray,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Normalize [windows, channels, time] using median and MAD over time."""
    values = np.asarray(windows, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError(f"Expected [N, C, T] windows, got shape={values.shape}.")
    median = np.median(values, axis=-1, keepdims=True)
    mad = np.median(np.abs(values - median), axis=-1, keepdims=True)
    return np.asarray((values - median) / (1.4826 * mad + epsilon), dtype=np.float32)


def robust_zscore_channels(traces: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    values = np.asarray(traces, dtype=np.float32)
    median = np.median(values, axis=-1, keepdims=True)
    mad = np.median(np.abs(values - median), axis=-1, keepdims=True)
    return np.asarray((values - median) / (1.4826 * mad + epsilon), dtype=np.float32)


def resample_traces(
    traces: np.ndarray,
    source_rate_hz: float,
    target_rate_hz: float,
) -> np.ndarray:
    if np.isclose(source_rate_hz, target_rate_hz, rtol=0, atol=1e-6):
        return np.asarray(traces, dtype=np.float32)
    ratio = Fraction(target_rate_hz / source_rate_hz).limit_denominator(10_000)
    values = resample_poly(traces, ratio.numerator, ratio.denominator, axis=-1)
    return np.asarray(values, dtype=np.float32)


def make_windows(
    traces: np.ndarray,
    window_samples: int,
    hop_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(traces, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"Expected [C, T] traces, got shape={values.shape}.")
    if window_samples <= 0 or hop_samples <= 0:
        raise ValueError("window_samples and hop_samples must be positive.")
    if values.shape[1] < window_samples:
        raise ValueError(
            f"Selected interval has {values.shape[1]} samples, fewer than the "
            f"required window size {window_samples}."
        )
    view = np.lib.stride_tricks.sliding_window_view(values, window_samples, axis=1)
    view = view[:, ::hop_samples, :]
    windows = np.moveaxis(view, 0, 1)
    starts = np.arange(windows.shape[0], dtype=np.int64) * hop_samples
    return windows, starts


def downsample_preview(
    traces: np.ndarray,
    start_sec: float,
    sampling_rate_hz: float,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    sample_count = traces.shape[1]
    if sample_count <= max_points:
        indices = np.arange(sample_count)
    else:
        indices = np.linspace(0, sample_count - 1, max_points, dtype=np.int64)
    times = start_sec + indices / sampling_rate_hz
    return times.astype(np.float64), np.asarray(traces[:, indices], dtype=np.float32)
