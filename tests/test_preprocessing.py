import numpy as np

from weblfp.preprocessing import make_windows, robust_zscore_per_window_channel


def test_make_windows_uses_requested_window_and_hop() -> None:
    traces = np.arange(2 * 20, dtype=np.float32).reshape(2, 20)

    windows, starts = make_windows(traces, window_samples=8, hop_samples=4)

    assert windows.shape == (4, 2, 8)
    assert starts.tolist() == [0, 4, 8, 12]
    np.testing.assert_array_equal(windows[1], traces[:, 4:12])


def test_robust_zscore_is_per_window_and_channel() -> None:
    windows = np.array(
        [
            [[1, 2, 3, 4, 100], [8, 8, 8, 8, 8]],
            [[10, 11, 12, 13, 14], [-2, -1, 0, 1, 2]],
        ],
        dtype=np.float32,
    )

    normalized = robust_zscore_per_window_channel(windows)

    assert normalized.shape == windows.shape
    assert np.isfinite(normalized).all()
    np.testing.assert_allclose(normalized[0, 1], 0)
    np.testing.assert_allclose(np.median(normalized, axis=-1), 0, atol=1e-6)
