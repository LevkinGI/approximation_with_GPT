import numpy as np
from spectral_pipeline.fit import _fallback_peak, GHZ


def test_fallback_with_avg_fft():
    fs = 1e11
    t = np.arange(400) / fs
    f0 = 40 * GHZ
    y = np.cos(2 * np.pi * f0 * t)
    # Force Burg failure by using too high order
    f_est = _fallback_peak(t, y, fs, (30 * GHZ, 50 * GHZ), f_rough=0.0, order_burg=512)
    assert abs(f_est - f0) < 0.5 * GHZ


def test_fallback_frequency_changes_with_length():
    fs = 1e11
    f0 = 40.3 * GHZ

    t_short = np.arange(400) / fs
    y_short = np.cos(2 * np.pi * f0 * t_short)

    t_long = np.arange(800) / fs
    y_long = np.cos(2 * np.pi * f0 * t_long)

    f_short = _fallback_peak(t_short, y_short, fs, (30 * GHZ, 50 * GHZ), f_rough=0.0, order_burg=512)
    f_long = _fallback_peak(t_long, y_long, fs, (30 * GHZ, 50 * GHZ), f_rough=0.0, order_burg=512)

    assert f_short is not None and f_long is not None
    assert f_short != f_long


def test_fallback_short_signal_returns_none():
    fs = 1e11
    f0 = 40.3 * GHZ
    t = np.arange(20) / fs
    y = np.cos(2 * np.pi * f0 * t)

    f_est = _fallback_peak(t, y, fs, (30 * GHZ, 50 * GHZ), f_rough=0.0)

    assert f_est is None
