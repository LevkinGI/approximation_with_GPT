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
