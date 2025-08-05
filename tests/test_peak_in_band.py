from spectral_pipeline.fit import _peak_in_band, GHZ
import numpy as np

def test_peak_in_band_fallback_without_theory():
    fs = 1e11
    t = np.arange(400) / fs
    f0 = 10 * GHZ
    y = np.cos(2 * np.pi * f0 * t)
    freqs = np.linspace(0, fs / 2, 500)
    amps = np.zeros_like(freqs)
    f_est = _peak_in_band(freqs, amps, 8, 12, t=t, y=y, fs=fs)
    assert f_est is not None
    assert abs(f_est - f0) < 0.5 * GHZ
