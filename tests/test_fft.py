import numpy as np
from spectral_pipeline.fit import _fft_spectrum


def test_fft_spectrum_peak():
    fs = 1e9
    t = np.arange(1024) / fs
    s = np.sin(2 * np.pi * 50e6 * t)
    freqs, asd = _fft_spectrum(s, fs)
    peak_freq = freqs[np.argmax(asd)]
    assert abs(peak_freq - 50e6) < 2e6
