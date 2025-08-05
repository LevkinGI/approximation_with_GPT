import numpy as np
from pathlib import Path
from spectral_pipeline import DataSet, TimeSeries, RecordMeta, FittingResult, GHZ
from spectral_pipeline import fit


def test_process_pair_passes_matching_series(monkeypatch):
    fs = 1e11
    t = np.arange(10) / fs
    ts_lf = TimeSeries(t=t, s=np.zeros_like(t), meta=RecordMeta(fs=fs))
    ts_hf = TimeSeries(t=t, s=np.ones_like(t), meta=RecordMeta(fs=fs))
    lf = DataSet(field_mT=1, temp_K=1, tag="LF", ts=ts_lf, root=Path("."))
    hf = DataSet(field_mT=1, temp_K=1, tag="HF", ts=ts_hf, root=Path("."))

    monkeypatch.setattr(fit, "_load_guess", lambda *a, **k: (9 * GHZ, 27 * GHZ))
    monkeypatch.setattr(
        fit,
        "_fft_spectrum",
        lambda y, fs, window_name="hamming", df_target_GHz=0.1: (np.array([0.0, 1.0]), np.array([0.0, 1.0])),
    )
    monkeypatch.setattr(
        fit,
        "_esprit_freqs_and_decay",
        lambda r, fs, p=6: (np.array([9 * GHZ, 27 * GHZ]), np.array([1.0, 1.0])),
    )
    calls = []

    def fake_peak(freqs, amps, fmin, fmax, **kwargs):
        calls.append(kwargs.get("y"))
        return 1 * GHZ

    monkeypatch.setattr(fit, "_peak_in_band", fake_peak)
    monkeypatch.setattr(
        fit,
        "fit_pair",
        lambda *a, **k: (FittingResult(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), 0.0),
    )

    fit.process_pair(lf, hf)

    assert calls[0] is lf.ts.s
    assert calls[1] is hf.ts.s
