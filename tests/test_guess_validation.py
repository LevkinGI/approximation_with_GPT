import logging
import numpy as np

from spectral_pipeline import DataSet, TimeSeries, RecordMeta, FittingResult, GHZ
from spectral_pipeline import fit


def _make_ds(tag, root):
    ts = TimeSeries(t=np.linspace(0, 1e-9, 5), s=np.zeros(5), meta=RecordMeta(fs=1e9))
    return DataSet(field_mT=1, temp_K=1, tag=tag, ts=ts, root=root)


def test_out_of_band_guess_is_ignored(monkeypatch, tmp_path):
    lf = _make_ds("LF", tmp_path)
    hf = _make_ds("HF", tmp_path)

    monkeypatch.setattr(fit, "_load_guess", lambda *a, **k: (4 * GHZ, 9 * GHZ))
    monkeypatch.setattr(fit, "_esprit_freqs_and_decay", lambda *a, **k: (np.array([]), np.array([])))
    monkeypatch.setattr(
        fit,
        "_fft_spectrum",
        lambda sig, fs, window_name="hamming", df_target_GHz=0.1: (
            np.array([0.0, 1.0, 2.0]) * GHZ,
            np.array([0.0, 0.0, 0.0]),
        ),
    )
    monkeypatch.setattr(fit, "_peak_in_band", lambda *a, **k: None)
    monkeypatch.setattr(fit, "_fallback_peak", lambda *a, **k: 10 * GHZ)
    monkeypatch.setattr(
        fit,
        "_single_sine_refine",
        lambda *a, **k: (10 * GHZ, 0.0, 1.0, 1e-9),
    )

    def fake_fit_pair(ds_lf, ds_hf, freq_bounds=None):
        assert ds_lf.f1_init == 10 * GHZ
        assert ds_hf.f2_init == 10 * GHZ
        res = FittingResult(
            f1=10 * GHZ,
            f2=40 * GHZ,
            zeta1=1.0,
            zeta2=1.0,
            phi1=0.0,
            phi2=0.0,
            A1=1.0,
            A2=1.0,
            k_lf=1.0,
            k_hf=1.0,
            C_lf=0.0,
            C_hf=0.0,
            cost=1.0,
        )
        return res, 1.0

    monkeypatch.setattr(fit, "fit_pair", fake_fit_pair)

    fit.process_pair(lf, hf)
