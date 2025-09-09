import numpy as np
import pytest

from spectral_pipeline import DataSet, TimeSeries, RecordMeta, FittingResult, GHZ
from spectral_pipeline import fit


def _make_ds(tag, root):
    ts = TimeSeries(t=np.linspace(0, 1e-9, 5), s=np.zeros(5), meta=RecordMeta(fs=1e9))
    return DataSet(field_mT=1, temp_K=1, tag=tag, ts=ts, root=root)


def _dummy_fit(*args, **kwargs):
    return FittingResult(
        f1=0.0,
        f2=0.0,
        tau1=1.0,
        tau2=1.0,
        phi1=0.0,
        phi2=0.0,
        A1=1.0,
        A2=1.0,
        k_lf=1.0,
        k_hf=1.0,
        C_lf=0.0,
        C_hf=0.0,
        cost=1.0,
    ), 1.0


def test_hf_relax_uses_local_window(monkeypatch, tmp_path):
    lf = _make_ds("LF", tmp_path)
    hf = _make_ds("HF", tmp_path)

    monkeypatch.setattr(fit, "_load_guess", lambda *a, **k: (10 * GHZ, 30 * GHZ, 1e-9, 1e-9))
    monkeypatch.setattr(fit, "multichannel_esprit", lambda signals, fs: (np.array([5 * GHZ]), np.array([1e-9])))
    monkeypatch.setattr(fit, "_fft_spectrum", lambda *a, **k: (np.array([0.0, 1.0]), np.array([1.0, 0.5])))
    monkeypatch.setattr(fit, "_peak_in_band", lambda *a, **k: None)

    band = {}

    def fake_fallback(t, y, fs, rng, guess, avoid=None):
        band["range"] = rng
        return guess

    monkeypatch.setattr(fit, "_fallback_peak", fake_fallback)
    monkeypatch.setattr(fit, "fit_pair", _dummy_fit)

    fit.process_pair(lf, hf)

    assert "range" in band
    low, high = band["range"]
    assert low == pytest.approx(25 * GHZ)
    assert high == pytest.approx(35 * GHZ)


def test_negative_tau_excluded(monkeypatch, tmp_path):
    lf = _make_ds("LF", tmp_path)
    hf = _make_ds("HF", tmp_path)

    monkeypatch.setattr(fit, "_load_guess", lambda *a, **k: (10 * GHZ, 30 * GHZ, 1e-9, 1e-9))
    monkeypatch.setattr(fit, "multichannel_esprit", lambda signals, fs: (np.array([30 * GHZ]), np.array([-1e-9])))
    monkeypatch.setattr(fit, "_fft_spectrum", lambda *a, **k: (np.array([0.0, 1.0]), np.array([1.0, 0.5])))
    monkeypatch.setattr(fit, "_peak_in_band", lambda *a, **k: None)

    called = {"flag": False}

    def fake_fallback(*args, **kwargs):
        called["flag"] = True
        return 30 * GHZ

    monkeypatch.setattr(fit, "_fallback_peak", fake_fallback)
    monkeypatch.setattr(fit, "fit_pair", _dummy_fit)

    fit.process_pair(lf, hf)

    assert called["flag"]
