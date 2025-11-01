import logging
import numpy as np

from spectral_pipeline import DataSet, TimeSeries, RecordMeta, FittingResult, GHZ
from spectral_pipeline import fit


def _make_ds(tag, root):
    ts = TimeSeries(t=np.linspace(0, 1e-9, 5), s=np.zeros(5), meta=RecordMeta(fs=1e9))
    return DataSet(field_mT=1, temp_K=1, tag=tag, ts=ts, root=root)


def test_no_debug_when_freqs_within_bounds(monkeypatch, tmp_path, caplog):
    lf = _make_ds("LF", tmp_path)
    hf = _make_ds("HF", tmp_path)

    monkeypatch.setattr(fit, "_load_guess", lambda *args, **kwargs: (10 * GHZ, 40 * GHZ))

    def fake_esprit(r, fs, p=6):
        return np.array([10 * GHZ, 40 * GHZ]), np.array([1.0, 1.0])

    monkeypatch.setattr(fit, "_esprit_freqs_and_decay", fake_esprit)
    monkeypatch.setattr(
        fit,
        "_fft_spectrum",
        lambda sig, fs, window_name="hamming", df_target_GHz=0.1: (
            np.array([0.0, 1.0, 2.0]),
            np.array([1.0, 0.5, 0.2]),
        ),
    )
    monkeypatch.setattr(fit, "_peak_in_band", lambda *a, **k: None)

    def fake_fit_pair(ds_lf, ds_hf, freq_bounds=None):
        res = FittingResult(
            f1=ds_lf.f1_init,
            f2=ds_hf.f2_init,
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

    with caplog.at_level(logging.DEBUG, logger="spectral_pipeline"):
        fit.process_pair(lf, hf, use_theory_guess=True)

    assert not any("вне freq_bounds" in r.message for r in caplog.records)


def test_process_pair_uses_guess_flag(monkeypatch, tmp_path):
    lf = _make_ds("LF", tmp_path)
    hf = _make_ds("HF", tmp_path)

    guess_calls = {"count": 0}

    def fake_load_guess(*args, **kwargs):
        guess_calls["count"] += 1
        return 10 * GHZ, 40 * GHZ

    monkeypatch.setattr(fit, "_load_guess", fake_load_guess)
    monkeypatch.setattr(fit, "_single_sine_refine", lambda *a, **k: (10 * GHZ, 0.0, 1.0, 1e-9))
    monkeypatch.setattr(
        fit,
        "multichannel_esprit",
        lambda signals, fs: (np.array([10 * GHZ, 40 * GHZ]), np.array([1.0, 1.0])),
    )
    monkeypatch.setattr(
        fit,
        "_fft_spectrum",
        lambda *a, **k: (
            np.array([0.0, 10 * GHZ, 40 * GHZ]),
            np.array([0.0, 1.0, 0.5]),
        ),
    )

    def fake_peak_in_band(freqs, amps, fmin_GHz, fmax_GHz, **_):
        return 10 * GHZ if fmax_GHz <= 20 else 40 * GHZ

    monkeypatch.setattr(fit, "_peak_in_band", fake_peak_in_band)
    monkeypatch.setattr(fit, "_cwt_gaussian_candidates", lambda *a, **k: ([], []))

    def fake_fit_pair(ds_lf, ds_hf, freq_bounds=None):
        res = FittingResult(
            f1=ds_lf.f1_init,
            f2=ds_hf.f2_init,
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
        ds_lf.fit = ds_hf.fit = res
        return res, 1.0

    monkeypatch.setattr(fit, "fit_pair", fake_fit_pair)

    fit.process_pair(lf, hf)
    assert guess_calls["count"] == 0

    lf2 = _make_ds("LF", tmp_path)
    hf2 = _make_ds("HF", tmp_path)
    fit.process_pair(lf2, hf2, use_theory_guess=True)
    assert guess_calls["count"] == 1

