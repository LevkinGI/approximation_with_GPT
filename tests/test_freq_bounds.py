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

    expected_bounds = ((5 * GHZ, 15 * GHZ), (35 * GHZ, 40 * GHZ))
    monkeypatch.setattr(
        fit,
        "_load_guess",
        lambda *args, **kwargs: (10 * GHZ, 40 * GHZ, *expected_bounds),
    )

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
        assert freq_bounds == expected_bounds
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
        fit.process_pair(lf, hf)

    assert not any("вне freq_bounds" in r.message for r in caplog.records)

