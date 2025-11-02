import numpy as np
import pytest

from spectral_pipeline import DataSet, TimeSeries, RecordMeta, GHZ, FittingResult
from spectral_pipeline import fit
from spectral_pipeline.fit import _core_signal, MAX_COST


def _make_pair() -> tuple[DataSet, DataSet, dict[str, float]]:
    params = {
        "f1": 9.5 * GHZ,
        "f2": 29.0 * GHZ,
        "tau1": 0.28e-9,
        "tau2": 0.12e-9,
        "phi1": 0.3,
        "phi2": -0.9,
        "A1": 0.8,
        "A2": 1.1,
        "k_lf": 1.2,
        "k_hf": 1.5,
        "C_lf": 0.05,
        "C_hf": -0.02,
    }
    fs_lf = 3.0e11
    fs_hf = 3.0e12
    t_lf = np.arange(600) / fs_lf
    t_hf = np.arange(1800) / fs_hf
    core_lf = _core_signal(
        t_lf,
        params["A1"],
        params["A2"],
        params["tau1"],
        params["tau2"],
        params["f1"],
        params["f2"],
        params["phi1"],
        params["phi2"],
    )
    core_hf = _core_signal(
        t_hf,
        params["A1"],
        params["A2"],
        params["tau1"],
        params["tau2"],
        params["f1"],
        params["f2"],
        params["phi1"],
        params["phi2"],
    )
    rng = np.random.default_rng(42)
    noise_lf = rng.normal(scale=0.01, size=core_lf.shape)
    noise_hf = rng.normal(scale=0.01, size=core_hf.shape)
    y_lf = params["k_lf"] * core_lf + params["C_lf"] + noise_lf
    y_hf = params["k_hf"] * core_hf + params["C_hf"] + noise_hf
    ds_lf = DataSet(
        field_mT=100,
        temp_K=300,
        tag="LF",
        ts=TimeSeries(t=t_lf, s=y_lf, meta=RecordMeta(fs=fs_lf)),
    )
    ds_hf = DataSet(
        field_mT=100,
        temp_K=300,
        tag="HF",
        ts=TimeSeries(t=t_hf, s=y_hf, meta=RecordMeta(fs=fs_hf)),
    )
    return ds_lf, ds_hf, params


def test_process_pair_uses_reverse_hierarchy(monkeypatch):
    ds_lf, ds_hf, params = _make_pair()

    def forbid(*_args, **_kwargs):  # pragma: no cover - ensure legacy code unused
        raise AssertionError("legacy candidate search should not be used")

    monkeypatch.setattr(fit, "_cwt_gaussian_candidates", forbid)
    monkeypatch.setattr(fit, "_peak_in_band", forbid)
    monkeypatch.setattr(fit, "_fallback_peak", forbid)

    fit_res = fit.process_pair(ds_lf, ds_hf)
    assert fit_res is not None
    assert ds_lf.fit is fit_res and ds_hf.fit is fit_res

    assert abs(fit_res.f1 - params["f1"]) < 0.5 * GHZ
    assert abs(fit_res.f2 - params["f2"]) < 0.5 * GHZ
    assert fit_res.f2 > fit_res.f1
    assert abs(fit_res.f1 - 10 * GHZ) >= 0.5 * GHZ
    assert abs(fit_res.f2 - 20 * GHZ) >= 0.5 * GHZ
    assert fit_res.cost is not None and fit_res.cost < MAX_COST


def test_process_pair_rejects_large_cost(monkeypatch):
    ds_lf, ds_hf, _ = _make_pair()

    def fake_fit_pair(*_args, **_kwargs):
        res = FittingResult(
            f1=10 * GHZ,
            f2=20 * GHZ,
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
            cost=MAX_COST + 1.0,
        )
        return res, res.cost

    monkeypatch.setattr(fit, "fit_pair", fake_fit_pair)

    assert fit.process_pair(ds_lf, ds_hf) is None
    assert ds_lf.fit is None and ds_hf.fit is None
