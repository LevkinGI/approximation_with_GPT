import numpy as np

from spectral_pipeline import (
    DataSet,
    TimeSeries,
    RecordMeta,
    FittingResult,
    GHZ,
)
from spectral_pipeline import cli


def _make_ds(tag, root, *, field=1, temp=1):
    ts = TimeSeries(t=np.linspace(0, 1e-9, 5), s=np.zeros(5), meta=RecordMeta(fs=1e9))
    return DataSet(field_mT=field, temp_K=temp, tag=tag, ts=ts, root=root)


def test_no_crossing_switch_when_force_lf_only_false(monkeypatch, tmp_path):
    lf1 = _make_ds("LF", tmp_path, field=1, temp=1)
    hf1 = _make_ds("HF", tmp_path, field=1, temp=1)
    lf2 = _make_ds("LF", tmp_path, field=1, temp=3)
    hf2 = _make_ds("HF", tmp_path, field=1, temp=3)
    datasets = [lf1, hf1, lf2, hf2]
    monkeypatch.setattr(cli, "load_records", lambda root: datasets)

    axis = np.array([1, 2, 3], dtype=float)
    hf = np.array([40.0, 20.0, 10.0])
    lf = np.array([10.0, 20.0, 30.0])
    np.save(tmp_path / "H_1.npy", np.vstack([axis, hf, lf]))

    called = {"pair": 0, "lf": 0}
    fit_res = FittingResult(
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
    )

    def fake_process_pair(ds_lf, ds_hf, *, use_theory_guess=False):
        called["pair"] += 1
        ds_lf.fit = ds_hf.fit = fit_res
        return fit_res

    def fake_process_lf_only(ds_lf, *, use_theory_guess=False):
        called["lf"] += 1
        ds_lf.fit = fit_res
        return fit_res

    monkeypatch.setattr(cli, "process_pair", fake_process_pair)
    monkeypatch.setattr(cli, "process_lf_only", fake_process_lf_only)

    triples = cli.main(str(tmp_path), return_datasets=True, do_plot=False)

    assert called["pair"] == 2
    assert called["lf"] == 0
    assert len(triples) == 2
    assert triples[0] == (lf1, hf1)
    assert triples[1] == (lf2, hf2)
