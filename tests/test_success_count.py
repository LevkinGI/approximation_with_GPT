import logging

import numpy as np

from spectral_pipeline import DataSet, TimeSeries, RecordMeta, FittingResult, GHZ
from spectral_pipeline import cli


def _make_ds(tag, root, *, field=1, temp=1):
    ts = TimeSeries(t=np.linspace(0, 1e-9, 5), s=np.zeros(5), meta=RecordMeta(fs=1e9))
    return DataSet(field_mT=field, temp_K=temp, tag=tag, ts=ts, root=root)


def test_success_count_and_export(monkeypatch, tmp_path, caplog):
    lf1 = _make_ds("LF", tmp_path, field=1)
    hf1 = _make_ds("HF", tmp_path, field=1)
    lf2 = _make_ds("LF", tmp_path, field=2)
    hf2 = _make_ds("HF", tmp_path, field=2)
    datasets = [lf1, hf1, lf2, hf2]
    monkeypatch.setattr(cli, "load_records", lambda root: datasets)

    fit_res = FittingResult(
        f1=10 * GHZ,
        f2=40 * GHZ,
        zeta1=1.0,
        zeta2=1.0,
        phi1=0.0,
        phi2=0.0,
        A1=1.0,
        A2=1.0,
        k_scale=1.0,
        C_lf=0.0,
        C_hf=0.0,
        cost=1.0,
    )

    def fake_process_pair(ds_lf, ds_hf):
        if ds_lf is lf1:
            ds_lf.fit = ds_hf.fit = fit_res
            return fit_res
        ds_lf.fit = ds_hf.fit = None
        return None

    monkeypatch.setattr(cli, "process_pair", fake_process_pair)
    exported = []
    monkeypatch.setattr(cli, "export_freq_tables", lambda triples, root, outfile=None: exported.append(triples))

    with caplog.at_level(logging.INFO, logger="spectral_pipeline"):
        triples = cli.main(str(tmp_path), return_datasets=True, do_plot=False)

    assert len(triples) == 1
    assert triples[0] == (lf1, hf1)
    assert exported and exported[0] == triples
