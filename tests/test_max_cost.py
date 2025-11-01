import numpy as np

from spectral_pipeline import DataSet, TimeSeries, RecordMeta, FittingResult, GHZ
from spectral_pipeline.cli import export_freq_tables
from spectral_pipeline.fit import MAX_COST


def _make_ds(tag, root):
    ts = TimeSeries(t=np.linspace(0, 1e-9, 5), s=np.zeros(5), meta=RecordMeta(fs=1e9))
    return DataSet(field_mT=1, temp_K=1, tag=tag, ts=ts, root=root)


def test_export_skips_unsuccessful_pairs(tmp_path):
    lf = _make_ds("LF", tmp_path)
    hf = _make_ds("HF", tmp_path)

    out = tmp_path / "freq.xlsx"
    export_freq_tables([(lf, hf)], tmp_path, outfile=out)
    assert not out.exists()

    fit = FittingResult(
        f1=10 * GHZ,
        f2=40 * GHZ,
        zeta1=1.0,
        zeta2=1.0,
        phi1=0.0,
        phi2=0.0,
        A1=1.0,
        A2=1.0,
        k_lf=1.0,
        C_lf=0.0,
        C_hf=0.0,
        cost=MAX_COST,
    )
    lf.fit = hf.fit = fit

    export_freq_tables([(lf, hf)], tmp_path, outfile=out)
    assert out.exists() and out.stat().st_size > 0

