import numpy as np
from pathlib import Path

from spectral_pipeline.io import load_records


def test_baseline_const_extraction(tmp_path: Path):
    # Construct a simple signal with a flat baseline before a peak
    x = np.linspace(137.1, 143.2, 25)
    pre_peak = np.array([10.0, 10.0, 10.0, 9.0, 10.0])  # median baseline should be 10
    peak_and_tail = np.array([12.0, 15.0, 20.0, 14.0, 13.0, 12.0, 11.5, 11.0, 10.5, 10.0])
    s = np.concatenate([pre_peak, peak_and_tail, np.full(10, 10.0)])
    data = np.column_stack((x[: s.size], s))
    path = tmp_path / "test_1mT_1K_LF_dummy.dat"
    np.savetxt(path, data)

    datasets = load_records(tmp_path)
    assert len(datasets) == 1
    ds = datasets[0]
    assert np.isclose(ds.baseline_const, 10.0, atol=1e-6)
    # Ensure signal was baseline-corrected
    assert np.all(np.abs(ds.ts.s) < 15.0)  # shifted down from ~20 peak
