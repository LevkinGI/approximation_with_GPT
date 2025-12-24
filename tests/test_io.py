from pathlib import Path

import numpy as np

from spectral_pipeline.io import load_records

def test_load_records():
    data_dir = Path(__file__).resolve().parents[1] / "data"
    datasets = load_records(data_dir)
    assert datasets
    assert all(ds.ts.t.size > 0 for ds in datasets)


def test_load_records_additive_constant(tmp_path):
    x = np.linspace(0.0, 1.0, 80)
    baseline = np.full(20, 2.5)
    transition = np.array([2.0, 1.0, 2.0, 3.0])
    peak_and_tail = np.array([5.0, 8.0, 10.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.5])
    signal = np.concatenate([baseline, transition, peak_and_tail])
    signal = np.pad(signal, (0, max(0, x.size - signal.size)), mode="edge")
    data = np.column_stack((x, signal))
    path = tmp_path / "sample_100mT_4K_LF_dummy.dat"
    np.savetxt(path, data)

    datasets = load_records(tmp_path)
    assert datasets, "dataset should be loaded from synthetic file"
    const_init = datasets[0].additive_const_init
    assert const_init is not None
    assert np.isclose(const_init, np.median(baseline))
