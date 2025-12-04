from pathlib import Path
import numpy as np

from spectral_pipeline.io import load_records
from spectral_pipeline import C_M_S

def test_load_records():
    data_dir = Path(__file__).resolve().parents[1] / "data"
    datasets = load_records(data_dir)
    assert datasets
    assert all(ds.ts.t.size > 0 for ds in datasets)


def test_noise_preserved_for_fitting(tmp_path):
    path = tmp_path / "sample_10mT_20K_LF_demo.dat"
    x = np.linspace(0.0, 2.9, 30)
    noise = np.zeros(30)
    s_true = 5.0 * np.exp(-0.3 * x)
    mult_true = 1.7
    add_true = -0.4
    s_raw = s_true + mult_true * noise + add_true
    np.savetxt(path, np.column_stack((x, s_raw, noise)))

    datasets = load_records(tmp_path)

    assert len(datasets) == 1
    ds = datasets[0]

    x0 = x[np.argmax(s_raw)]
    t_all = 2.0 * (x - x0) / C_M_S

    pk = int(np.argmax(s_raw))
    minima = np.where(
        (np.diff(np.signbit(np.diff(s_raw))) > 0)
        & (np.arange(len(s_raw))[1:-1] > pk)
    )[0]
    st = minima[0] + 1 if minima.size else pk + 1

    t = t_all[st:]
    s_expected = s_raw[st:]
    noise_expected = noise[st:]

    cutoff = 0.7e-9
    end = np.searchsorted(t, t[0] + cutoff, "right")
    t = t[:end]
    s_expected = s_expected[:end]
    noise_expected = noise_expected[:end]

    np.testing.assert_allclose(ds.ts.t, t)
    np.testing.assert_allclose(ds.ts.s, s_expected)
    np.testing.assert_allclose(ds.ts.noise, noise_expected)
