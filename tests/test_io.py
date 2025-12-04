from pathlib import Path
import numpy as np

from spectral_pipeline.io import load_records

def test_load_records():
    data_dir = Path(__file__).resolve().parents[1] / "data"
    datasets = load_records(data_dir)
    assert datasets
    assert all(ds.ts.t.size > 0 for ds in datasets)


def test_noise_scaled_and_subtracted(tmp_path):
    path = tmp_path / "sample_10mT_20K_LF_demo.dat"
    x = np.linspace(0.0, 2.9, 30)
    noise = np.linspace(-1.0, 1.0, 30)
    s_true = 5.0 * np.exp(-0.1 * (x - 1.0) ** 2)
    mult_true = 1.7
    add_true = -0.4
    s_raw = s_true + mult_true * noise + add_true
    np.savetxt(path, np.column_stack((x, s_raw, noise)))

    datasets = load_records(tmp_path)

    assert len(datasets) == 1
    ds = datasets[0]

    design = np.column_stack((noise, np.ones_like(noise)))
    coef, *_ = np.linalg.lstsq(design, s_raw, rcond=None)
    s_clean = s_raw - (coef[0] * noise + coef[1])

    pk = int(np.argmax(s_clean))
    minima = np.where(
        (np.diff(np.signbit(np.diff(s_clean))) > 0)
        & (np.arange(len(s_clean))[1:-1] > pk)
    )[0]
    st = minima[0] + 1 if minima.size else pk + 1
    expected = s_clean[st:]

    np.testing.assert_allclose(ds.ts.s, expected)
