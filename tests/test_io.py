from pathlib import Path
from spectral_pipeline.io import load_records

def test_load_records():
    data_dir = Path(__file__).resolve().parents[1] / "data"
    datasets = load_records(data_dir)
    assert datasets
    assert all(ds.ts.t.size > 0 for ds in datasets)
