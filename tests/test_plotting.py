import pathlib

from spectral_pipeline.approximation_config import ApproximationConfig
from spectral_pipeline.io import load_records
from spectral_pipeline.plotting import visualize_stacked


def test_visualize_stacked_html(tmp_path):
    data_dir = pathlib.Path(__file__).resolve().parents[1] / "data"
    datasets = load_records(data_dir)
    assert datasets
    grouped = {}
    for ds in datasets:
        grouped.setdefault((ds.field_mT, ds.temp_K), {})[ds.tag] = ds
    pairs = []
    for pair in grouped.values():
        if "LF" in pair and "HF" in pair:
            pairs.append((pair["LF"], pair["HF"]))
    assert pairs
    out = tmp_path / "plot.html"
    cfg = ApproximationConfig(spectrum_freq_bounds_ghz=(5.0, 40.0))
    visualize_stacked(pairs, outfile=str(out), approximation_config=cfg)
    assert out.exists() and out.stat().st_size > 0
