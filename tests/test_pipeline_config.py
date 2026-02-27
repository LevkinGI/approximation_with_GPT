import logging

from spectral_pipeline import logger
from spectral_pipeline.approximation_config import ApproximationConfig
from spectral_pipeline.pipeline import PipelineHooks, run_pipeline


def test_run_pipeline_uses_log_level_from_approximation_config(tmp_path):
    hooks = PipelineHooks(
        loader=lambda root, cfg: [],
        pair_processor=lambda ds_lf, ds_hf, *, approximation_config: None,
        lf_only_processor=lambda ds_lf, *, approximation_config: None,
        plotter=lambda triples, *, use_theory_guess, approximation_config: None,
        exporter=lambda triples, root, outfile=None: None,
        crossing_finder=lambda root, field_mT, temp_K: None,
    )

    run_pipeline(
        str(tmp_path),
        do_plot=False,
        approximation_config=ApproximationConfig(log_level="INFO"),
        hooks=hooks,
    )

    assert logger.level == logging.INFO
