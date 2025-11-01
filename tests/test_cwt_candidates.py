from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pywt")

from spectral_pipeline.fit import _cwt_gaussian_candidates, GHZ


def _contains_freq(candidates: list[float], target: float, tol: float) -> bool:
    return any(abs(freq - target) < tol for freq in candidates)


def test_cwt_gaussian_candidates_detect_modes() -> None:
    rng = np.random.default_rng(42)
    fs = 160.0 * GHZ
    duration = 1.5e-9
    t = np.arange(0.0, duration, 1.0 / fs)
    f_lf = 12.0 * GHZ
    f_hf = 31.0 * GHZ
    y = (
        0.9 * np.cos(2 * np.pi * f_lf * t)
        + 0.6 * np.cos(2 * np.pi * f_hf * t)
        + 0.05 * rng.standard_normal(t.size)
    )

    lf_cand, hf_cand = _cwt_gaussian_candidates(
        t,
        y,
        highcut_GHz=40.0,
        time_cutoffs=[(float(t[0]), float(t[-1]))],
    )

    assert _contains_freq(lf_cand, f_lf, 0.5 * GHZ), (
        f"LF candidates {np.array(lf_cand) / GHZ} ГГц do not include {f_lf / GHZ:.2f} ГГц"
    )
    assert _contains_freq(hf_cand, f_hf, 1.0 * GHZ), (
        f"HF candidates {np.array(hf_cand) / GHZ} ГГц do not include {f_hf / GHZ:.2f} ГГц"
    )
