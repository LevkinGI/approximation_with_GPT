from __future__ import annotations

from typing import Tuple
import math
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from . import GHZ, HF_BAND, LF_BAND


def _resolve_tau_bounds(
    zeta: float | None,
    tau_init_fallback: float,
    default_lo: float,
    default_hi: float,
) -> tuple[float, float, float]:
    """Return (tau_init, tau_lo, tau_hi) with positive, sane bounds."""

    if zeta is not None and np.isfinite(zeta) and zeta > 0:
        tau_init = 1.0 / zeta
        return tau_init, tau_init * 0.8, tau_init * 1.2
    if zeta is not None and np.isfinite(zeta) and zeta < 0:
        tau_init = -1.0 / zeta
        return tau_init, tau_init * 0.8, tau_init * 1.2

    tau_init = tau_init_fallback
    if not np.isfinite(tau_init) or tau_init <= 0:
        tau_init = math.sqrt(default_lo * default_hi)
    return tau_init, default_lo, default_hi


def _single_sine_refine(
    t: NDArray,
    y: NDArray,
    f0: float,
    *,
    lf_band_hz: tuple[float, float] = LF_BAND,
    hf_band_hz: tuple[float, float] = HF_BAND,
) -> tuple[float, float, float, float]:
    A0 = 0.5 * (y.max() - y.min())
    tau0 = (t[-1] - t[0]) / 3
    p0 = [A0, f0, 0.0, tau0, y.mean()]
    lo = np.array([0.0, lf_band_hz[0], -np.pi, tau0 / 10, y.min() - abs(np.ptp(y))])
    hi = np.array([3 * A0, hf_band_hz[1], np.pi, tau0 * 10, y.max() + abs(np.ptp(y))])

    def model(p):
        A, f, phi, tau, C = p
        return A * np.exp(-t / tau) * np.cos(2 * np.pi * f * t + phi) + C

    sol = least_squares(lambda p: model(p) - y, p0, bounds=(lo, hi), method="trf")
    A_est, f_est, phi_est, tau_est, _ = sol.x
    return f_est, phi_est, A_est, tau_est


def _load_guess(directory, field_mT: int, temp_K: int) -> tuple[float, float] | None:
    """Load first-approximation frequencies if file exists."""
    path_H = directory / f"H_{field_mT}.npy"
    path_T = directory / f"T_{temp_K}.npy"
    if path_H.exists():
        path = path_H
        axis_value = temp_K
    elif path_T.exists():
        path = path_T
        axis_value = field_mT
    else:
        return None
    try:
        arr = np.load(path)
    except Exception:
        return None
    if arr.shape[0] < 3:
        return None
    axis = arr[0]
    hf = arr[1]
    lf = arr[2]
    idx = int(np.argmin(np.abs(axis - axis_value)))
    f_lf_GHz = float(lf[idx])
    f_hf_GHz = float(hf[idx])
    return f_lf_GHz * GHZ, f_hf_GHz * GHZ

