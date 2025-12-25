from __future__ import annotations

import math
from typing import Iterable, Optional

import numpy as np
from numba import njit
from numpy.typing import NDArray
from scipy.optimize import least_squares

from . import FittingResult, GHZ, PI, logger

REF_FREQ_RANGE = (7 * GHZ, 14 * GHZ)
RATIO_MIN = 1e-6
RATIO_MAX = 5.0


@njit(fastmath=True, cache=True)
def _numba_residuals_ratio(p, t_all, y_all, weights_all, split_idx, base_is_lf):
    # Параметры: k_lf, k_hf, C_lf, C_hf, A_base, A_other,
    # tau_base, f_base, ratio, phi_base, phi_other
    k_lf = p[0]
    k_hf = p[1]
    C_lf = p[2]
    C_hf = p[3]
    A_base = p[4]
    A_other = p[5]
    tau_base = p[6]
    f_base = p[7]
    ratio = p[8]
    phi_base = p[9]
    phi_other = p[10]

    ratio_safe = ratio if ratio > 0 else RATIO_MIN
    f_other = f_base * ratio_safe
    tau_other = tau_base / ratio_safe

    if base_is_lf:
        A1, A2 = A_base, A_other
        tau1, tau2 = tau_base, tau_other
        f1, f2 = f_base, f_other
        phi1, phi2 = phi_base, phi_other
    else:
        A1, A2 = A_other, A_base
        tau1, tau2 = tau_other, tau_base
        f1, f2 = f_other, f_base
        phi1, phi2 = phi_other, phi_base

    inv_tau1 = -1.0 / tau1
    inv_tau2 = -1.0 / tau2
    omega1 = 2.0 * np.pi * f1
    omega2 = 2.0 * np.pi * f2

    residuals = np.empty_like(y_all)

    for i in range(split_idx):
        t = t_all[i]
        signal = (
            A1 * np.exp(t * inv_tau1) * np.cos(omega1 * t + phi1)
            + A2 * np.exp(t * inv_tau2) * np.cos(omega2 * t + phi2)
        )
        residuals[i] = (k_lf * signal + C_lf - y_all[i]) * weights_all[i]

    for i in range(split_idx, len(t_all)):
        t = t_all[i]
        signal = (
            A1 * np.exp(t * inv_tau1) * np.cos(omega1 * t + phi1)
            + A2 * np.exp(t * inv_tau2) * np.cos(omega2 * t + phi2)
        )
        residuals[i] = (k_hf * signal + C_hf - y_all[i]) * weights_all[i]

    return residuals


def _piecewise_time_weights(t: np.ndarray) -> np.ndarray:
    if t.size == 0:
        return np.ones_like(t)
    t_min = t.min()
    t_len = t.max() - t_min
    if t_len <= 0:
        return np.ones_like(t)
    borders = t_min + np.array([1, 2]) * t_len / 3
    w = np.ones_like(t)
    w[t >= borders[0]] = 1
    w[t >= borders[1]] = 1
    return w


def _span_around(value: float, margin: float) -> tuple[float, float]:
    lo = value * (1 - margin)
    hi = value * (1 + margin)
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi


def _sigma_from_cov(cov: NDArray, idx: int) -> float:
    try:
        val = float(cov[idx, idx])
    except Exception:
        return float("nan")
    if not np.isfinite(val) or val < 0:
        return float("nan")
    return math.sqrt(val)


def _choose_bases(fit: FittingResult) -> list[bool]:
    bases: list[bool] = []
    if REF_FREQ_RANGE[0] <= fit.f1 <= REF_FREQ_RANGE[1]:
        bases.append(True)
    if REF_FREQ_RANGE[0] <= fit.f2 <= REF_FREQ_RANGE[1]:
        bases.append(False)
    return bases


def _prepare_ratio_problem(
    ds_lf,
    ds_hf,
    init_fit: FittingResult,
    base_is_lf: bool,
    bounds_margin: float,
    freq_bounds: Optional[tuple[tuple[float, float], tuple[float, float]]] = None,
):
    t_lf, y_lf = ds_lf.ts.t, ds_lf.ts.s
    t_hf, y_hf = ds_hf.ts.t, ds_hf.ts.s
    if t_lf.size == 0 or t_hf.size == 0:
        raise ValueError("empty time series")

    if base_is_lf:
        base_freq = init_fit.f1
        other_freq = init_fit.f2
        base_zeta = init_fit.zeta1
        base_amp = init_fit.A1
        other_amp = init_fit.A2
        base_phi = init_fit.phi1
        other_phi = init_fit.phi2
        k_base, k_other = init_fit.k_lf, init_fit.k_hf
        c_base, c_other = init_fit.C_lf, init_fit.C_hf
    else:
        base_freq = init_fit.f2
        other_freq = init_fit.f1
        base_zeta = init_fit.zeta2
        base_amp = init_fit.A2
        other_amp = init_fit.A1
        base_phi = init_fit.phi2
        other_phi = init_fit.phi1
        k_base, k_other = init_fit.k_hf, init_fit.k_lf
        c_base, c_other = init_fit.C_hf, init_fit.C_lf

    if not np.isfinite(base_freq) or base_freq <= 0:
        raise ValueError("invalid base frequency")
    if not np.isfinite(other_freq) or other_freq <= 0:
        raise ValueError("invalid other frequency")
    if base_zeta is None or not np.isfinite(base_zeta) or base_zeta <= 0:
        raise ValueError("invalid base zeta")

    tau_base = 1.0 / base_zeta
    ratio_init = other_freq / base_freq
    if ratio_init <= 0 or not np.isfinite(ratio_init):
        raise ValueError("invalid ratio")

    k_lf = init_fit.k_lf
    k_hf = init_fit.k_hf
    C_lf = init_fit.C_lf
    C_hf = init_fit.C_hf

    p0 = np.array(
        [
            k_lf,
            k_hf,
            C_lf,
            C_hf,
            base_amp,
            other_amp,
            tau_base,
            base_freq,
            ratio_init,
            base_phi,
            other_phi,
        ]
    )

    lo_k_lf, hi_k_lf = _span_around(k_lf, bounds_margin)
    lo_k_hf, hi_k_hf = _span_around(k_hf, bounds_margin)
    lo_C_lf, hi_C_lf = _span_around(C_lf, bounds_margin)
    lo_C_hf, hi_C_hf = _span_around(C_hf, bounds_margin)
    lo_A_base, hi_A_base = _span_around(base_amp, bounds_margin)
    lo_A_other, hi_A_other = _span_around(other_amp, bounds_margin)
    lo_tau_base, hi_tau_base = _span_around(tau_base, bounds_margin)
    lo_f_base, hi_f_base = _span_around(base_freq, bounds_margin)
    lo_ratio, hi_ratio = _span_around(ratio_init, bounds_margin)
    lo_ratio = max(lo_ratio, RATIO_MIN)
    hi_ratio = min(hi_ratio, RATIO_MAX)

    if freq_bounds is not None:
        (f1_lo, f1_hi), (f2_lo, f2_hi) = freq_bounds
        if base_is_lf:
            lo_f_base = max(lo_f_base, f1_lo)
            hi_f_base = min(hi_f_base, f1_hi)
            lo_ratio = max(lo_ratio, f2_lo / max(hi_f_base, 1e-12))
            hi_ratio = min(hi_ratio, f2_hi / max(lo_f_base, 1e-12))
        else:
            lo_f_base = max(lo_f_base, f2_lo)
            hi_f_base = min(hi_f_base, f2_hi)
            lo_ratio = max(lo_ratio, f1_lo / max(hi_f_base, 1e-12))
            hi_ratio = min(hi_ratio, f1_hi / max(lo_f_base, 1e-12))
    if hi_f_base <= lo_f_base:
        lo_f_base, hi_f_base = _span_around(base_freq, bounds_margin)
    if hi_ratio <= lo_ratio:
        hi_ratio = max(lo_ratio * 1.01, lo_ratio + RATIO_MIN)

    lo = np.array(
        [
            lo_k_lf,
            lo_k_hf,
            lo_C_lf,
            lo_C_hf,
            max(0.0, lo_A_base),
            max(0.0, lo_A_other),
            max(1e-15, lo_tau_base),
            max(1e3, lo_f_base),
            lo_ratio,
            -PI - 1e-5,
            -PI - 1e-5,
        ]
    )
    hi = np.array(
        [
            hi_k_lf,
            hi_k_hf,
            hi_C_lf,
            hi_C_hf,
            max(hi_A_base, 0.0),
            max(hi_A_other, 0.0),
            max(hi_tau_base, lo_tau_base * 1.1),
            hi_f_base,
            hi_ratio,
            PI + 1e-5,
            PI + 1e-5,
        ]
    )

    w_lf_raw = _piecewise_time_weights(t_lf)
    n_lf = y_lf.size
    n_hf = y_hf.size
    norm_lf = 1.0 / math.sqrt(n_lf) if n_lf > 0 else 0.0
    norm_hf = 1.0 / math.sqrt(n_hf) if n_hf > 0 else 0.0
    weights_lf = w_lf_raw * norm_lf
    weights_hf = np.full(n_hf, norm_hf, dtype=y_hf.dtype)
    weights_all = np.concatenate((weights_lf, weights_hf))
    t_all = np.concatenate((t_lf, t_hf))
    y_all = np.concatenate((y_lf, y_hf))
    split_idx = len(t_lf)

    return p0, lo, hi, t_all, y_all, weights_all, split_idx


def _result_from_solution(
    sol,
    base_is_lf: bool,
):
    p = sol.x
    cost = sol.cost
    try:
        m, n = sol.jac.shape
        cov = np.linalg.inv(sol.jac.T @ sol.jac) * 2 * cost / max(m - n, 1)
    except Exception:
        cov = np.full((p.size, p.size), np.nan)

    k_lf, k_hf, C_lf, C_hf = p[0], p[1], p[2], p[3]
    A_base, A_other = p[4], p[5]
    tau_base, f_base, ratio = p[6], p[7], p[8]
    phi_base, phi_other = p[9], p[10]

    ratio_safe = ratio if ratio > 0 else RATIO_MIN
    f_other = f_base * ratio_safe
    tau_other = tau_base / ratio_safe

    if base_is_lf:
        A1, A2 = A_base, A_other
        tau1, tau2 = tau_base, tau_other
        f1, f2 = f_base, f_other
        phi1, phi2 = phi_base, phi_other
    else:
        A1, A2 = A_other, A_base
        tau1, tau2 = tau_other, tau_base
        f1, f2 = f_other, f_base
        phi1, phi2 = phi_other, phi_base

    sigma_k_lf = _sigma_from_cov(cov, 0)
    sigma_k_hf = _sigma_from_cov(cov, 1)
    sigma_C_lf = _sigma_from_cov(cov, 2)
    sigma_C_hf = _sigma_from_cov(cov, 3)
    sigma_A_base = _sigma_from_cov(cov, 4)
    sigma_A_other = _sigma_from_cov(cov, 5)
    sigma_tau_base = _sigma_from_cov(cov, 6)
    sigma_f_base = _sigma_from_cov(cov, 7)
    sigma_ratio = _sigma_from_cov(cov, 8)
    sigma_phi_base = _sigma_from_cov(cov, 9)
    sigma_phi_other = _sigma_from_cov(cov, 10)

    cov_ratio_base = float(cov[7, 8]) if cov.shape[0] > 8 else float("nan")
    cov_tau_ratio = float(cov[6, 8]) if cov.shape[0] > 8 else float("nan")

    sigma_f_other = float("nan")
    if np.isfinite(sigma_f_base) and np.isfinite(sigma_ratio):
        sigma_f_other_sq = (
            (ratio_safe ** 2) * (sigma_f_base ** 2)
            + (f_base ** 2) * (sigma_ratio ** 2)
            + 2 * ratio_safe * f_base * cov_ratio_base
        )
        if sigma_f_other_sq >= 0:
            sigma_f_other = math.sqrt(sigma_f_other_sq)

    sigma_tau_other = float("nan")
    if np.isfinite(sigma_tau_base) and np.isfinite(sigma_ratio):
        d_tau_base = 1.0 / ratio_safe
        d_ratio = -tau_base / (ratio_safe ** 2)
        sigma_tau_other_sq = (
            (d_tau_base ** 2) * (sigma_tau_base ** 2)
            + (d_ratio ** 2) * (sigma_ratio ** 2)
            + 2 * d_tau_base * d_ratio * cov_tau_ratio
        )
        if sigma_tau_other_sq >= 0:
            sigma_tau_other = math.sqrt(sigma_tau_other_sq)

    if base_is_lf:
        sigma_A1, sigma_A2 = sigma_A_base, sigma_A_other
        sigma_f1, sigma_f2 = sigma_f_base, sigma_f_other
        sigma_tau1, sigma_tau2 = sigma_tau_base, sigma_tau_other
        sigma_phi1, sigma_phi2 = sigma_phi_base, sigma_phi_other
    else:
        sigma_A1, sigma_A2 = sigma_A_other, sigma_A_base
        sigma_f1, sigma_f2 = sigma_f_other, sigma_f_base
        sigma_tau1, sigma_tau2 = sigma_tau_other, sigma_tau_base
        sigma_phi1, sigma_phi2 = sigma_phi_other, sigma_phi_base

    return FittingResult(
        f1=f1,
        f2=f2,
        zeta1=1 / tau1,
        zeta2=1 / tau2,
        phi1=phi1,
        phi2=phi2,
        A1=A1,
        A2=A2,
        k_lf=k_lf,
        k_hf=k_hf,
        C_lf=C_lf,
        C_hf=C_hf,
        f1_err=sigma_f1,
        f2_err=sigma_f2,
        k_lf_err=sigma_k_lf,
        k_hf_err=sigma_k_hf,
        C_lf_err=sigma_C_lf,
        C_hf_err=sigma_C_hf,
        A1_err=sigma_A1,
        A2_err=sigma_A2,
        tau1_err=sigma_tau1,
        tau2_err=sigma_tau2,
        phi1_err=sigma_phi1,
        phi2_err=sigma_phi2,
        cost=cost,
    )


def refine_single_with_ratio(
    ds_lf,
    ds_hf,
    init_fit: FittingResult,
    base_is_lf: bool,
    *,
    bounds_margin: float = 0.15,
    freq_bounds: Optional[tuple[tuple[float, float], tuple[float, float]]] = None,
):
    prepared = _prepare_ratio_problem(
        ds_lf,
        ds_hf,
        init_fit,
        base_is_lf,
        bounds_margin,
        freq_bounds,
    )
    p0, lo, hi, t_all, y_all, weights_all, split_idx = prepared

    def residuals(p):
        return _numba_residuals_ratio(p, t_all, y_all, weights_all, split_idx, base_is_lf)

    sol = least_squares(
        residuals,
        p0,
        bounds=(lo, hi),
        method="trf",
        ftol=3e-16,
        xtol=3e-16,
        gtol=3e-16,
        max_nfev=5000,
        loss="soft_l1",
        f_scale=0.1,
        x_scale="jac",
        verbose=0,
    )
    fit_res = _result_from_solution(sol, base_is_lf)
    return fit_res, float(sol.cost)


def refine_ratio_candidates(
    ds_lf,
    ds_hf,
    candidates: Iterable[tuple[FittingResult, float]],
    freq_bounds: Optional[tuple[tuple[float, float], tuple[float, float]]] = None,
    *,
    bounds_margin: float = 0.15,
    keep_margin: float = 0.15,
    enabled: bool = True,
):
    if not enabled:
        return []

    candidates = list(candidates)
    if not candidates:
        return []

    best_cost = min(cost for _, cost in candidates)
    seeds = [
        fit for fit, cost in candidates
        if cost <= best_cost * (1 + keep_margin)
    ]
    results: list[tuple[FittingResult, float]] = []
    for seed in seeds:
        base_flags = _choose_bases(seed)
        if not base_flags:
            continue
        for base_is_lf in base_flags:
            try:
                refined = refine_single_with_ratio(
                    ds_lf,
                    ds_hf,
                    seed,
                    base_is_lf,
                    bounds_margin=bounds_margin,
                    freq_bounds=freq_bounds,
                )
            except Exception as exc:
                logger.debug(
                    "Ratio refinement skipped (base_is_lf=%s, f_base=%.3f ГГц): %s",
                    base_is_lf,
                    seed.f1 / GHZ if base_is_lf else seed.f2 / GHZ,
                    exc,
                )
                continue
            results.append(refined)
    return results
