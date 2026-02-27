from __future__ import annotations

import numpy as np
from numba import njit
from numpy.typing import NDArray


@njit(fastmath=True, cache=True)
def _core_signal(t: NDArray, A1, A2, tau1, tau2, f1, f2, phi1, phi2) -> NDArray:
    inv_tau1 = -1.0 / tau1
    inv_tau2 = -1.0 / tau2
    omega1 = 2 * np.pi * f1
    omega2 = 2 * np.pi * f2

    term1 = A1 * np.exp(t * inv_tau1) * np.cos(omega1 * t + phi1)
    term2 = A2 * np.exp(t * inv_tau2) * np.cos(omega2 * t + phi2)
    return term1 + term2


@njit(fastmath=True, cache=True)
def _core_signal_sine_zero_phase(t: NDArray, A1, A2, tau1, tau2, f1, f2) -> NDArray:
    inv_tau1 = -1.0 / tau1
    inv_tau2 = -1.0 / tau2
    omega1 = 2 * np.pi * f1
    omega2 = 2 * np.pi * f2
    return A1 * np.exp(t * inv_tau1) * np.sin(omega1 * t) + A2 * np.exp(t * inv_tau2) * np.sin(omega2 * t)


@njit(fastmath=True, cache=True)
def _numba_residuals(p, t_all, y_all, weights_all, split_idx):
    # Порядок: k_lf, k_hf, C_lf, C_hf, A1, A2, tau1, tau2, f1, f2, phi1, phi2
    k_lf = p[0]
    k_hf = p[1]
    C_lf = p[2]
    C_hf = p[3]
    A1 = p[4]
    A2 = p[5]
    tau1 = p[6]
    tau2 = p[7]
    f1 = p[8]
    f2 = p[9]
    phi1 = p[10]
    phi2 = p[11]

    inv_tau1 = -1.0 / tau1
    inv_tau2 = -1.0 / tau2
    omega1 = 2.0 * np.pi * f1
    omega2 = 2.0 * np.pi * f2

    residuals = np.empty_like(y_all)
    for i in range(split_idx):
        t = t_all[i]
        signal = (A1 * np.exp(t * inv_tau1) * np.cos(omega1 * t + phi1) +
                  A2 * np.exp(t * inv_tau2) * np.cos(omega2 * t + phi2))
        residuals[i] = (k_lf * signal + C_lf - y_all[i]) * weights_all[i]

    for i in range(split_idx, len(t_all)):
        t = t_all[i]
        signal = (A1 * np.exp(t * inv_tau1) * np.cos(omega1 * t + phi1) +
                  A2 * np.exp(t * inv_tau2) * np.cos(omega2 * t + phi2))
        residuals[i] = (k_hf * signal + C_hf - y_all[i]) * weights_all[i]

    return residuals


@njit(fastmath=True, cache=True)
def _numba_residuals_single(p, t, y, w):
    # Порядок: k, C, A1, A2, tau1, tau2, f1, f2, phi1, phi2
    k = p[0]
    C = p[1]
    A1 = p[2]
    A2 = p[3]
    tau1 = p[4]
    tau2 = p[5]
    f1 = p[6]
    f2 = p[7]
    phi1 = p[8]
    phi2 = p[9]

    inv_tau1 = -1.0 / tau1
    inv_tau2 = -1.0 / tau2
    omega1 = 2.0 * np.pi * f1
    omega2 = 2.0 * np.pi * f2

    res = np.empty_like(y)
    for i in range(len(t)):
        ti = t[i]
        core = (A1 * np.exp(ti * inv_tau1) * np.cos(omega1 * ti + phi1) +
                A2 * np.exp(ti * inv_tau2) * np.cos(omega2 * ti + phi2))
        res[i] = w[i] * (k * core + C - y[i])
    return res


@njit(fastmath=True, cache=True)
def _numba_residuals_sine_zero_phase_equal_amp(p, t_all, y_all, weights_all, split_idx):
    # Порядок: k_lf, k_hf, C_lf, C_hf, A, tau1, tau2, f1, f2
    k_lf = p[0]
    k_hf = p[1]
    C_lf = p[2]
    C_hf = p[3]
    A = p[4]
    tau1 = p[5]
    tau2 = p[6]
    f1 = p[7]
    f2 = p[8]

    inv_tau1 = -1.0 / tau1
    inv_tau2 = -1.0 / tau2
    omega1 = 2.0 * np.pi * f1
    omega2 = 2.0 * np.pi * f2

    residuals = np.empty_like(y_all)
    for i in range(split_idx):
        t = t_all[i]
        signal = (A * np.exp(t * inv_tau1) * np.sin(omega1 * t) +
                  A * np.exp(t * inv_tau2) * np.sin(omega2 * t))
        residuals[i] = (k_lf * signal + C_lf - y_all[i]) * weights_all[i]

    for i in range(split_idx, len(t_all)):
        t = t_all[i]
        signal = (A * np.exp(t * inv_tau1) * np.sin(omega1 * t) +
                  A * np.exp(t * inv_tau2) * np.sin(omega2 * t))
        residuals[i] = (k_hf * signal + C_hf - y_all[i]) * weights_all[i]

    return residuals


@njit(fastmath=True, cache=True)
def _numba_residuals_single_sine_zero_phase_equal_amp(p, t, y, w):
    # Порядок: k, C, A, tau1, tau2, f1, f2
    k = p[0]
    C = p[1]
    A = p[2]
    tau1 = p[3]
    tau2 = p[4]
    f1 = p[5]
    f2 = p[6]

    inv_tau1 = -1.0 / tau1
    inv_tau2 = -1.0 / tau2
    omega1 = 2.0 * np.pi * f1
    omega2 = 2.0 * np.pi * f2

    res = np.empty_like(y)
    for i in range(len(t)):
        ti = t[i]
        core = (A * np.exp(ti * inv_tau1) * np.sin(omega1 * ti) +
                A * np.exp(ti * inv_tau2) * np.sin(omega2 * ti))
        res[i] = w[i] * (k * core + C - y[i])
    return res
