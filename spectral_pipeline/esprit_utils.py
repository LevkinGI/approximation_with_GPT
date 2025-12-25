from __future__ import annotations

from typing import List, Tuple
import numpy as np
from numpy.typing import NDArray

from . import GHZ, logger


def _hankel_matrix(x: NDArray, L: int) -> NDArray:
    N = len(x)
    if L <= 0 or L >= N:
        raise ValueError("L must satisfy 0 < L < N")
    return np.lib.stride_tricks.sliding_window_view(x, L).T


def _esprit_freqs_and_decay(r: NDArray, fs: float, p: int = 6) -> Tuple[NDArray, NDArray]:
    L = len(r) // 2
    H = _hankel_matrix(r, L)
    U, _, _ = np.linalg.svd(H, full_matrices=False)
    Us = U[:, :p]
    Us1, Us2 = Us[:-1], Us[1:]
    Psi = np.linalg.pinv(Us1) @ Us2
    lam = np.linalg.eigvals(Psi)
    dt = 1.0 / fs
    f = np.abs(np.angle(lam)) / (2 * np.pi * dt)
    zeta = -np.log(np.abs(lam)) / dt
    logger.debug("ESPRIT raw freqs: %s", np.round(f / GHZ, 3))
    logger.debug("ESPRIT raw zeta: %s", np.round(zeta, 3))
    return f, zeta


def multichannel_esprit(r_list: List[NDArray], fs: float, p: int = 6) -> Tuple[NDArray, NDArray]:
    """Estimate modal frequencies and decays using multichannel ESPRIT."""
    if not r_list:
        raise ValueError("r_list must contain at least one signal")

    # Use the shortest signal length to build Hankel matrices with equal shapes
    N = min(len(r) for r in r_list)
    L = N // 2
    if L <= 0:
        raise ValueError("signals are too short for ESPRIT")

    hankels = [_hankel_matrix(np.asarray(r)[:N], L) for r in r_list]
    H = np.vstack(hankels)

    try:
        U, _, _ = np.linalg.svd(H, full_matrices=False)
        Us = U[:, :p]
        Us1, Us2 = Us[:-1], Us[1:]
        Psi = np.linalg.pinv(Us1) @ Us2
        lam = np.linalg.eigvals(Psi)
        dt = 1.0 / fs
        f = np.abs(np.angle(lam)) / (2 * np.pi * dt)
        zeta = -np.log(np.abs(lam)) / dt
    except np.linalg.LinAlgError:
        f, zeta = _esprit_freqs_and_decay(r_list[0][:N], fs, p)
    if (not np.all(np.isfinite(f)) or not np.any(f)):
        f, zeta = _esprit_freqs_and_decay(r_list[0][:N], fs, p)
    logger.debug("ESPRIT raw freqs: %s", np.round(f / GHZ, 3))
    logger.debug("ESPRIT raw zeta: %s", np.round(zeta, 3))
    return f, zeta

