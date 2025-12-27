from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit

from . import GHZ, logger


@dataclass(slots=True)
class CloseFreqCandidate:
    """Две близкие частоты, найденные в спектре LF.

    f1, f2   – частоты (Гц)
    sigma1/2 – оценка ширины гауссиан (Гц)
    amp1/2   – амплитуды гауссиан
    offset   – постоянная составляющая
    """

    f1: float
    f2: float
    sigma1: float
    sigma2: float
    amp1: float
    amp2: float
    offset: float

    def tau_estimates(self) -> tuple[float | None, float | None]:
        return tuple(_tau_from_sigma(s) for s in (self.sigma1, self.sigma2))  # type: ignore[return-value]

    def freq_bounds(self, margin: float = 0.01) -> tuple[tuple[float, float], tuple[float, float]]:
        return (
            (self.f1 * (1 - margin), self.f1 * (1 + margin)),
            (self.f2 * (1 - margin), self.f2 * (1 + margin)),
        )

    def evaluate(self, freqs_hz: NDArray) -> NDArray:
        freqs_GHz = freqs_hz / GHZ
        m1 = self.f1 / GHZ
        m2 = self.f2 / GHZ
        s1 = self.sigma1 / GHZ
        s2 = self.sigma2 / GHZ
        return _double_gaussian(freqs_GHz, self.amp1, m1, s1, self.amp2, m2, s2, self.offset)


def _double_gaussian(freqs_GHz: NDArray, a1, m1, s1, a2, m2, s2, c):
    return (
        c
        + a1 * np.exp(-0.5 * ((freqs_GHz - m1) / s1) ** 2)
        + a2 * np.exp(-0.5 * ((freqs_GHz - m2) / s2) ** 2)
    )


def _tau_from_sigma(sigma_hz: float, *, min_tau: float = 5e-12, max_tau: float = 1e-7) -> float | None:
    if sigma_hz <= 0 or not np.isfinite(sigma_hz):
        return None
    tau = 1.0 / (2 * np.pi * sigma_hz)
    return float(np.clip(tau, min_tau, max_tau))


def _top_peaks(freqs: NDArray, amps: NDArray) -> Iterable[tuple[float, float]]:
    idx = np.argpartition(amps, -2)[-2:]
    idx_sorted = idx[np.argsort(amps[idx])[::-1]]
    for i in idx_sorted:
        yield float(freqs[i]), float(amps[i])


def find_close_frequency_candidate(
    freqs: NDArray,
    amps: NDArray,
    *,
    window_GHz: float = 5.0,
    min_points: int = 8,
) -> CloseFreqCandidate | None:
    if freqs.size == 0 or amps.size == 0:
        return None
    if freqs.shape != amps.shape:
        raise ValueError("freqs and amps must have the same shape")

    pk_idx = int(np.argmax(amps))
    pk_freq = float(freqs[pk_idx])
    half_win = window_GHz * GHZ
    mask = (freqs >= pk_freq - half_win) & (freqs <= pk_freq + half_win)
    if not np.any(mask):
        logger.debug("Окно для двойного пика пусто")
        return None

    f_win = freqs[mask]
    a_win = amps[mask]
    if f_win.size < min_points:
        logger.debug("Слишком мало точек в окне двойного пика: %d", f_win.size)
        return None

    f_win_GHz = f_win / GHZ
    offset0 = float(np.percentile(a_win, 10))
    peaks = list(_top_peaks(f_win, a_win))
    if len(peaks) == 0:
        return None
    (m1, a1) = peaks[0]
    if len(peaks) == 1:
        m2 = m1 + 1 * GHZ
        a2 = a1 * 0.8
    else:
        m2, a2 = peaks[1]
    # работать в ГГц для численной устойчивости
    m1_GHz = m1 / GHZ
    m2_GHz = m2 / GHZ
    sigma0 = max(0.2, window_GHz / 6)
    p0 = [a1, m1_GHz, sigma0, a2, m2_GHz, sigma0, offset0]
    lower = [0.0, m1_GHz - window_GHz, 1e-3, 0.0, m2_GHz - window_GHz, 1e-3, -np.inf]
    upper = [np.inf, m1_GHz + window_GHz, window_GHz, np.inf, m2_GHz + window_GHz, window_GHz, np.inf]

    try:
        popt, _ = curve_fit(
            _double_gaussian,
            f_win_GHz,
            a_win,
            p0=p0,
            bounds=(lower, upper),
            maxfev=8000,
        )
    except Exception as exc:  # pragma: no cover - curve_fit errors are data-dependent
        logger.debug("Не удалось аппроксимировать двойной пик: %s", exc)
        return None

    a1_fit, m1_fit, s1_fit, a2_fit, m2_fit, s2_fit, offset_fit = popt
    if a1_fit < a2_fit:
        m1_fit, m2_fit = m2_fit, m1_fit
        s1_fit, s2_fit = s2_fit, s1_fit
        a1_fit, a2_fit = a2_fit, a1_fit

    f1 = float(min(m1_fit, m2_fit) * GHZ)
    f2 = float(max(m1_fit, m2_fit) * GHZ)
    if abs(f2 - f1) > 5 * GHZ:
        logger.debug("Двойной пик отвергнут: частоты далеко друг от друга")
        return None

    sigma1 = float(abs(s1_fit) * GHZ)
    sigma2 = float(abs(s2_fit) * GHZ)
    logger.info(
        "Двойная гаусс-аппроксимация LF: f1=%.3f ГГц, f2=%.3f ГГц",
        f1 / GHZ,
        f2 / GHZ,
    )
    return CloseFreqCandidate(
        f1=f1,
        f2=f2,
        sigma1=sigma1,
        sigma2=sigma2,
        amp1=float(a1_fit),
        amp2=float(a2_fit),
        offset=float(offset_fit),
    )


__all__ = ["CloseFreqCandidate", "find_close_frequency_candidate", "_tau_from_sigma"]
