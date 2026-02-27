from __future__ import annotations

from typing import Tuple
import numpy as np
from numpy.typing import NDArray
from scipy.signal import welch, find_peaks, get_window

from . import GHZ, logger


def burg(x: NDArray, order: int = 8):
    """Простейшая реализация AR‑Burg."""
    x = np.asarray(x, dtype=np.float64)
    N = x.size
    if order >= N:
        raise ValueError("order must be < len(x)")
    ef = eb = x[1:].copy()
    a = np.zeros(order + 1)
    a[0] = 1.0
    E = np.dot(x, x) / N
    for m in range(1, order + 1):
        num = -2.0 * np.dot(eb, ef.conj())
        den = np.dot(ef, ef.conj()) + np.dot(eb, eb.conj())
        k = num / den if den else 0.0
        a_prev = a.copy()
        a[1:m + 1] += k * a_prev[m - 1::-1]
        ef, eb = ef[1:] + k * eb[1:], eb[:-1] + k * ef[:-1]
        E *= 1 - k.real ** 2
    return -a[1:], E


def _fft_spectrum(
    sig: np.ndarray,
    fs: float,
    *,
    window_name: str = "hamming",
    df_target_GHz: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Амплитудный спектр (ASD) сигнала."""
    N = sig.size
    sig = sig - sig.mean()
    if window_name:
        win = get_window(window_name, N)
        sig = sig * win
        win_sum_sq = np.sum(win ** 2)
    else:
        win_sum_sq = float(N)
    df_target_Hz = df_target_GHz * GHZ
    N_req = int(np.ceil(fs / df_target_Hz))
    N_fft = max(N_req, N)
    sig = np.pad(sig, (0, N_fft - N), mode="constant")
    F = np.fft.rfft(sig, n=N_fft)
    freqs = np.fft.rfftfreq(N_fft, d=1 / fs)
    psd = 2.0 * np.abs(F) ** 2 / (fs * win_sum_sq + 1e-16)
    asd = np.sqrt(psd)
    asd = np.nan_to_num(asd)
    return freqs, asd


def _peak_in_band(
    freqs: np.ndarray,
    amps: np.ndarray,
    fmin_GHz: float,
    fmax_GHz: float,
    *,
    max_expansions: int = 1,
    expansion_step_GHz: float = 2.0,
) -> float | None:
    """Return peak frequency inside band or ``None`` if absent.

    If the strongest point lies closer than 5 % of the band width to a
    boundary, the band is considered unreliable. The search is then repeated
    with an expanded range. Expansion step and maximum number of expansions
    are configurable via ``expansion_step_GHz`` and ``max_expansions``.
    """

    expansions = 0
    while True:
        logger.debug("FFT peak search: %.1f–%.1f ГГц", fmin_GHz, fmax_GHz)
        mask = (freqs >= fmin_GHz * GHZ) & (freqs <= fmax_GHz * GHZ)
        if not mask.any():
            logger.debug(
                "No data in range %.1f–%.1f ГГц", fmin_GHz, fmax_GHz,
            )
            logger.debug(
                "Total expansions: %d, final range %.1f–%.1f ГГц",
                expansions,
                fmin_GHz,
                fmax_GHz,
            )
            return None
        f_band = freqs[mask]
        a_band = amps[mask]
        if a_band.size < 3:
            logger.debug(
                "Not enough points in range %.1f–%.1f ГГц", fmin_GHz, fmax_GHz,
            )
            logger.debug(
                "Total expansions: %d, final range %.1f–%.1f ГГц",
                expansions,
                fmin_GHz,
                fmax_GHz,
            )
            return None
        med = np.median(a_band)
        std = np.std(a_band)
        df = f_band[1] - f_band[0]
        dist = int(0.3 * GHZ / df)

        found_peak = False
        for k in (0.3, 0.05):
            thr = med + k * std
            prom = k * std
            pk, props = find_peaks(
                a_band,
                height=thr,
                prominence=prom,
                distance=max(1, dist),
                plateau_size=True,
            )
            logger.debug(
                "find_peaks: height>=%.3g, prom>=%.3g -> %d peaks",
                thr,
                prom,
                pk.size,
            )
            if pk.size:
                heights = props.get("peak_heights")
                idx = int(np.argmax(heights))
                if props.get("plateau_sizes", np.zeros(len(pk)))[idx] > 1:
                    left = int(props["left_edges"][idx])
                    right = int(props["right_edges"][idx]) - 1
                    best_idx = (left + right) // 2
                else:
                    best_idx = pk[idx]
                f_best = float(f_band[best_idx])
                found_peak = True
                break

        if not found_peak:
            max_idx = int(np.argmax(a_band))
            f_best = float(f_band[max_idx])
            logger.debug(
                "no peaks found, fallback to max: f=%.3f ГГц, amp=%.3g",
                f_best / GHZ,
                a_band[max_idx],
            )

        band_width = f_band[-1] - f_band[0]
        margin = 0.05 * band_width
        if band_width > 0 and (
            (f_best - f_band[0] < margin) or (f_band[-1] - f_best < margin)
        ):
            if expansions < max_expansions:
                expansions += 1
                fmin_GHz -= expansion_step_GHz
                fmax_GHz += expansion_step_GHz
                logger.debug(
                    "Peak near boundary, expanding search to %.1f–%.1f ГГц (attempt %d/%d)",
                    fmin_GHz,
                    fmax_GHz,
                    expansions,
                    max_expansions,
                )
                continue
            logger.debug(
                "Peak near boundary even after %d expansions: %.1f–%.1f ГГц",
                expansions,
                fmin_GHz,
                fmax_GHz,
            )
            logger.debug(
                "Total expansions: %d, final range %.1f–%.1f ГГц",
                expansions,
                fmin_GHz,
                fmax_GHz,
            )
            return None

        if found_peak:
            logger.debug(
                "Peak found at %.3f ГГц within %.1f–%.1f ГГц",
                f_best / GHZ,
                fmin_GHz,
                fmax_GHz,
            )
        else:
            logger.debug(
                "Selected fallback peak %.3f ГГц within %.1f–%.1f ГГц",
                f_best / GHZ,
                fmin_GHz,
                fmax_GHz,
            )
        logger.debug(
            "Total expansions: %d, final range %.1f–%.1f ГГц",
            expansions,
            fmin_GHz,
            fmax_GHz,
        )
        return f_best


def _fallback_peak(
    t: NDArray,
    y: NDArray,
    fs: float,
    f_range: Tuple[float, float],
    f_rough: float,
    avoid: float | None = None,
    df_min: float = 0.5 * GHZ,
    order_burg: int = 8,
    n_avg_fft: int = 4,
) -> float | None:
    logger.debug(
        "Fallback search: range=[%.1f, %.1f] ГГц, avoid=%s",
        f_range[0] / GHZ,
        f_range[1] / GHZ,
        None if avoid is None else f"{avoid / GHZ:.3f}",
    )

    nperseg = len(y) // 4
    if nperseg < 8:
        logger.debug("Signal too short for fallback peak search: %d samples", len(y))
        return None
    try:
        ar, _ = burg(y - y.mean(), order=order_burg)
        roots = np.roots(np.r_[1, -ar])
        root = roots[np.argmax(np.abs(roots))]
        f_burg = abs(np.angle(root)) * fs / (2 * np.pi)
        logger.debug("Burg estimate: %.3f ГГц", f_burg / GHZ)
        if (
            f_burg > 0
            and np.isfinite(f_burg)
            and f_range[0] <= f_burg <= f_range[1]
            and (avoid is None or abs(f_burg - avoid) >= df_min)
        ):
            return float(f_burg)
    except Exception as exc:
        logger.debug("Burg failed: %s", exc)

    def _welch_peak() -> tuple[float | None, float]:
        f, P = welch(y, fs=fs, nperseg=nperseg, detrend="constant", scaling="density")
        mask = (f >= f_range[0]) & (f <= f_range[1])
        if avoid is not None:
            mask &= np.abs(f - avoid) >= df_min
        if not np.any(mask):
            return None, 0.0
        idx = np.argmax(P[mask])
        return float(f[mask][idx]), float(P[mask][idx])

    def _avg_fft_peak() -> tuple[float | None, float]:
        seg_len = len(y) // n_avg_fft
        if seg_len < 8:
            return None, 0.0
        trimmed = y[: seg_len * n_avg_fft]
        segs = trimmed.reshape(n_avg_fft, seg_len)
        spec = np.abs(np.fft.rfft(segs, axis=1))
        avg_spec = np.mean(spec, axis=0)
        freqs = np.fft.rfftfreq(seg_len, d=1 / fs)
        mask = (freqs >= f_range[0]) & (freqs <= f_range[1])
        if avoid is not None:
            mask &= np.abs(freqs - avoid) >= df_min
        if not np.any(mask):
            return None, 0.0
        idx = np.argmax(avg_spec[mask])
        return float(freqs[mask][idx]), float(avg_spec[mask][idx])

    fw, pw = _welch_peak()
    if fw is not None:
        logger.debug("Welch estimate: %.3f ГГц", fw / GHZ)
    fa, pa = _avg_fft_peak()
    if fa is not None:
        logger.debug("AvgFFT estimate: %.3f ГГц", fa / GHZ)

    candidates: list[tuple[float, float, float]] = []
    if fw is not None:
        candidates.append((fw, pw, abs(fw - f_rough)))
    if fa is not None:
        candidates.append((fa, pa, abs(fa - f_rough)))
    if not candidates:
        return None
    # choose by amplitude, then by closeness to rough estimate
    candidates.sort(key=lambda c: (c[1], -c[2]), reverse=True)
    return candidates[0][0]

