from __future__ import annotations

from typing import Iterable, Tuple, List
from collections.abc import Iterable as IterableABC

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares
from scipy.signal import find_peaks

try:  # pragma: no cover - optional dependency
    import pywt  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - handled at runtime
    pywt = None  # type: ignore[assignment]

from . import GHZ, HF_BAND, LF_BAND, logger


def _cwt_gaussian_candidates(
    t: NDArray,
    y: NDArray,
    highcut_GHz: float,
    time_cutoffs: Iterable[Tuple[float | None, float | None]] | Tuple[float | None, float | None] | None,
) -> tuple[list[float], list[float]]:
    """Estimate LF/HF frequency candidates using a CWT-based approach."""

    lf_candidates: list[float] = []
    hf_candidates: list[float] = []

    if pywt is None:
        if not getattr(_cwt_gaussian_candidates, "_pywt_warned", False):
            logger.warning("CWT candidate extraction skipped: PyWavelets is not installed")
            setattr(_cwt_gaussian_candidates, "_pywt_warned", True)
        return lf_candidates, hf_candidates

    if t.size < 4 or y.size != t.size:
        logger.debug("CWT candidates skipped: invalid input sizes (%d, %d)", t.size, y.size)
        return lf_candidates, hf_candidates

    try:
        dt = float(np.median(np.diff(t)))
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("invalid sampling step")

        max_freq_hz = float(min(highcut_GHz * GHZ, 0.5 / dt))
        if max_freq_hz <= 0:
            raise ValueError("invalid frequency range")

        # Avoid zero frequency to keep scales finite
        min_freq_hz = max(1.0 / (t[-1] - t[0]), 1e6) if t[-1] > t[0] else 1e6
        min_freq_hz = min(min_freq_hz, max_freq_hz * 0.5)
        freq_grid = np.linspace(min_freq_hz, max_freq_hz, 256)
        wavelet = "morl"
        scales = pywt.scale2frequency(wavelet, 1.0) / (freq_grid * dt)

        # Prepare time mask if cutoffs are provided.
        time_mask = np.ones_like(t, dtype=bool)
        if time_cutoffs is not None:
            try:
                cut_iter: Iterable[Tuple[float | None, float | None]]
                if isinstance(time_cutoffs, tuple):
                    cut_iter = [time_cutoffs]
                else:
                    cut_iter = list(time_cutoffs)  # type: ignore[arg-type]
                mask = np.zeros_like(t, dtype=bool)
                for window in cut_iter:
                    if window is None:
                        continue
                    if not isinstance(window, IterableABC):
                        continue
                    try:
                        start, stop = window  # type: ignore[misc]
                    except (TypeError, ValueError):
                        continue
                    start_val = float(start) if start is not None else float(t[0])
                    stop_val = float(stop) if stop is not None else float(t[-1])
                    if start_val > stop_val:
                        start_val, stop_val = stop_val, start_val
                    mask |= (t >= start_val) & (t <= stop_val)
                if mask.any():
                    time_mask = mask
            except Exception as exc:
                logger.debug("CWT time cutoff handling failed: %s", exc)

        coeffs, _ = pywt.cwt(y, scales, wavelet, sampling_period=dt)
        power = np.abs(coeffs) ** 2
        if not time_mask.any():
            time_mask = np.ones_like(t, dtype=bool)
        power = power[:, time_mask]
        spectrum = np.mean(power, axis=1)
        if not np.any(np.isfinite(spectrum)):
            logger.debug("CWT candidates skipped: non-finite spectrum")
            return lf_candidates, hf_candidates

        spectrum = np.nan_to_num(spectrum, nan=0.0, posinf=0.0, neginf=0.0)
        spec_max = float(np.max(spectrum))
        if spec_max <= 0:
            return lf_candidates, hf_candidates

        prominence = max(spec_max * 0.05, 1e-12)
        distance = max(1, spectrum.size // 50)
        peaks, _ = find_peaks(spectrum, prominence=prominence, distance=distance)
        if peaks.size == 0:
            peaks = np.array([int(np.argmax(spectrum))])

        peak_order = np.argsort(spectrum[peaks])[::-1]
        top_peaks = peaks[peak_order[:2]]

        initial_centers = freq_grid[top_peaks]
        if initial_centers.size == 1:
            initial_centers = np.array([initial_centers[0], initial_centers[0]])

        width_guess = max((max_freq_hz - min_freq_hz) / 20.0, min_freq_hz * 0.1)
        p0 = [
            spectrum[top_peaks[0]] if top_peaks.size else spec_max,
            initial_centers[0],
            width_guess,
            spectrum[top_peaks[1]] if top_peaks.size > 1 else spec_max * 0.8,
            initial_centers[1],
            width_guess,
        ]

        lower = [0.0, min_freq_hz, width_guess * 1e-3, 0.0, min_freq_hz, width_guess * 1e-3]
        upper = [
            spec_max * 10,
            max_freq_hz,
            (max_freq_hz - min_freq_hz) * 2,
            spec_max * 10,
            max_freq_hz,
            (max_freq_hz - min_freq_hz) * 2,
        ]

        def _gauss_sum(params: NDArray, freq: NDArray) -> NDArray:
            a1, m1, s1, a2, m2, s2 = params
            g1 = a1 * np.exp(-0.5 * ((freq - m1) / max(s1, 1e-9)) ** 2)
            g2 = a2 * np.exp(-0.5 * ((freq - m2) / max(s2, 1e-9)) ** 2)
            return g1 + g2

        def _residuals(params: NDArray) -> NDArray:
            return _gauss_sum(params, freq_grid) - spectrum

        fitted_freqs: list[float] = []
        try:
            res = least_squares(_residuals, p0, bounds=(lower, upper), max_nfev=200)
            if res.success and res.x.size == 6 and np.all(np.isfinite(res.x)):
                fitted_freqs = [float(res.x[1]), float(res.x[4])]
        except Exception as exc:
            logger.debug("CWT gaussian fit failed: %s", exc)

        candidate_freqs = list(initial_centers.astype(float))
        candidate_freqs.extend(fitted_freqs)

        def _append_local(target: list[float], freq_hz: float) -> None:
            if not np.isfinite(freq_hz) or freq_hz <= 0:
                return
            for existing in target:
                if abs(existing - freq_hz) < 20e6:
                    return
            target.append(freq_hz)

        for freq_hz in candidate_freqs:
            if LF_BAND[0] <= freq_hz <= min(LF_BAND[1], max_freq_hz):
                _append_local(lf_candidates, float(freq_hz))
            if HF_BAND[0] <= freq_hz <= min(HF_BAND[1], max_freq_hz):
                _append_local(hf_candidates, float(freq_hz))

        lf_candidates.sort()
        hf_candidates.sort()
        logger.debug(
            "CWT candidates: LF=%s, HF=%s (ГГц)",
            np.round(np.array(lf_candidates) / GHZ, 3) if lf_candidates else [],
            np.round(np.array(hf_candidates) / GHZ, 3) if hf_candidates else [],
        )
    except Exception as exc:
        logger.warning("CWT candidate extraction failed: %s", exc)

    return lf_candidates, hf_candidates


def _top2_nearest(freqs: NDArray, zetas: NDArray, f0: float) -> List[Tuple[float, float | None]]:
    if freqs.size == 0:
        return []
    idx = np.argsort(np.abs(freqs - f0))[:4]
    return [(float(freqs[i]), float(zetas[i])) for i in idx]

