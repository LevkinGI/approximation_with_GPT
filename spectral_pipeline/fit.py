from __future__ import annotations

from typing import Iterable, Tuple, List, Optional
from collections.abc import Iterable as IterableABC
from pathlib import Path
import math
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares
from scipy.signal import welch, find_peaks, get_window

try:  # pragma: no cover - optional dependency
    import pywt  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - handled at runtime
    pywt = None  # type: ignore[assignment]

from . import DataSet, FittingResult, GHZ, PI, logger, LF_BAND, HF_BAND


def _resolve_tau_bounds(
    zeta: float | None,
    tau_init_fallback: float,
    default_lo: float,
    default_hi: float,
) -> tuple[float, float, float]:
    """Return (tau_init, tau_lo, tau_hi) with positive, sane bounds.

    If ``zeta`` is positive and finite, derive ``tau`` from it and provide a
    narrow bracket. Otherwise keep ``tau_init_fallback`` (or replace it with the
    geometric mean of the defaults if it's invalid) and use the default bounds.
    """

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


def _cwt_gaussian_candidates(
    t: NDArray,
    y: NDArray,
    highcut_GHz: float,
    time_cutoffs: Iterable[Tuple[float | None, float | None]] | Tuple[float | None, float | None] | None,
) -> tuple[list[float], list[float]]:
    """Estimate LF/HF frequency candidates using a CWT-based approach.

    Parameters
    ----------
    t, y : ndarray
        Time axis (seconds) and corresponding signal samples.
    highcut_GHz : float
        Upper frequency bound (in GHz) for the search grid.
    time_cutoffs : iterable or tuple or None
        Optional list of ``(start, stop)`` pairs limiting the time ranges to
        analyse. ``None`` keeps the full signal. ``start``/``stop`` can be
        ``None`` to indicate the beginning or end of the record respectively.

    Returns
    -------
    tuple[list[float], list[float]]
        Candidate frequencies in Hz for LF and HF bands respectively.
    """

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

# Maximum acceptable fitting cost. Pairs with higher cost are rejected
# and treated as unsuccessful.
MAX_COST = 100


def _load_guess(directory: Path, field_mT: int, temp_K: int) -> tuple[float, float] | None:
    """Load first-approximation frequencies if file exists.

    Returns tuple (f_lf_Hz, f_hf_Hz) or None.
    """
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
    except Exception as exc:
        logger.warning("Не удалось загрузить %s: %s", path, exc)
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


def _fft_spectrum(sig: np.ndarray, fs: float, *, window_name: str = "hamming",
                  df_target_GHz: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
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
        logger.info("FFT peak search: %.1f–%.1f ГГц", fmin_GHz, fmax_GHz)
        mask = (freqs >= fmin_GHz * GHZ) & (freqs <= fmax_GHz * GHZ)
        if not mask.any():
            logger.info(
                "No data in range %.1f–%.1f ГГц", fmin_GHz, fmax_GHz,
            )
            logger.info(
                "Total expansions: %d, final range %.1f–%.1f ГГц",
                expansions,
                fmin_GHz,
                fmax_GHz,
            )
            return None
        f_band = freqs[mask]
        a_band = amps[mask]
        if a_band.size < 3:
            logger.info(
                "Not enough points in range %.1f–%.1f ГГц", fmin_GHz, fmax_GHz,
            )
            logger.info(
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
                logger.info(
                    "Peak near boundary, expanding search to %.1f–%.1f ГГц (attempt %d/%d)",
                    fmin_GHz,
                    fmax_GHz,
                    expansions,
                    max_expansions,
                )
                continue
            logger.info(
                "Peak near boundary even after %d expansions: %.1f–%.1f ГГц",
                expansions,
                fmin_GHz,
                fmax_GHz,
            )
            logger.info(
                "Total expansions: %d, final range %.1f–%.1f ГГц",
                expansions,
                fmin_GHz,
                fmax_GHz,
            )
            return None

        if found_peak:
            logger.info(
                "Peak found at %.3f ГГц within %.1f–%.1f ГГц",
                f_best / GHZ,
                fmin_GHz,
                fmax_GHz,
            )
        else:
            logger.info(
                "Selected fallback peak %.3f ГГц within %.1f–%.1f ГГц",
                f_best / GHZ,
                fmin_GHz,
                fmax_GHz,
            )
        logger.info(
            "Total expansions: %d, final range %.1f–%.1f ГГц",
            expansions,
            fmin_GHz,
            fmax_GHz,
        )
        return f_best


def _fallback_peak(t: NDArray, y: NDArray, fs: float, f_range: Tuple[float, float],
                   f_rough: float, avoid: float | None = None, df_min: float = 0.5 * GHZ,
                   order_burg: int = 8, n_avg_fft: int = 4) -> float | None:
    logger.debug(
        "Fallback search: range=[%.1f, %.1f] ГГц, avoid=%s", f_range[0]/GHZ,
        f_range[1]/GHZ, None if avoid is None else f"{avoid/GHZ:.3f}")

    nperseg = len(y) // 4
    if nperseg < 8:
        logger.info("Signal too short for fallback peak search: %d samples", len(y))
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
        f, P = welch(y, fs=fs, nperseg=nperseg,
                     detrend='constant', scaling='density')
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
        trimmed = y[:seg_len * n_avg_fft]
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


def _top2_nearest(freqs: NDArray, zetas: NDArray, f0: float
                  ) -> List[Tuple[float, Optional[float]]]:
    if freqs.size == 0:
        return []
    idx = np.argsort(np.abs(freqs - f0))[:4]
    return [(float(freqs[i]), float(zetas[i])) for i in idx]


def _hankel_matrix(x: NDArray, L: int) -> NDArray:
    N = len(x)
    if L <= 0 or L >= N:
        raise ValueError("L must satisfy 0 < L < N")
    return np.lib.stride_tricks.sliding_window_view(x, L).T


def multichannel_esprit(r_list: List[NDArray], fs: float, p: int = 6
                        ) -> Tuple[NDArray, NDArray]:
    """Estimate modal frequencies and decays using multichannel ESPRIT.

    Parameters
    ----------
    r_list : list of ndarray
        List of equally sampled signals. All signals are assumed to share the
        same sampling rate ``fs``.
    fs : float
        Sampling frequency in Hz.
    p : int, optional
        Number of components (model order) to retain, by default 6.

    Returns
    -------
    Tuple[NDArray, NDArray]
        Frequencies and decay rates of the estimated modes.
    """
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


def _esprit_freqs_and_decay(r: NDArray, fs: float, p: int = 6
                            ) -> Tuple[NDArray, NDArray]:
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


def _core_signal(t: NDArray, A1, A2, tau1, tau2, f1, f2, phi1, phi2) -> NDArray:
    return (
        A1 * np.exp(-t / tau1) * np.cos(2 * np.pi * f1 * t + phi1) +
        A2 * np.exp(-t / tau2) * np.cos(2 * np.pi * f2 * t + phi2)
    )


def _single_sine_refine(t: NDArray, y: NDArray, f0: float
                        ) -> tuple[float, float, float, float]:
    A0 = 0.5 * (y.max() - y.min())
    tau0 = (t[-1] - t[0]) / 3
    p0 = [A0, f0, 0.0, tau0, y.mean()]
    lo = np.array([0.0, LF_BAND[0], -np.pi, tau0/10, y.min() - abs(np.ptp(y))])
    hi = np.array([3*A0, HF_BAND[1], np.pi, tau0*10, y.max() + abs(np.ptp(y))])

    def model(p):
        A, f, phi, tau, C = p
        return A * np.exp(-t/tau) * np.cos(2*np.pi*f*t + phi) + C

    sol = least_squares(lambda p: model(p) - y, p0, bounds=(lo, hi), method="trf")
    A_est, f_est, phi_est, tau_est, _ = sol.x
    return f_est, phi_est, A_est, tau_est


def fit_pair(ds_lf: DataSet, ds_hf: DataSet,
             freq_bounds: tuple[tuple[float, float], tuple[float, float]] | None = None):
    t_lf, y_lf = ds_lf.ts.t, ds_lf.ts.s
    t_hf, y_hf = ds_hf.ts.t, ds_hf.ts.s

    def _piecewise_time_weights(t: np.ndarray) -> np.ndarray:
        if t.size == 0:
            return np.ones_like(t)
        t_min = t.min()
        t_len = t.max() - t_min
        if t_len <= 0:
            return np.ones_like(t)
        borders = t_min + np.array([1, 2]) * t_len / 3
        w = np.ones_like(t)
        w[t >= borders[0]] = 0.8
        w[t >= borders[1]] = 0.5
        return w

    w_lf = _piecewise_time_weights(t_lf)

    f1_init = ds_lf.f1_init
    f2_init = ds_hf.f2_init
    logger.debug(
        "Начальные оценки: f1=%.3f ГГц, f2=%.3f ГГц", f1_init/GHZ, f2_init/GHZ)

    _, phi1_init, A1_init, tau1_init = _single_sine_refine(t_lf, y_lf, f1_init)
    tau1_init, tau1_lo, tau1_hi = _resolve_tau_bounds(
        ds_lf.zeta1, tau1_init, 5e-11, 5e-9
    )

    proto_lf_hf = A1_init * np.exp(-t_hf / tau1_init) * np.cos(2 * PI * f1_init * t_hf + phi1_init)

    def _rms(signal: NDArray) -> float:
        if signal.size == 0:
            return 0.0
        centered = signal - float(np.mean(signal))
        return float(np.sqrt(np.mean(centered ** 2)))

    hf_scale = 1.0
    hf_target_amp = _rms(y_hf)
    proto_amp = _rms(proto_lf_hf)
    if proto_amp > 0 and np.isfinite(hf_target_amp):
        hf_scale = hf_target_amp / proto_amp
        if not np.isfinite(hf_scale) or hf_scale <= 0:
            hf_scale = 1.0
        proto_lf_hf = proto_lf_hf * hf_scale

    A1_init *= hf_scale

    _, phi2_init, A2_init, tau2_init = _single_sine_refine(t_hf, y_hf - proto_lf_hf, f2_init)
    tau2_init, tau2_lo, tau2_hi = _resolve_tau_bounds(
        ds_hf.zeta2, tau2_init, 5e-12, 5e-10
    )

    k_lf_init = 1
    k_hf_init = 1
    C_lf_init = np.mean(y_lf)
    C_hf_init = np.mean(y_hf)

    p0 = np.array([
        k_lf_init, k_hf_init,
        C_lf_init, C_hf_init,
        A1_init,    A2_init,
        tau1_init,  tau2_init,
        f1_init,    f2_init,
        phi1_init,  phi2_init
    ])

    if freq_bounds is None:
        f1_lo, f1_hi = f1_init * 0.9, f1_init * 1.2
        f2_lo, f2_hi = f2_init * 0.9, f2_init * 1.2
    else:
        (f1_lo, f1_hi), (f2_lo, f2_hi) = freq_bounds

    lo = np.array([
        0.5, 0.5,
        C_lf_init - np.std(y_lf), C_hf_init - np.std(y_hf),
        0.0, 0.0,
        tau1_lo, tau2_lo,
        f1_lo, f2_lo,
        -PI, -PI
    ])
    hi = np.array([
        2, 2,
        C_lf_init + np.std(y_lf), C_hf_init + np.std(y_hf),
        A1_init * 2, A2_init * 2,
        tau1_hi, tau2_hi,
        f1_hi, f2_hi,
        PI, PI
    ])

    def residuals(p):
        (k_lf, k_hf, C_lf, C_hf,
         A1, A2, tau1, tau2,
         f1_, f2_, phi1_, phi2_) = p

        core_lf = _core_signal(t_lf, A1, A2, tau1, tau2, f1_, f2_, phi1_, phi2_)
        core_hf = _core_signal(t_hf, A1, A2, tau1, tau2, f1_, f2_, phi1_, phi2_)
        res_lf = w_lf * (k_lf * core_lf + C_lf - y_lf)
        res_hf = k_hf * core_hf + C_hf - y_hf

        # Normalize channel residuals so that the sum of squares corresponds to
        # the mean squared error for each channel individually.
        n_lf = y_lf.size
        n_hf = y_hf.size
        if n_lf:
            res_lf = res_lf / math.sqrt(n_lf)
        if n_hf:
            res_hf = res_hf / math.sqrt(n_hf)

        return np.concatenate([res_lf, res_hf])

    sol = least_squares(
        residuals,
        p0,
        bounds=(lo, hi),
        method='trf',
        ftol=1e-15, xtol=1e-15, gtol=1e-15,
        max_nfev=5000,
        loss='soft_l1',
        f_scale=0.1,
        x_scale='jac',
        verbose=0,
    )
    p = sol.x
    cost = sol.cost
    m, n = sol.jac.shape
    try:
        cov = np.linalg.inv(sol.jac.T @ sol.jac) * 2 * cost / max(m - n, 1)
    except np.linalg.LinAlgError:
        cov = np.full((n, n), np.nan)

    idx_f1 = 8
    idx_f2 = 9
    sigma_f1 = math.sqrt(abs(cov[idx_f1, idx_f1]))
    sigma_f2 = math.sqrt(abs(cov[idx_f2, idx_f2]))

    (k_lf, k_hf, C_lf, C_hf,
      A1, A2, tau1, tau2,
      f1_fin, f2_fin, phi1_fin, phi2_fin) = p
    logger.debug(
        "Результат: f1=%.3f±%.3f ГГц, f2=%.3f±%.3f ГГц, cost=%.3e",
        f1_fin/GHZ, sigma_f1/GHZ, f2_fin/GHZ, sigma_f2/GHZ, cost)

    return FittingResult(
        f1=f1_fin,
        f2=f2_fin,
        zeta1=1/tau1,
        zeta2=1/tau2,
        phi1=phi1_fin,
        phi2=phi2_fin,
        A1=A1,
        A2=A2,
        k_lf=k_lf,
        k_hf=k_hf,
        C_lf=C_lf,
        C_hf=C_hf,
        f1_err=sigma_f1,
        f2_err=sigma_f2,
        cost=cost,
    ), cost


def process_pair(
    ds_lf: DataSet,
    ds_hf: DataSet,
    *,
    use_theory_guess: bool = True,
) -> Optional[FittingResult]:
    logger.info("Обработка пары T=%d K, H=%d mT", ds_lf.temp_K, ds_lf.field_mT)
    tau_guess_lf, tau_guess_hf = 3e-10, 3e-11
    t_lf, y_lf = ds_lf.ts.t, ds_lf.ts.s
    t_hf, y_hf = ds_hf.ts.t, ds_hf.ts.s

    def _prepare_signals() -> tuple[NDArray, NDArray, float]:
        """Return LF/HF signals aligned to a common sampling rate."""
        spec_lf = y_lf - np.mean(y_lf)
        spec_hf = y_hf - np.mean(y_hf)
        fs_lf = ds_lf.ts.meta.fs
        fs_hf = ds_hf.ts.meta.fs
        if abs(fs_lf - fs_hf) > 1e-9:
            fs_common = min(fs_lf, fs_hf)
            t_end = min(t_lf[-1], t_hf[-1])
            t_common = np.arange(0.0, t_end, 1.0 / fs_common)
            spec_lf_c = np.interp(t_common, t_lf, spec_lf)
            spec_hf_c = np.interp(t_common, t_hf, spec_hf)
        else:
            fs_common = fs_lf
            n = min(spec_lf.size, spec_hf.size)
            spec_lf_c = spec_lf[:n]
            spec_hf_c = spec_hf[:n]
        return spec_lf_c, spec_hf_c, fs_common

    def _search_candidates() -> tuple[list[tuple[float, Optional[float]]],
                                      list[tuple[float, Optional[float]]],
                                      tuple[tuple[float, float], tuple[float, float]] | None]:
        f1_rough, phi1, A1, tau1 = _single_sine_refine(t_lf, y_lf, f0=10 * GHZ)
        proto_lf = A1 * np.exp(-t_hf / tau1) * np.cos(2 * np.pi * f1_rough * t_hf + phi1)
        residual = y_hf - proto_lf
        f2_rough, _, _, _ = _single_sine_refine(t_hf, residual, f0=40 * GHZ)
        logger.debug("Rough estimates: f1=%.3f ГГц, f2=%.3f ГГц",
                     f1_rough/GHZ, f2_rough/GHZ)
        spec_lf_c, spec_hf_c, fs_common = _prepare_signals()
        f_all, zeta_all = multichannel_esprit([spec_lf_c, spec_hf_c], fs_common)
        mask_hf = (
            (zeta_all > -5e8)
            & (HF_BAND[0] <= f_all)
            & (f_all <= HF_BAND[1])
        )
        logger.debug("ESPRIT HF filtered: %s", np.round(f_all[mask_hf] / GHZ, 3))
        if not np.any(mask_hf):
            mask_hf = (HF_BAND[0] <= f_all) & (f_all <= HF_BAND[1])
            logger.debug("ESPRIT HF relaxed: %s", np.round(f_all[mask_hf] / GHZ, 3))
        hf_c: list[tuple[float, Optional[float]]] = []
        if np.any(mask_hf):
            hf_c.extend(
                [
                    (float(f), float(z) if z > 0 else None)
                    for f, z in zip(f_all[mask_hf], zeta_all[mask_hf])
                ]
            )
        else:
            logger.warning(
                f"({ds_hf.temp_K}, {ds_hf.field_mT}): ESPRIT не дал HF-кандидатов"
            )

        # range_hf = (
        #     (f2_rough - 5 * GHZ, f2_rough + 5 * GHZ)
        #     if f2_rough is not None
        #     else HF_BAND
        # )
        range_hf = HF_BAND
        logger.info(
            "HF fallback range: %.1f–%.1f ГГц",
            range_hf[0] / GHZ,
            range_hf[1] / GHZ,
        )
        f2_fallback = _fallback_peak(
            t_hf,
            y_hf,
            ds_hf.ts.meta.fs,
            range_hf,
            f2_rough if f2_rough is not None else 0.0,
        )
        if f2_fallback is not None:
            if all(abs(f - f2_fallback) >= 20e6 for f, _ in hf_c):
                hf_c.append((float(f2_fallback), None))
                logger.info(
                    "(%d, %d): добавлен HF-кандидат %.3f ГГц методом fallback",
                    ds_hf.temp_K,
                    ds_hf.field_mT,
                    f2_fallback / GHZ,
                )
        elif not hf_c:
            raise RuntimeError("HF-тон не найден")
        mask_lf = (
            (zeta_all > 0)
            & (LF_BAND[0] <= f_all)
            & (f_all <= LF_BAND[1])
        )
        logger.debug("ESPRIT LF filtered: %s", np.round(f_all[mask_lf] / GHZ, 3))
        if np.any(mask_lf):
            lf_c = _top2_nearest(f_all[mask_lf], zeta_all[mask_lf], f1_rough)
        else:
            logger.warning(
                f"({ds_lf.temp_K}, {ds_lf.field_mT}): ESPRIT не дал LF-кандидатов"
            )
            lf_c = []

        # range_lf = (
        #     (f1_rough - 5 * GHZ, f1_rough + 5 * GHZ)
        #     if f1_rough is not None
        #     else LF_BAND
        # )
        range_lf = LF_BAND
        logger.info(
            "LF fallback range: %.1f–%.1f ГГц",
            range_lf[0] / GHZ,
            range_lf[1] / GHZ,
        )
        f1_fallback = _fallback_peak(
            t_lf,
            y_lf,
            ds_lf.ts.meta.fs,
            range_lf,
            f1_rough if f1_rough is not None else 0.0,
            avoid=hf_c[0][0] if hf_c else None,
        )
        if f1_fallback is not None:
            if all(abs(f - f1_fallback) >= 20e6 for f, _ in lf_c):
                lf_c.append((float(f1_fallback), None))
                logger.info(
                    "(%d, %d): добавлен LF-кандидат %.3f ГГц методом fallback",
                    ds_lf.temp_K,
                    ds_lf.field_mT,
                    f1_fallback / GHZ,
                )
        elif not lf_c:
            raise RuntimeError("LF-тон не найден")
        return lf_c, hf_c, None

    guess = None
    if use_theory_guess and ds_lf.root:
        guess = _load_guess(ds_lf.root, ds_lf.field_mT, ds_lf.temp_K)

    if guess is not None:
        f1_guess, f2_guess = guess
        logger.info(
            "(%d, %d): использованы предварительные оценки f1=%.3f ГГц, f2=%.3f ГГц",
            ds_lf.temp_K, ds_lf.field_mT, f1_guess/GHZ, f2_guess/GHZ)
        spec_lf_c, spec_hf_c, fs_common = _prepare_signals()
        f_all, zeta_all = multichannel_esprit([spec_lf_c, spec_hf_c], fs_common)
        mask_hf = (
            (zeta_all > -5e8)
            & (HF_BAND[0] <= f_all)
            & (f_all <= HF_BAND[1])
            & (np.abs(f_all - f2_guess) <= 7 * GHZ)
        )
        logger.debug("ESPRIT HF filtered: %s", np.round(f_all[mask_hf] / GHZ, 3))
        if not np.any(mask_hf):
            mask_hf = (HF_BAND[0] <= f_all) & (f_all <= HF_BAND[1])
            logger.debug("ESPRIT HF relaxed: %s", np.round(f_all[mask_hf] / GHZ, 3))
        hf_cand = _top2_nearest(f_all[mask_hf], np.maximum(zeta_all[mask_hf], 0.0), f2_guess) if np.any(mask_hf) else []
        if not hf_cand:
            logger.warning(
                f"({ds_hf.temp_K}, {ds_hf.field_mT}): ESPRIT не дал HF-кандидатов"
            )

        range_hf = (f2_guess - 5 * GHZ, f2_guess + 5 * GHZ)
        logger.info(
            "HF fallback range: %.1f–%.1f ГГц",
            range_hf[0] / GHZ,
            range_hf[1] / GHZ,
        )
        f2_fallback = _fallback_peak(
            t_hf,
            y_hf,
            ds_hf.ts.meta.fs,
            range_hf,
            f2_guess,
        )
        if f2_fallback is not None and all(
            abs(f - f2_fallback) >= 20e6 for f, _ in hf_cand
        ):
            hf_cand.append((float(f2_fallback), None))
            logger.info(
                "(%d, %d): добавлен HF-кандидат %.3f ГГц методом fallback",
                ds_hf.temp_K,
                ds_hf.field_mT,
                f2_fallback / GHZ,
            )
        elif not hf_cand:
            raise RuntimeError("HF-тон не найден")
        mask_lf = (
            (zeta_all > 0)
            & (LF_BAND[0] <= f_all)
            & (f_all <= LF_BAND[1])
            & (np.abs(f_all - f1_guess) <= 5 * GHZ)
        )
        logger.debug("ESPRIT LF filtered: %s", np.round(f_all[mask_lf] / GHZ, 3))
        lf_cand = _top2_nearest(f_all[mask_lf], zeta_all[mask_lf], f1_guess) if np.any(mask_lf) else []
        if not lf_cand:
            logger.warning(
                f"({ds_lf.temp_K}, {ds_lf.field_mT}): ESPRIT не дал LF-кандидатов"
            )

        range_lf = (f1_guess - 5 * GHZ, f1_guess + 5 * GHZ)
        logger.info(
            "LF fallback range: %.1f–%.1f ГГц",
            range_lf[0] / GHZ,
            range_lf[1] / GHZ,
        )
        f1_fallback = _fallback_peak(
            t_lf,
            y_lf,
            ds_lf.ts.meta.fs,
            range_lf,
            f1_guess,
            avoid=hf_cand[0][0] if hf_cand else None,
        )
        if f1_fallback is not None and all(
            abs(f - f1_fallback) >= 20e6 for f, _ in lf_cand
        ):
            lf_cand.append((float(f1_fallback), None))
            logger.info(
                "(%d, %d): добавлен LF-кандидат %.3f ГГц методом fallback",
                ds_lf.temp_K,
                ds_lf.field_mT,
                f1_fallback / GHZ,
            )
        elif not lf_cand:
            raise RuntimeError("LF-тон не найден")
        def _ensure_guess(cands, freq):
            for f, _ in cands:
                if abs(f - freq) < 20e6:
                    return
            cands.append((freq, None))
        _ensure_guess(lf_cand, f1_guess)
        _ensure_guess(hf_cand, f2_guess)
        freq_bounds = ((f1_guess - 5 * GHZ, f1_guess + 5 * GHZ),
                       (f2_guess - 5 * GHZ, f2_guess + 5 * GHZ))
    else:
        logger.info("(%d, %d): поиск предварительных оценок", ds_lf.temp_K, ds_lf.field_mT)
        lf_cand, hf_cand, freq_bounds = _search_candidates()
    fs_hf = 1.0 / float(np.mean(np.diff(t_hf)))
    fs_lf = 1.0 / float(np.mean(np.diff(t_lf)))
    freqs_fft, amps_fft = _fft_spectrum(y_hf, fs_hf)
    ds_hf.freq_fft, ds_hf.asd_fft = freqs_fft, amps_fft
    ds_lf.freq_fft, ds_lf.asd_fft = _fft_spectrum(y_lf, fs_lf)
    pk = int(np.argmax(amps_fft))
    minima = np.where(
        (np.diff(np.signbit(np.diff(amps_fft))) > 0) & (np.arange(len(amps_fft))[1:-1] > pk)
    )[0]
    start_band_HF = freqs_fft[minima[0]]/GHZ if minima.size else 20.0
    if start_band_HF > 30.0:
        start_band_HF = 30.0

    if guess is not None:
        lf_lo = (f1_guess - 5 * GHZ) / GHZ
        lf_hi = (f1_guess + 5 * GHZ) / GHZ
        hf_lo = (f2_guess - 5 * GHZ) / GHZ
        hf_hi = (f2_guess + 5 * GHZ) / GHZ
        f1_hz = _peak_in_band(freqs_fft, amps_fft, lf_lo, lf_hi)
        f2_hz = _peak_in_band(freqs_fft, amps_fft, hf_lo, hf_hi)
        range_lf = f"{lf_lo:.0f}–{lf_hi:.0f}"
        range_hf = f"{hf_lo:.0f}–{hf_hi:.0f}"
    else:
        f1_hz = _peak_in_band(freqs_fft, amps_fft, LF_BAND[0]/GHZ, LF_BAND[1]/GHZ)
        f2_hz = _peak_in_band(freqs_fft, amps_fft, start_band_HF, HF_BAND[1]/GHZ)
        range_lf = f"{LF_BAND[0]/GHZ:.0f}–{LF_BAND[1]/GHZ:.0f}"
        range_hf = f"{start_band_HF:.0f}–{HF_BAND[1]/GHZ:.0f}"

    if f1_hz is not None:
        logger.info(
            f"({ds_lf.temp_K}, {ds_lf.field_mT}): найден пик  f1 = {f1_hz/1e9:.1f} ГГц ({range_lf})")
    else:
        logger.warning(
            f"({ds_lf.temp_K}, {ds_lf.field_mT}): в полосе {range_lf} ГГц пиков не найдено")
    if f2_hz is not None:
        logger.info(
            f"({ds_lf.temp_K}, {ds_lf.field_mT}): найден пик  f2 = {f2_hz/1e9:.1f} ГГц ({range_hf})")
    else:
        logger.warning(
            f"({ds_lf.temp_K}, {ds_lf.field_mT}): в полосе {range_hf} ГГц пиков не найдено")

    def _append_unique(target_list, new_freq_hz, *, label: str, source: str) -> bool:
        if new_freq_hz is None:
            return False
        for old_f, _ in target_list:
            if abs(old_f - new_freq_hz) < 20e6:
                logger.debug(
                    "(%d, %d) %s: пропуск кандидата %.3f ГГц из %s (слишком близко к %.3f)",
                    ds_lf.temp_K,
                    ds_lf.field_mT,
                    label,
                    new_freq_hz / GHZ,
                    source,
                    old_f / GHZ,
                )
                return False
        target_list.append((float(new_freq_hz), None))
        logger.info(
            "(%d, %d) %s: добавлен кандидат %.3f ГГц (%s)",
            ds_lf.temp_K,
            ds_lf.field_mT,
            label,
            new_freq_hz / GHZ,
            source,
        )
        return True

    _append_unique(lf_cand, f1_hz, label="LF", source="FFT")
    _append_unique(hf_cand, f2_hz, label="HF", source="FFT")

    cwt_lf, _ = _cwt_gaussian_candidates(
        t_lf,
        y_lf,
        highcut_GHz=LF_BAND[1] / GHZ,
        time_cutoffs=None,
    )
    _, cwt_hf = _cwt_gaussian_candidates(
        t_hf,
        y_hf,
        highcut_GHz=HF_BAND[1] / GHZ,
        time_cutoffs=None,
    )

    for freq_hz in cwt_lf:
        if LF_BAND[0] <= freq_hz <= LF_BAND[1]:
            _append_unique(lf_cand, freq_hz, label="LF", source="CWT")

    for freq_hz in cwt_hf:
        if HF_BAND[0] <= freq_hz <= HF_BAND[1]:
            _append_unique(hf_cand, freq_hz, label="HF", source="CWT")

    logger.info("LF candidates: %s", [(round(f/GHZ,3), z) for f, z in lf_cand])
    logger.info("HF candidates: %s", [(round(f/GHZ,3), z) for f, z in hf_cand])

    seen: set[tuple[float, float]] = set()
    best_cost = np.inf
    best_fit = None
    for f1, z1 in lf_cand:
        for f2, z2 in hf_cand:
            if (f1, f2) in seen:
                continue
            if freq_bounds is not None:
                (f1_lo, f1_hi), (f2_lo, f2_hi) = freq_bounds
                if not (f1_lo <= f1 <= f1_hi and f2_lo <= f2 <= f2_hi):
                    logger.debug(
                        "Комбинация вне freq_bounds: f1=%.3f ГГц, f2=%.3f ГГц",
                        f1 / GHZ,
                        f2 / GHZ,
                    )
                    seen.add((f1, f2))
                    continue
            ds_lf.f1_init, ds_lf.zeta1 = f1, z1
            ds_hf.f2_init, ds_hf.zeta2 = f2, z2
            try:
                fit, cost = fit_pair(ds_lf, ds_hf, freq_bounds=freq_bounds)
            except Exception as exc:
                logger.debug(
                    "Неудачная попытка f1=%.3f ГГц, f2=%.3f ГГц: %s",
                    f1 / GHZ,
                    f2 / GHZ,
                    exc,
                )
            else:
                if cost < best_cost:
                    best_cost = cost
                    best_fit = fit
            finally:
                seen.add((f1, f2))
    if best_fit is None and guess is not None:
        logger.warning("(%d, %d): не удалось аппроксимировать с первым приближением, поиск альтернативы", ds_lf.temp_K, ds_lf.field_mT)
        lf_cand, hf_cand, freq_bounds = _search_candidates()
        best_cost = np.inf
        for f1, z1 in lf_cand:
            for f2, z2 in hf_cand:
                if (f1, f2) in seen:
                    continue
                if freq_bounds is not None:
                    (f1_lo, f1_hi), (f2_lo, f2_hi) = freq_bounds
                    if not (f1_lo <= f1 <= f1_hi and f2_lo <= f2 <= f2_hi):
                        logger.debug(
                            "Комбинация вне freq_bounds: f1=%.3f ГГц, f2=%.3f ГГц",
                            f1 / GHZ,
                            f2 / GHZ,
                        )
                        seen.add((f1, f2))
                        continue
                ds_lf.f1_init, ds_lf.zeta1 = f1, z1
                ds_hf.f2_init, ds_hf.zeta2 = f2, z2
                try:
                    fit, cost = fit_pair(ds_lf, ds_hf, freq_bounds=freq_bounds)
                except Exception as exc:
                    logger.debug(
                        "Неудачная попытка f1=%.3f ГГц, f2=%.3f ГГц: %s",
                        f1 / GHZ,
                        f2 / GHZ,
                        exc,
                    )
                else:
                    if cost < best_cost:
                        best_cost = cost
                        best_fit = fit
                finally:
                    seen.add((f1, f2))

    if best_fit is None:
        logger.error("(%d, %d): ни одна комбинация не аппроксимировалась", ds_lf.temp_K, ds_lf.field_mT)
        raise RuntimeError("Ни одна комбинация не аппроксимировалась")
    if best_fit.f1 > best_fit.f2:
        logger.info(
            "(%d, %d): f1=%.3f ГГц > f2=%.3f ГГц, перестановка",
            ds_lf.temp_K,
            ds_lf.field_mT,
            best_fit.f1 / GHZ,
            best_fit.f2 / GHZ,
        )
        best_fit = FittingResult(
            f1=best_fit.f2,
            f2=best_fit.f1,
            zeta1=best_fit.zeta2,
            zeta2=best_fit.zeta1,
            phi1=best_fit.phi2,
            phi2=best_fit.phi1,
            A1=best_fit.A2,
            A2=best_fit.A1,
            k_lf=best_fit.k_lf,
            k_hf=best_fit.k_hf,
            C_lf=best_fit.C_lf,
            C_hf=best_fit.C_hf,
            f1_err=best_fit.f2_err,
            f2_err=best_fit.f1_err,
            cost=best_fit.cost,
        )
    if best_fit.cost is not None and best_fit.cost > MAX_COST:
        logger.warning(
            "(%d, %d): аппроксимация отклонена f1=%.3f ГГц, f2=%.3f ГГц, cost=%.3e",
            ds_lf.temp_K,
            ds_lf.field_mT,
            best_fit.f1 / GHZ,
            best_fit.f2 / GHZ,
            best_fit.cost,
        )
        ds_lf.fit = ds_hf.fit = None
        return None
    else:
        ds_lf.fit = ds_hf.fit = best_fit
        logger.info(
            "(%d, %d): аппроксимация успешна f1=%.3f ГГц, f2=%.3f ГГц, cost=%.3e",
            ds_lf.temp_K,
            ds_lf.field_mT,
            best_fit.f1 / GHZ,
            best_fit.f2 / GHZ,
            best_fit.cost,
        )
        return best_fit


def fit_single(ds: DataSet,
               freq_bounds: tuple[tuple[float, float], tuple[float, float]] | None = None):
    t, y = ds.ts.t, ds.ts.s

    def _piecewise_time_weights(t: np.ndarray) -> np.ndarray:
        if t.size == 0:
            return np.ones_like(t)
        t_min = t.min()
        t_len = t.max() - t_min
        if t_len <= 0:
            return np.ones_like(t)
        borders = t_min + np.array([1, 2]) * t_len / 3
        w = np.ones_like(t)
        w[t >= borders[0]] = 0.8
        w[t >= borders[1]] = 0.5
        return w

    w = _piecewise_time_weights(t)

    f1_init = ds.f1_init
    f2_init = ds.f2_init
    logger.debug(
        "Начальные оценки (LF only): f1=%.3f ГГц, f2=%.3f ГГц",
        f1_init / GHZ,
        f2_init / GHZ,
    )

    _, phi1_init, A1_init, tau1_init = _single_sine_refine(t, y, f1_init)
    tau1_init, tau1_lo, tau1_hi = _resolve_tau_bounds(
        ds.zeta1, tau1_init, 5e-11, 5e-9
    )

    proto = A1_init * np.exp(-t / tau1_init) * np.cos(2 * PI * f1_init * t + phi1_init)
    _, phi2_init, A2_init, tau2_init = _single_sine_refine(t, y - proto, f2_init)
    tau2_init, tau2_lo, tau2_hi = _resolve_tau_bounds(
        ds.zeta2, tau2_init, 5e-12, 5e-10
    )

    k_init = 1.0
    C_init = np.mean(y)
    p0 = np.array([
        k_init,
        C_init,
        A1_init,
        A2_init,
        tau1_init,
        tau2_init,
        f1_init,
        f2_init,
        phi1_init,
        phi2_init,
    ])

    if freq_bounds is None:
        f1_lo, f1_hi = f1_init * 0.9, f1_init * 1.2
        f2_lo, f2_hi = f2_init * 0.9, f2_init * 1.2
    else:
        (f1_lo, f1_hi), (f2_lo, f2_hi) = freq_bounds

    lo = np.array([
        0.5,
        C_init - np.std(y),
        0.0,
        0.0,
        tau1_lo,
        tau2_lo,
        f1_lo,
        f2_lo,
        -PI,
        -PI,
    ])
    hi = np.array([
        2.0,
        C_init + np.std(y),
        A1_init * 2,
        A2_init * 2,
        tau1_hi,
        tau2_hi,
        f1_hi,
        f2_hi,
        PI,
        PI,
    ])

    def residuals(p):
        (k, C, A1, A2, tau1, tau2, f1_, f2_, phi1_, phi2_) = p
        core = _core_signal(t, A1, A2, tau1, tau2, f1_, f2_, phi1_, phi2_)
        return w * (k * core + C - y)

    sol = least_squares(
        residuals,
        p0,
        bounds=(lo, hi),
        method="trf",
        ftol=1e-15,
        xtol=1e-15,
        gtol=1e-15,
        max_nfev=100000,
        loss="soft_l1",
        f_scale=0.1,
        x_scale="jac",
        verbose=0,
    )

    p = sol.x
    cost = sol.cost
    m, n = sol.jac.shape
    try:
        cov = np.linalg.inv(sol.jac.T @ sol.jac) * 2 * cost / max(m - n, 1)
    except np.linalg.LinAlgError:
        cov = np.full((n, n), np.nan)

    idx_f1 = 6
    idx_f2 = 7
    sigma_f1 = math.sqrt(abs(cov[idx_f1, idx_f1]))
    sigma_f2 = math.sqrt(abs(cov[idx_f2, idx_f2]))

    (k_fin, C_fin, A1_fin, A2_fin, tau1_fin, tau2_fin,
     f1_fin, f2_fin, phi1_fin, phi2_fin) = p
    logger.debug(
        "Результат LF-only: f1=%.3f±%.3f ГГц, f2=%.3f±%.3f ГГц, cost=%.3e",
        f1_fin / GHZ,
        sigma_f1 / GHZ,
        f2_fin / GHZ,
        sigma_f2 / GHZ,
        cost,
    )

    return (
        FittingResult(
            f1=f1_fin,
            f2=f2_fin,
            zeta1=1 / tau1_fin,
            zeta2=1 / tau2_fin,
            phi1=phi1_fin,
            phi2=phi2_fin,
            A1=A1_fin,
            A2=A2_fin,
            k_lf=k_fin,
            k_hf=float("nan"),
            C_lf=C_fin,
            C_hf=float("nan"),
            f1_err=sigma_f1,
            f2_err=sigma_f2,
            cost=cost,
        ),
        cost,
    )


def process_lf_only(
    ds_lf: DataSet,
    *,
    use_theory_guess: bool = True,
) -> Optional[FittingResult]:
    logger.info(
        "LF-only обработка пары T=%d K, H=%d mT",
        ds_lf.temp_K,
        ds_lf.field_mT,
    )
    t, y = ds_lf.ts.t, ds_lf.ts.s

    def _search_candidates_single() -> tuple[
        list[tuple[float, Optional[float]]],
        list[tuple[float, Optional[float]]],
        tuple[tuple[float, float], tuple[float, float]] | None,
    ]:
        f1_rough, phi1, A1, tau1 = _single_sine_refine(t, y, f0=10 * GHZ)
        proto = A1 * np.exp(-t / tau1) * np.cos(2 * np.pi * f1_rough * t + phi1)
        residual = y - proto
        f2_rough, _, _, _ = _single_sine_refine(t, residual, f0=40 * GHZ)
        logger.debug(
            "Rough estimates (LF only): f1=%.3f ГГц, f2=%.3f ГГц",
            f1_rough / GHZ,
            f2_rough / GHZ,
        )
        spec_hf = residual - np.mean(residual)
        f_all_hf, zeta_all_hf = _esprit_freqs_and_decay(spec_hf, ds_lf.ts.meta.fs)
        mask_hf = (
            (zeta_all_hf > -5e8)
            & (HF_BAND[0] <= f_all_hf)
            & (f_all_hf <= HF_BAND[1])
            & (np.abs(f_all_hf - f2_rough) <= 7 * GHZ)
        )
        logger.debug(
            "ESPRIT HF filtered (LF only): %s",
            np.round(f_all_hf[mask_hf] / GHZ, 3),
        )
        hf_c: list[tuple[float, Optional[float]]] = []
        if np.any(mask_hf):
            hf_c.extend(
                [
                    (float(f), float(z) if z > 0 else None)
                    for f, z in zip(f_all_hf[mask_hf], zeta_all_hf[mask_hf])
                ]
            )
        else:
            logger.warning(
                f"({ds_lf.temp_K}, {ds_lf.field_mT}): ESPRIT не дал HF-кандидатов (LF only)"
            )

        range_hf = (
            (f2_rough - 5 * GHZ, f2_rough + 5 * GHZ)
            if f2_rough is not None
            else HF_BAND
        )
        logger.info(
            "HF fallback range: %.1f–%.1f ГГц",
            range_hf[0] / GHZ,
            range_hf[1] / GHZ,
        )
        f2_fallback = _fallback_peak(
            t,
            residual,
            ds_lf.ts.meta.fs,
            range_hf,
            f2_rough if f2_rough is not None else 0.0,
        )
        if f2_fallback is not None:
            if all(abs(f - f2_fallback) >= 20e6 for f, _ in hf_c):
                hf_c.append((float(f2_fallback), None))
                logger.info(
                    "(%d, %d): добавлен HF-кандидат %.3f ГГц методом fallback (LF only)",
                    ds_lf.temp_K,
                    ds_lf.field_mT,
                    f2_fallback / GHZ,
                )
        elif not hf_c:
            raise RuntimeError("HF-тон не найден")

        spec_lf = y - np.mean(y)
        f_all_lf, zeta_all_lf = _esprit_freqs_and_decay(spec_lf, ds_lf.ts.meta.fs)
        mask_lf = (
            (zeta_all_lf > 0)
            & (LF_BAND[0] <= f_all_lf)
            & (f_all_lf <= LF_BAND[1])
            & (np.abs(f_all_lf - f1_rough) <= 5 * GHZ)
        )
        logger.debug(
            "ESPRIT LF filtered (LF only): %s",
            np.round(f_all_lf[mask_lf] / GHZ, 3),
        )
        if np.any(mask_lf):
            lf_c = _top2_nearest(f_all_lf[mask_lf], zeta_all_lf[mask_lf], f1_rough)
        else:
            logger.warning(
                f"({ds_lf.temp_K}, {ds_lf.field_mT}): ESPRIT не дал LF-кандидатов (LF only)"
            )
            lf_c = []

        range_lf = (
            (f1_rough - 5 * GHZ, f1_rough + 5 * GHZ)
            if f1_rough is not None
            else LF_BAND
        )
        logger.info(
            "LF fallback range: %.1f–%.1f ГГц",
            range_lf[0] / GHZ,
            range_lf[1] / GHZ,
        )
        f1_fallback = _fallback_peak(
            t,
            y,
            ds_lf.ts.meta.fs,
            range_lf,
            f1_rough if f1_rough is not None else 0.0,
            avoid=hf_c[0][0] if hf_c else None,
        )
        if f1_fallback is not None:
            if all(abs(f - f1_fallback) >= 20e6 for f, _ in lf_c):
                lf_c.append((float(f1_fallback), None))
                logger.info(
                    "(%d, %d): добавлен LF-кандидат %.3f ГГц методом fallback (LF only)",
                    ds_lf.temp_K,
                    ds_lf.field_mT,
                    f1_fallback / GHZ,
                )
        elif not lf_c:
            raise RuntimeError("LF-тон не найден")
        return lf_c, hf_c, None

    guess = None
    if use_theory_guess and ds_lf.root:
        guess = _load_guess(ds_lf.root, ds_lf.field_mT, ds_lf.temp_K)

    if guess is not None:
        f1_guess, f2_guess = guess
        logger.info(
            "(%d, %d): использованы предварительные оценки f1=%.3f ГГц, f2=%.3f ГГц",
            ds_lf.temp_K,
            ds_lf.field_mT,
            f1_guess / GHZ,
            f2_guess / GHZ,
        )
        f1_rough, phi1, A1, tau1 = _single_sine_refine(t, y, f1_guess)
        proto = A1 * np.exp(-t / tau1) * np.cos(2 * PI * f1_rough * t + phi1)
        residual = y - proto
        spec_hf = residual - np.mean(residual)
        f_all_hf, zeta_all_hf = _esprit_freqs_and_decay(spec_hf, ds_lf.ts.meta.fs)
        mask_hf = (
            (zeta_all_hf > -5e8)
            & (HF_BAND[0] <= f_all_hf)
            & (f_all_hf <= HF_BAND[1])
            & (np.abs(f_all_hf - f2_guess) <= 7 * GHZ)
        )
        hf_cand = _top2_nearest(
            f_all_hf[mask_hf], np.maximum(zeta_all_hf[mask_hf], 0.0), f2_guess
        ) if np.any(mask_hf) else []
        if not hf_cand:
            logger.warning(
                f"({ds_lf.temp_K}, {ds_lf.field_mT}): ESPRIT не дал HF-кандидатов (LF only)"
            )

        range_hf = (f2_guess - 5 * GHZ, f2_guess + 5 * GHZ)
        logger.info(
            "HF fallback range: %.1f–%.1f ГГц",
            range_hf[0] / GHZ,
            range_hf[1] / GHZ,
        )
        f2_fallback = _fallback_peak(
            t,
            residual,
            ds_lf.ts.meta.fs,
            range_hf,
            f2_guess,
        )
        if f2_fallback is not None and all(
            abs(f - f2_fallback) >= 20e6 for f, _ in hf_cand
        ):
            hf_cand.append((float(f2_fallback), None))
            logger.info(
                "(%d, %d): добавлен HF-кандидат %.3f ГГц методом fallback (LF only)",
                ds_lf.temp_K,
                ds_lf.field_mT,
                f2_fallback / GHZ,
            )
        elif not hf_cand:
            raise RuntimeError("HF-тон не найден")
        spec_lf = y - np.mean(y)
        f_all_lf, zeta_all_lf = _esprit_freqs_and_decay(spec_lf, ds_lf.ts.meta.fs)
        mask_lf = (
            (zeta_all_lf > 0)
            & (LF_BAND[0] <= f_all_lf)
            & (f_all_lf <= LF_BAND[1])
            & (np.abs(f_all_lf - f1_guess) <= 5 * GHZ)
        )
        lf_cand = _top2_nearest(f_all_lf[mask_lf], zeta_all_lf[mask_lf], f1_guess) if np.any(mask_lf) else []
        if not lf_cand:
            logger.warning(
                f"({ds_lf.temp_K}, {ds_lf.field_mT}): ESPRIT не дал LF-кандидатов (LF only)"
            )

        range_lf = (f1_guess - 5 * GHZ, f1_guess + 5 * GHZ)
        logger.info(
            "LF fallback range: %.1f–%.1f ГГц",
            range_lf[0] / GHZ,
            range_lf[1] / GHZ,
        )
        f1_fallback = _fallback_peak(
            t,
            y,
            ds_lf.ts.meta.fs,
            range_lf,
            f1_guess,
            avoid=hf_cand[0][0] if hf_cand else None,
        )
        if f1_fallback is not None and all(
            abs(f - f1_fallback) >= 20e6 for f, _ in lf_cand
        ):
            lf_cand.append((float(f1_fallback), None))
            logger.info(
                "(%d, %d): добавлен LF-кандидат %.3f ГГц методом fallback (LF only)",
                ds_lf.temp_K,
                ds_lf.field_mT,
                f1_fallback / GHZ,
            )
        elif not lf_cand:
            raise RuntimeError("LF-тон не найден")

        def _ensure_guess(cands, freq):
            for f, _ in cands:
                if abs(f - freq) < 20e6:
                    return
            cands.append((freq, None))

        _ensure_guess(lf_cand, f1_guess)
        _ensure_guess(hf_cand, f2_guess)
        freq_bounds = (
            (f1_guess - 5 * GHZ, f1_guess + 5 * GHZ),
            (f2_guess - 5 * GHZ, f2_guess + 5 * GHZ),
        )
    else:
        logger.info(
            "(%d, %d): поиск предварительных оценок (LF only)",
            ds_lf.temp_K,
            ds_lf.field_mT,
        )
        lf_cand, hf_cand, freq_bounds = _search_candidates_single()

    seen: set[tuple[float, float]] = set()
    best_cost = np.inf
    best_fit = None
    for f1, z1 in lf_cand:
        for f2, z2 in hf_cand:
            if (f1, f2) in seen:
                continue
            if freq_bounds is not None:
                (f1_lo, f1_hi), (f2_lo, f2_hi) = freq_bounds
                if not (f1_lo <= f1 <= f1_hi and f2_lo <= f2 <= f2_hi):
                    seen.add((f1, f2))
                    continue
            ds_lf.f1_init, ds_lf.zeta1 = f1, z1
            ds_lf.f2_init, ds_lf.zeta2 = f2, z2
            try:
                fit, cost = fit_single(ds_lf, freq_bounds=freq_bounds)
            except Exception as exc:
                logger.debug(
                    "Неудачная попытка f1=%.3f ГГц, f2=%.3f ГГц: %s",
                    f1 / GHZ,
                    f2 / GHZ,
                    exc,
                )
            else:
                if cost < best_cost:
                    best_cost = cost
                    best_fit = fit
            finally:
                seen.add((f1, f2))

    if best_fit is None:
        logger.error(
            "(%d, %d): ни одна комбинация не аппроксимировалась (LF only)",
            ds_lf.temp_K,
            ds_lf.field_mT,
        )
        raise RuntimeError("Ни одна комбинация не аппроксимировалась")

    if best_fit.f1 > best_fit.f2:
        best_fit = FittingResult(
            f1=best_fit.f2,
            f2=best_fit.f1,
            zeta1=best_fit.zeta2,
            zeta2=best_fit.zeta1,
            phi1=best_fit.phi2,
            phi2=best_fit.phi1,
            A1=best_fit.A2,
            A2=best_fit.A1,
            k_lf=best_fit.k_lf,
            k_hf=best_fit.k_hf,
            C_lf=best_fit.C_lf,
            C_hf=best_fit.C_hf,
            f1_err=best_fit.f2_err,
            f2_err=best_fit.f1_err,
            cost=best_fit.cost,
        )

    if best_fit.cost is not None and best_fit.cost > MAX_COST:
        logger.warning(
            "(%d, %d): аппроксимация отклонена f1=%.3f ГГц, f2=%.3f ГГц, cost=%.3e",
            ds_lf.temp_K,
            ds_lf.field_mT,
            best_fit.f1 / GHZ,
            best_fit.f2 / GHZ,
            best_fit.cost,
        )
        ds_lf.fit = None
        return None

    ds_lf.fit = best_fit
    logger.info(
        "(%d, %d): аппроксимация успешна f1=%.3f ГГц, f2=%.3f ГГц, cost=%.3e",
        ds_lf.temp_K,
        ds_lf.field_mT,
        best_fit.f1 / GHZ,
        best_fit.f2 / GHZ,
        best_fit.cost,
    )
    return best_fit
