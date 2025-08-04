from __future__ import annotations

from typing import Tuple, List, Optional
from pathlib import Path
import math
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares
from scipy.signal import welch, find_peaks, get_window, czt

from . import DataSet, FittingResult, GHZ, PI, logger, LF_BAND, HF_BAND

# Maximum acceptable fitting cost. Pairs with higher cost are rejected
# and treated as unsuccessful.
MAX_COST = 120

# Weight applied to the relative deviation between the HF/LF ratio obtained
# from a candidate fit and the expected ratio. Larger values make the
# selection prefer frequency pairs that preserve the anticipated harmonic
# relation even if the raw least‑squares cost is slightly worse.
# The penalty is normalised by ``RATIO_DEV_TOL`` so that a deviation of
# that magnitude increases the score by roughly ``1 + RATIO_PENALTY``.
# Keeping ``RATIO_PENALTY`` near 1 balances theory with fit quality.
# Increasing it strengthens the preference for ratios close to the
# theoretical target.  A smaller value relaxes this bias so that a
# noticeably better fit can override disagreements with the expected
# harmonic relation.
RATIO_PENALTY = 0.5

# Acceptable bounds for the HF/LF frequency ratio.  Candidate fits outside
# this window are strongly down‑weighted to avoid selecting harmonics that
# are obviously unphysical (e.g. HF ≈ LF).
RATIO_MIN = 1.5
RATIO_MAX = 4.0

# Weight applied to the relative deviation from the theoretical frequency
# guesses.  A value of zero disables the penalty while larger values keep the
# optimisation closer to the expected frequencies derived from theory.
# The same reasoning as above applies – a value close to one gently nudges the
# search towards the supplied guesses while still allowing the optimiser to
# favour markedly better fits when the data disagrees with theory.  A
# lighter penalty grants the optimiser more freedom to deviate from the
# initial guess when doing so reduces the residual error.
GUESS_PENALTY = 0.25

# Hard limit for how far the final fit is allowed to drift from the
# theoretical prediction before additional penalty is applied.  Deviations
# within this ±5 % band are deemed "not critical" and do not increase the
# score.  Beyond the band the penalty grows quadratically.
GUESS_DEV_TOL = 0.05

# Broader tolerance used only during candidate search and window setup.  A
# wider ±15 % corridor keeps alternatives on the table even when the true
# signal is somewhat offset from the theoretical guess.  This value is
# intentionally larger than ``GUESS_DEV_TOL`` to avoid prematurely rejecting
# good fits that slightly violate theory.
GUESS_SEARCH_TOL = 0.15

# Relative tolerance for deviations from the expected HF/LF ratio when a
# theoretical ratio is available. The ratio penalty uses this value for
# normalisation in the same way as the frequency guess penalty above.
# Using the same 15 % window keeps harmonic relationships in check while
# still allowing significant departures when they improve the fit.
RATIO_DEV_TOL = 0.15


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


def _parabolic_peak(freqs: np.ndarray, amps: np.ndarray, idx: int) -> float:
    """Refine peak location using parabolic interpolation.

    ``freqs`` must be uniformly spaced, as is the case for FFT bins.  The
    function returns the interpolated frequency in Hz.  If the index is at a
    boundary or the curvature is non-positive the original bin frequency is
    returned.
    """

    if idx <= 0 or idx >= len(amps) - 1:
        return float(freqs[idx])

    y0, y1, y2 = amps[idx - 1], amps[idx], amps[idx + 1]
    denom = y0 - 2 * y1 + y2
    if denom <= 0:
        return float(freqs[idx])

    delta = 0.5 * (y0 - y2) / denom
    df = freqs[1] - freqs[0]
    return float(freqs[idx] + delta * df)


def _peak_in_band(
    freqs: np.ndarray,
    amps: np.ndarray,
    fmin_GHz: float,
    fmax_GHz: float,
    *,
    max_expansions: int = 3,
    expansion_step_GHz: float = 2.0,
    theory_GHz: float | None = None,
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
                f_best = _parabolic_peak(f_band, a_band, best_idx)
                found_peak = True
                break

        if not found_peak:
            if theory_GHz is not None and fmin_GHz <= theory_GHz <= fmax_GHz:
                idx = int(np.argmin(np.abs(f_band - theory_GHz * GHZ)))
                f_best = _parabolic_peak(f_band, a_band, idx)
                logger.debug(
                    "no peaks found, using theory: f=%.3f ГГц, amp=%.3g",
                    f_best / GHZ,
                    a_band[idx],
                )
            else:
                max_idx = int(np.argmax(a_band))
                f_best = _parabolic_peak(f_band, a_band, max_idx)
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
        if f_range[0] <= f_burg <= f_range[1] and \
           (avoid is None or abs(f_burg - avoid) >= df_min):
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

    def _czt_peak() -> tuple[float | None, float]:
        """High-resolution spectral peak using the chirp z-transform."""
        if len(y) < 512:
            return None, 0.0
        n = max(256, int(4 * (f_range[1] - f_range[0]) / df_min))
        f1, f2 = f_range
        w = np.exp(-1j * 2 * np.pi * (f2 - f1) / (n * fs))
        a = np.exp(1j * 2 * np.pi * f1 / fs)
        spec = np.abs(czt(y, n, w, a))
        freqs = f1 + np.arange(n) * (f2 - f1) / n
        if avoid is not None:
            mask = np.abs(freqs - avoid) >= df_min
        else:
            mask = np.ones_like(freqs, dtype=bool)
        if not np.any(mask):
            return None, 0.0
        idx = np.argmax(spec[mask])
        return float(freqs[mask][idx]), float(spec[mask][idx])

    def _music_peak() -> tuple[float | None, float]:
        """Frequency estimate using a simple MUSIC spectral estimator."""
        p = min(8, len(y) // 3)
        if p < 2:
            return None, 0.0
        r = np.array([np.dot(y[: len(y) - k], y[k:]) for k in range(p)])
        from scipy.linalg import toeplitz
        R = toeplitz(r)
        try:
            w, v = np.linalg.eigh(R)
        except np.linalg.LinAlgError:
            return None, 0.0
        En = v[:, :-1]
        freqs = np.linspace(f_range[0], f_range[1], 512)
        a = np.exp(-1j * 2 * np.pi * np.arange(p)[:, None] * freqs / fs)
        ps = 1.0 / (np.sum(np.abs(En.conj().T @ a) ** 2, axis=0) + 1e-16)
        if avoid is not None:
            mask = np.abs(freqs - avoid) >= df_min
        else:
            mask = np.ones_like(freqs, dtype=bool)
        if not np.any(mask):
            return None, 0.0
        idx = int(np.argmax(ps[mask]))
        return float(freqs[mask][idx]), float(ps[mask][idx])

    def _acf_peak() -> tuple[float | None, float]:
        y0 = y - y.mean()
        corr = np.correlate(y0, y0, mode="full")[len(y0)-1:]
        lags = np.arange(len(corr)) / fs
        fmax, fmin = f_range[1], f_range[0]
        # convert frequency bounds to lag bounds
        lag_min = 1.0 / fmax if fmax > 0 else lags[-1]
        lag_max = 1.0 / fmin if fmin > 0 else lags[-1]
        mask = (lags >= lag_min) & (lags <= lag_max)
        if avoid is not None:
            lag_avoid = 1.0 / avoid if avoid != 0 else 0.0
            mask &= np.abs(lags - lag_avoid) >= df_min / (avoid**2 + 1e-16)
        if not np.any(mask):
            return None, 0.0
        idx = np.argmax(corr[mask])
        lag = lags[mask][idx]
        if lag <= 0:
            return None, 0.0
        return float(1.0 / lag), float(corr[mask][idx] / (corr[0] + 1e-16))

    def _zc_peak() -> tuple[float | None, float]:
        """Estimate frequency from zero‑crossing intervals.

        The signal is linearly interpolated around each sign change to obtain
        sub‑sample crossing times.  The mean distance between every second
        crossing corresponds to one full period.  This very time‑domain view is
        largely insensitive to spectral leakage and provides a lightweight
        fallback when frequency content is broad or heavily damped.
        """

        y0 = y - y.mean()
        s = np.sign(y0)
        crossings = np.where(np.diff(s) != 0)[0]
        if crossings.size < 3:
            return None, 0.0
        t0 = t[crossings]
        t1 = t[crossings + 1]
        y0c = y0[crossings]
        y1c = y0[crossings + 1]
        zt = t0 - y0c * (t1 - t0) / (y1c - y0c)
        if zt.size < 3:
            return None, 0.0
        periods = zt[2:] - zt[:-2]
        if np.any(periods <= 0):
            return None, 0.0
        f_est = 1.0 / np.mean(periods)
        if not (f_range[0] <= f_est <= f_range[1]):
            return None, 0.0
        if avoid is not None and abs(f_est - avoid) < df_min:
            return None, 0.0
        amp = float(np.max(np.abs(y0)))
        return float(f_est), amp

    fw, pw = _welch_peak()
    if fw is not None:
        logger.debug("Welch estimate: %.3f ГГц", fw / GHZ)
    fa, pa = _avg_fft_peak()
    if fa is not None:
        logger.debug("AvgFFT estimate: %.3f ГГц", fa / GHZ)
    fczt, pczt = _czt_peak()
    if fczt is not None:
        logger.debug("CZT estimate: %.3f ГГц", fczt / GHZ)
    fm, pm = _music_peak()
    if fm is not None:
        logger.debug("MUSIC estimate: %.3f ГГц", fm / GHZ)
    fc, pc = _acf_peak()
    if fc is not None:
        logger.debug("ACF estimate: %.3f ГГц", fc / GHZ)
    fz, pz = _zc_peak()
    if fz is not None:
        logger.debug("ZC estimate: %.3f ГГц", fz / GHZ)

    candidates: list[tuple[float, float, float]] = []
    if fw is not None:
        candidates.append((fw, pw, abs(fw - f_rough)))
    if fa is not None:
        candidates.append((fa, pa, abs(fa - f_rough)))
    if fczt is not None:
        candidates.append((fczt, pczt, abs(fczt - f_rough)))
    if fm is not None:
        candidates.append((fm, pm, abs(fm - f_rough)))
    if fc is not None:
        candidates.append((fc, pc, abs(fc - f_rough)))
    if fz is not None:
        candidates.append((fz, pz, abs(fz - f_rough)))
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
    if ds_lf.zeta1 is None:
        tau1_lo, tau1_hi = 5e-11, 5e-9
    else:
        tau1_init = 1.0 / ds_lf.zeta1
        tau1_lo, tau1_hi = tau1_init * 0.8, tau1_init * 1.2

    proto_lf_hf = A1_init * np.exp(-t_hf / tau1_init) * np.cos(2 * PI * f1_init * t_hf + phi1_init)
    _, phi2_init, A2_init, tau2_init = _single_sine_refine(t_hf, y_hf - proto_lf_hf, f2_init)
    if ds_hf.zeta2 is None:
        tau2_lo, tau2_hi = 5e-12, 5e-10
    else:
        tau2_init = 1.0 / ds_hf.zeta2
        tau2_lo, tau2_hi = tau2_init * 0.8, tau2_init * 1.2

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
        return np.concatenate([res_lf, res_hf])

    sol = least_squares(
        residuals,
        p0,
        bounds=(lo, hi),
        method='trf',
        ftol=1e-15, xtol=1e-15, gtol=1e-15,
        max_nfev=100000,
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


def process_pair(ds_lf: DataSet, ds_hf: DataSet) -> Optional[FittingResult]:
    logger.info("Обработка пары T=%d K, H=%d mT", ds_lf.temp_K, ds_lf.field_mT)
    tau_guess_lf, tau_guess_hf = 3e-10, 3e-11
    t_lf, y_lf = ds_lf.ts.t, ds_lf.ts.s
    t_hf, y_hf = ds_hf.ts.t, ds_hf.ts.s

    def _search_candidates() -> tuple[list[tuple[float, Optional[float]]],
                                      list[tuple[float, Optional[float]]],
                                      tuple[tuple[float, float], tuple[float, float]] | None,
                                      float | None]:
        f1_rough, phi1, A1, tau1 = _single_sine_refine(t_lf, y_lf, f0=10 * GHZ)
        proto_lf = A1 * np.exp(-t_hf / tau1) * np.cos(2 * np.pi * f1_rough * t_hf + phi1)
        residual = y_hf - proto_lf
        f2_rough, _, _, _ = _single_sine_refine(t_hf, residual, f0=40 * GHZ)
        logger.debug("Rough estimates: f1=%.3f ГГц, f2=%.3f ГГц",
                     f1_rough/GHZ, f2_rough/GHZ)
        spec_hf = y_hf - np.mean(y_hf)
        f_all_hf, zeta_all_hf = _esprit_freqs_and_decay(spec_hf, ds_hf.ts.meta.fs)
        mask_hf = (
            (zeta_all_hf > -5e8)
            & (HF_BAND[0] <= f_all_hf)
            & (f_all_hf <= HF_BAND[1])
            & (np.abs(f_all_hf - f2_rough) <= GUESS_SEARCH_TOL * f2_rough)
        )
        logger.debug("ESPRIT HF filtered: %s", np.round(f_all_hf[mask_hf] / GHZ, 3))
        if not np.any(mask_hf):
            mask_hf = (HF_BAND[0] <= f_all_hf) & (f_all_hf <= HF_BAND[1])
            logger.debug("ESPRIT HF relaxed: %s", np.round(f_all_hf[mask_hf] / GHZ, 3))
        if np.any(mask_hf):
            hf_c = [(f, z if z > 0 else None)
                    for f, z in zip(f_all_hf[mask_hf], zeta_all_hf[mask_hf])]
        else:
            logger.warning(f"({ds_hf.temp_K}, {ds_hf.field_mT}): вызван fallback для HF")
            span = 10 * GHZ
            if f2_rough is None or f2_rough < HF_BAND[0] or f2_rough > HF_BAND[1]:
                range_hf = HF_BAND
            else:
                range_hf = (
                    max(HF_BAND[0], f2_rough - span),
                    min(HF_BAND[1], f2_rough + span),
                )
            logger.info("HF fallback range: %.1f–%.1f ГГц", range_hf[0]/GHZ, range_hf[1]/GHZ)
            f2_fallback = _fallback_peak(
                t_hf,
                y_hf,
                ds_hf.ts.meta.fs,
                range_hf,
                f2_rough if f2_rough is not None else 0.0,
            )
            if f2_fallback is None:
                raise RuntimeError("HF-тон не найден")
            hf_c = [(f2_fallback, None)]
        spec_lf = y_lf - np.mean(y_lf)
        f_all_lf, zeta_all_lf = _esprit_freqs_and_decay(spec_lf, ds_lf.ts.meta.fs)
        mask_lf = (
            (zeta_all_lf > 0)
            & (LF_BAND[0] <= f_all_lf)
            & (f_all_lf <= LF_BAND[1])
            & (np.abs(f_all_lf - f1_rough) <= GUESS_SEARCH_TOL * f1_rough)
        )
        logger.debug("ESPRIT LF filtered: %s", np.round(f_all_lf[mask_lf] / GHZ, 3))
        if np.any(mask_lf):
            lf_c = _top2_nearest(f_all_lf[mask_lf], zeta_all_lf[mask_lf], f1_rough)
        else:
            logger.warning(f"({ds_lf.temp_K}, {ds_lf.field_mT}): вызван fallback для LF")
            span = 10 * GHZ
            if f1_rough is None or f1_rough < LF_BAND[0] or f1_rough > LF_BAND[1]:
                range_lf = LF_BAND
            else:
                range_lf = (
                    max(LF_BAND[0], f1_rough - span),
                    min(LF_BAND[1], f1_rough + span),
                )
            logger.info("LF fallback range: %.1f–%.1f ГГц", range_lf[0]/GHZ, range_lf[1]/GHZ)
            f1_fallback = _fallback_peak(
                t_lf,
                y_lf,
                ds_lf.ts.meta.fs,
                range_lf,
                f1_rough if f1_rough is not None else 0.0,
                avoid=hf_c[0][0] if hf_c else None,
            )
            if f1_fallback is None:
                raise RuntimeError("LF-тон не найден")
            lf_c = [(f1_fallback, None)]
        ratio_rough = f2_rough / f1_rough if f1_rough else None
        return lf_c, hf_c, None, ratio_rough

    guess = None
    target_ratio: float | None = None
    if ds_lf.root:
        guess = _load_guess(ds_lf.root, ds_lf.field_mT, ds_lf.temp_K)

    if guess is not None:
        f1_guess, f2_guess = guess
        f1_guess = float(np.clip(f1_guess, LF_BAND[0], LF_BAND[1]))
        f2_guess = float(np.clip(f2_guess, HF_BAND[0], HF_BAND[1]))
        ratio_guess = f2_guess / f1_guess if f1_guess > 0 else np.inf
        if ratio_guess < RATIO_MIN:
            f2_guess = f1_guess * RATIO_MIN
            ratio_guess = RATIO_MIN
        elif ratio_guess > RATIO_MAX:
            f2_guess = f1_guess * RATIO_MAX
            ratio_guess = RATIO_MAX
        target_ratio = ratio_guess
        guess = (f1_guess, f2_guess)
        logger.info(
            "(%d, %d): использованы предварительные оценки f1=%.3f ГГц, f2=%.3f ГГц",
            ds_lf.temp_K, ds_lf.field_mT, f1_guess/GHZ, f2_guess/GHZ)
        spec_hf = y_hf - np.mean(y_hf)
        f_all_hf, zeta_all_hf = _esprit_freqs_and_decay(spec_hf, ds_hf.ts.meta.fs)
        mask_hf = (
            (zeta_all_hf > -5e8)
            & (HF_BAND[0] <= f_all_hf)
            & (f_all_hf <= HF_BAND[1])
            & (np.abs(f_all_hf - f2_guess) <= GUESS_SEARCH_TOL * f2_guess)
        )
        logger.debug("ESPRIT HF filtered: %s", np.round(f_all_hf[mask_hf] / GHZ, 3))
        if not np.any(mask_hf):
            mask_hf = (HF_BAND[0] <= f_all_hf) & (f_all_hf <= HF_BAND[1])
            logger.debug("ESPRIT HF relaxed: %s", np.round(f_all_hf[mask_hf] / GHZ, 3))
        hf_cand = _top2_nearest(f_all_hf[mask_hf], np.maximum(zeta_all_hf[mask_hf], 0.0), f2_guess) if np.any(mask_hf) else []
        if not hf_cand:
            logger.warning(f"({ds_hf.temp_K}, {ds_hf.field_mT}): вызван fallback для HF")
            span = 10 * GHZ
            range_hf = (
                max(HF_BAND[0], f2_guess - span),
                min(HF_BAND[1], f2_guess + span),
            )
            logger.info("HF fallback range: %.1f–%.1f ГГц", range_hf[0]/GHZ, range_hf[1]/GHZ)
            f2_fallback = _fallback_peak(t_hf, y_hf, ds_hf.ts.meta.fs, range_hf, f2_guess)
            if f2_fallback is None:
                raise RuntimeError("HF-тон не найден")
            hf_cand = [(f2_fallback, None)]
        spec_lf = y_lf - np.mean(y_lf)
        f_all_lf, zeta_all_lf = _esprit_freqs_and_decay(spec_lf, ds_lf.ts.meta.fs)
        mask_lf = (
            (zeta_all_lf > 0)
            & (LF_BAND[0] <= f_all_lf)
            & (f_all_lf <= LF_BAND[1])
            & (np.abs(f_all_lf - f1_guess) <= GUESS_SEARCH_TOL * f1_guess)
        )
        logger.debug("ESPRIT LF filtered: %s", np.round(f_all_lf[mask_lf] / GHZ, 3))
        lf_cand = _top2_nearest(f_all_lf[mask_lf], zeta_all_lf[mask_lf], f1_guess) if np.any(mask_lf) else []
        if not lf_cand:
            logger.warning(f"({ds_lf.temp_K}, {ds_lf.field_mT}): вызван fallback для LF")
            span = 10 * GHZ
            range_lf = (
                max(LF_BAND[0], f1_guess - span),
                min(LF_BAND[1], f1_guess + span),
            )
            logger.info("LF fallback range: %.1f–%.1f ГГц", range_lf[0]/GHZ, range_lf[1]/GHZ)
            f1_fallback = _fallback_peak(t_lf, y_lf, ds_lf.ts.meta.fs,
                                        range_lf, f1_guess,
                                        avoid=hf_cand[0][0] if hf_cand else None)
            if f1_fallback is None:
                raise RuntimeError("LF-тон не найден")
            lf_cand = [(f1_fallback, None)]
        def _ensure_guess(cands, freq):
            for f, _ in cands:
                if abs(f - freq) < 20e6:
                    return
            cands.append((freq, None))
        _ensure_guess(lf_cand, f1_guess)
        _ensure_guess(hf_cand, f2_guess)
        f1_lo = max(LF_BAND[0], f1_guess * (1 - GUESS_SEARCH_TOL))
        f1_hi = min(LF_BAND[1], f1_guess * (1 + GUESS_SEARCH_TOL))
        f2_lo = max(HF_BAND[0], f2_guess * (1 - GUESS_SEARCH_TOL))
        f2_hi = min(HF_BAND[1], f2_guess * (1 + GUESS_SEARCH_TOL))
        freq_bounds = ((f1_lo, f1_hi), (f2_lo, f2_hi))
    else:
        logger.info("(%d, %d): поиск предварительных оценок", ds_lf.temp_K, ds_lf.field_mT)
        lf_cand, hf_cand, freq_bounds, rough_ratio = _search_candidates()
        if rough_ratio is not None and RATIO_MIN <= rough_ratio <= RATIO_MAX:
            target_ratio = rough_ratio
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
        lf_lo = f1_guess * (1 - GUESS_SEARCH_TOL) / GHZ
        lf_hi = f1_guess * (1 + GUESS_SEARCH_TOL) / GHZ
        hf_lo = f2_guess * (1 - GUESS_SEARCH_TOL) / GHZ
        hf_hi = f2_guess * (1 + GUESS_SEARCH_TOL) / GHZ
        f1_hz = _peak_in_band(freqs_fft, amps_fft, lf_lo, lf_hi, theory_GHz=f1_guess/GHZ)
        f2_hz = _peak_in_band(freqs_fft, amps_fft, hf_lo, hf_hi, theory_GHz=f2_guess/GHZ)
        range_lf = f"{lf_lo:.0f}–{lf_hi:.0f}"
        range_hf = f"{hf_lo:.0f}–{hf_hi:.0f}"
        if f1_hz is None:
            logger.warning(
                "(%d, %d): в узком диапазоне %s ГГц LF пик не найден, расширяем поиск", ds_lf.temp_K, ds_lf.field_mT, range_lf
            )
            f1_hz = _peak_in_band(freqs_fft, amps_fft, LF_BAND[0]/GHZ, LF_BAND[1]/GHZ)
            range_lf = f"{LF_BAND[0]/GHZ:.0f}–{LF_BAND[1]/GHZ:.0f}"
        if f2_hz is None:
            logger.warning(
                "(%d, %d): в узком диапазоне %s ГГц HF пик не найден, расширяем поиск", ds_lf.temp_K, ds_lf.field_mT, range_hf
            )
            f2_hz = _peak_in_band(freqs_fft, amps_fft, start_band_HF, HF_BAND[1]/GHZ)
            range_hf = f"{start_band_HF:.0f}–{HF_BAND[1]/GHZ:.0f}"
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

    # Occasionally the automatic FFT search returns a spurious HF peak
    # far away from the LF component, especially when the true HF tone
    # is weak.  If the ratio between the detected HF and LF peaks lies
    # outside the physically plausible range we perform a secondary
    # search constrained by this ratio.  This prevents the algorithm
    # from jumping to very low or very high frequencies when the spectrum
    # is dominated by noise.
    if f1_hz is not None and f2_hz is not None:
        ratio = f2_hz / f1_hz
        if ratio < RATIO_MIN or ratio > RATIO_MAX:
            fmin = max(HF_BAND[0] / GHZ, (f1_hz / GHZ) * RATIO_MIN)
            fmax = min(HF_BAND[1] / GHZ, (f1_hz / GHZ) * RATIO_MAX)
            logger.info(
                "HF peak %.1f ГГц (ratio %.2f) outside expected bounds;"
                " refining search to %.1f–%.1f ГГц",
                f2_hz / GHZ,
                ratio,
                fmin,
                fmax,
            )
            f2_alt = _peak_in_band(
                freqs_fft,
                amps_fft,
                fmin,
                fmax,
                theory_GHz=(f2_guess / GHZ if guess is not None else None),
            )
            if f2_alt is not None:
                f2_hz = f2_alt
                logger.info("Refined HF peak at %.1f ГГц", f2_hz / GHZ)
    if target_ratio is None and f1_hz is not None and f2_hz is not None and f1_hz > 0:
        ratio_fft = f2_hz / f1_hz
        if RATIO_MIN <= ratio_fft <= RATIO_MAX:
            target_ratio = ratio_fft
    def _append_unique(target_list, new_freq_hz):
        if new_freq_hz is None:
            return
        for old_f, _ in target_list:
            if abs(old_f - new_freq_hz) < 20e6:
                return
        target_list.append((new_freq_hz, None))

    _append_unique(lf_cand, f1_hz)
    _append_unique(hf_cand, f2_hz)

    logger.info("LF candidates: %s", [(round(f/GHZ,3), z) for f, z in lf_cand])
    logger.info("HF candidates: %s", [(round(f/GHZ,3), z) for f, z in hf_cand])

    seen: set[tuple[float, float]] = set()
    best_score = np.inf
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
                ratio = fit.f2 / fit.f1 if fit.f1 != 0 else np.inf
                if ratio < RATIO_MIN or ratio > RATIO_MAX:
                    score = cost + 1e6  # effectively discard implausible ratios
                else:
                    score = cost
                    if target_ratio is not None and RATIO_DEV_TOL > 0:
                        devr = abs(ratio - target_ratio) / target_ratio
                        # Quadratic penalty sharply increases the score for
                        # larger deviations while remaining gentle close to
                        # the target ratio.
                        score *= 1 + RATIO_PENALTY * (devr / RATIO_DEV_TOL) ** 2
                    if guess is not None:
                        f1g, f2g = guess
                        if f1g > 0 and f2g > 0:
                            dev1 = abs(fit.f1 - f1g) / f1g
                            dev2 = abs(fit.f2 - f2g) / f2g
                            avg_dev = (dev1 + dev2) / 2
                            # Apply a quadratic penalty only when the
                            # deviation exceeds the ``GUESS_DEV_TOL`` band.
                            # Differences within ±5 % are considered
                            # negligible.  Beyond that, the penalty grows
                            # quadratically with the excess deviation so that
                            # large departures from theory are discouraged but
                            # still allowed when they yield a markedly better
                            # residual.
                            if avg_dev > GUESS_DEV_TOL:
                                excess = (avg_dev - GUESS_DEV_TOL) / GUESS_DEV_TOL
                                score *= 1 + GUESS_PENALTY * excess ** 2
                if score < best_score:
                    best_score = score
                    best_fit = fit
            finally:
                seen.add((f1, f2))
    if best_fit is None and guess is not None:
        logger.warning(
            "(%d, %d): не удалось аппроксимировать с первым приближением, поиск альтернативы",
            ds_lf.temp_K,
            ds_lf.field_mT,
        )
        # Re-run the candidate search to broaden the pool but keep the
        # original target_ratio derived from FFT peaks or the initial
        # guess.  Earlier versions overwrote target_ratio here using the
        # first candidate pair, which could bias the selection toward
        # spurious low-frequency ratios when ESPRIT misidentified peaks.
        lf_cand, hf_cand, freq_bounds, _ = _search_candidates()
        best_score = np.inf
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
                    ratio = fit.f2 / fit.f1 if fit.f1 != 0 else np.inf
                    if ratio < RATIO_MIN or ratio > RATIO_MAX:
                        score = cost + 1e6
                    else:
                        score = cost
                        if target_ratio is not None and RATIO_DEV_TOL > 0:
                            devr = abs(ratio - target_ratio) / target_ratio
                            # Same quadratic penalty for ratio deviation as
                            # above to discourage harmonics far from the
                            # theoretical expectation.
                            score *= 1 + RATIO_PENALTY * (devr / RATIO_DEV_TOL) ** 2
                        if guess is not None:
                            f1g, f2g = guess
                            if f1g > 0 and f2g > 0:
                                dev1 = abs(fit.f1 - f1g) / f1g
                                dev2 = abs(fit.f2 - f2g) / f2g
                                avg_dev = (dev1 + dev2) / 2
                                if avg_dev > GUESS_DEV_TOL:
                                    excess = (avg_dev - GUESS_DEV_TOL) / GUESS_DEV_TOL
                                    score *= 1 + GUESS_PENALTY * excess ** 2
                    if score < best_score:
                        best_score = score
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
