from __future__ import annotations

from typing import Tuple, List, Optional
import math
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares
from scipy.signal import welch, find_peaks, get_window

from . import DataSet, FittingResult, GHZ, PI, logger, LF_BAND, HF_BAND

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


def _peak_in_band(freqs: np.ndarray, amps: np.ndarray, fmin_GHz: float,
                  fmax_GHz: float) -> float | None:
    mask = (freqs >= fmin_GHz * GHZ) & (freqs <= fmax_GHz * GHZ)
    if not mask.any():
        return None
    f_band = freqs[mask]
    a_band = amps[mask]
    thr = np.median(a_band) + 0.3 * np.std(a_band)
    df = f_band[1] - f_band[0] if len(f_band) > 1 else 1 * GHZ
    dist = int(0.3 * GHZ / df)
    pk, _ = find_peaks(a_band, height=thr, distance=max(1, dist))
    if pk.size == 0:
        return None
    best_idx = pk[np.argmax(a_band[pk])]
    return float(f_band[best_idx])


def _crop_signal(t: NDArray, s: NDArray, tag: str):
    pk = int(np.argmax(s))
    minima = np.where(
        (np.diff(np.signbit(np.diff(s))) > 0) & (np.arange(len(s))[1:-1] > pk)
    )[0]
    st = minima[0] + 1 if minima.size else pk + 1
    cutoff = 0.9e-9 if tag == "LF" else 0.12e-9
    end = st + np.searchsorted(t[st:], cutoff, "right")
    return t[st:end], s[st:end]


def _fallback_peak(t: NDArray, y: NDArray, fs: float, f_range: Tuple[float, float],
                   f_rough: float, avoid: float | None = None, df_min: float = 0.5 * GHZ,
                   order_burg: int = 8) -> float | None:
    try:
        ar, _ = burg(y - y.mean(), order=order_burg)
        roots = np.roots(np.r_[1, -ar])
        root = roots[np.argmax(np.abs(roots))]
        f_burg = abs(np.angle(root)) * fs / (2 * np.pi)
        if f_range[0] <= f_burg <= f_range[1] and \
           (avoid is None or abs(f_burg - avoid) >= df_min):
            return float(f_burg)
    except Exception:
        pass
    f, P = welch(y, fs=fs, nperseg=min(256, len(y)//2),
                 detrend='constant', scaling='density')
    mask = (f >= f_range[0]) & (f <= f_range[1])
    if avoid is not None:
        mask &= np.abs(f - avoid) >= df_min
    if not np.any(mask):
        return None
    return float(f[mask][np.argmax(P[mask])])


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


def _esprit_freqs_and_decay(r: NDArray, fs: float, p: int = 10
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


def fit_pair(ds_lf: DataSet, ds_hf: DataSet):
    t_lf, y_lf = _crop_signal(ds_lf.ts.t, ds_lf.ts.s, tag="LF")
    t_hf, y_hf = _crop_signal(ds_hf.ts.t, ds_hf.ts.s, tag="HF")

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

    lo = np.array([
        0.5, 0.5,
        C_lf_init - np.std(y_lf), C_hf_init - np.std(y_hf),
        0.0, 0.0,
        tau1_lo, tau2_lo,
        f1_init * 0.9, f2_init * 0.9,
        -PI, -PI
    ])
    hi = np.array([
        2, 2,
        C_lf_init + np.std(y_lf), C_hf_init + np.std(y_hf),
        A1_init * 2, A2_init * 2,
        tau1_hi, tau2_hi,
        f1_init * 1.2, f2_init * 1.2,
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
    ), cost


def process_pair(ds_lf: DataSet, ds_hf: DataSet) -> None:
    tau_guess_lf, tau_guess_hf = 3e-10, 3e-11
    t_lf, y_lf = _crop_signal(ds_lf.ts.t, ds_lf.ts.s, tag="LF")
    t_hf, y_hf = _crop_signal(ds_hf.ts.t, ds_hf.ts.s, tag="HF")
    f1_rough, phi1, A1, tau1 = _single_sine_refine(t_lf, y_lf, f0=10*GHZ)
    proto_lf = A1 * np.exp(-t_hf / tau1) * np.cos(2*np.pi * f1_rough * t_hf + phi1)
    residual = y_hf - proto_lf
    f2_rough, _, _, _ = _single_sine_refine(t_hf, residual, f0=40*GHZ)
    spec_hf = y_hf - np.mean(y_hf)
    f_all_hf, zeta_all_hf = _esprit_freqs_and_decay(spec_hf, ds_hf.ts.meta.fs)
    mask_hf = (zeta_all_hf > 0) & (HF_BAND[0] <= f_all_hf) & (f_all_hf <= HF_BAND[1])
    if np.any(mask_hf):
        hf_cand = list(zip(f_all_hf[mask_hf], zeta_all_hf[mask_hf]))
    else:
        logger.warning(f"({ds_hf.temp_K}, {ds_hf.field_mT}): вызван fallback для HF")
        f2_fallback = _fallback_peak(t_hf, y_hf, ds_hf.ts.meta.fs,
                                    HF_BAND, f2_rough)
        if f2_fallback is None:
            raise RuntimeError("HF-тон не найден")
        hf_cand = [(f2_fallback, None)]
    spec_lf = y_lf - np.mean(y_lf)
    f_all_lf, zeta_all_lf = _esprit_freqs_and_decay(spec_lf, ds_lf.ts.meta.fs)
    mask_lf = (zeta_all_lf > 0) & (LF_BAND[0] <= f_all_lf) & (f_all_lf <= LF_BAND[1])
    if np.any(mask_lf):
        lf_cand = _top2_nearest(f_all_lf[mask_lf], zeta_all_lf[mask_lf], f1_rough)
    else:
        logger.warning(f"({ds_lf.temp_K}, {ds_lf.field_mT}): вызван fallback для LF")
        f1_fallback = _fallback_peak(t_lf, y_lf, ds_lf.ts.meta.fs,
                                    LF_BAND, f1_rough,
                                    avoid=hf_cand[0][0] if np.any(mask_hf) else f2_fallback)
        if f1_fallback is None:
            raise RuntimeError("LF-тон не найден")
        lf_cand = [(f1_fallback, None)]
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
    f1_hz = _peak_in_band(freqs_fft, amps_fft, LF_BAND[0]/GHZ, LF_BAND[1]/GHZ)
    f2_hz = _peak_in_band(freqs_fft, amps_fft, start_band_HF, HF_BAND[1]/GHZ)
    if f1_hz is not None:
        logger.info(
            f"({ds_lf.temp_K}, {ds_lf.field_mT}): найден пик  f1 = {f1_hz/1e9:.1f} ГГц ({LF_BAND[0]/GHZ:.0f}–{LF_BAND[1]/GHZ:.0f})")
    else:
        logger.warning(
            f"({ds_lf.temp_K}, {ds_lf.field_mT}): в полосе {LF_BAND[0]/GHZ:.0f}–{LF_BAND[1]/GHZ:.0f} ГГц пиков не найдено")
    if f2_hz is not None:
        logger.info(
            f"({ds_lf.temp_K}, {ds_lf.field_mT}): найден пик  f2 = {f2_hz/1e9:.1f} ГГц ({start_band_HF:.0f}–{HF_BAND[1]/GHZ:.0f})")
    else:
        logger.warning(
            f"({ds_lf.temp_K}, {ds_lf.field_mT}): в полосе {start_band_HF:.0f}–{HF_BAND[1]/GHZ:.0f} ГГц пиков не найдено")

    def _append_unique(target_list, new_freq_hz):
        if new_freq_hz is None:
            return
        for old_f, _ in target_list:
            if abs(old_f - new_freq_hz) < 20e6:
                return
        target_list.append((new_freq_hz, None))

    _append_unique(lf_cand, f1_hz)
    _append_unique(hf_cand, f2_hz)

    best_cost = np.inf
    best_fit = None
    for f1, z1 in lf_cand:
        for f2, z2 in hf_cand:
            ds_lf.f1_init, ds_lf.zeta1 = f1, z1
            ds_hf.f2_init, ds_hf.zeta2 = f2, z2
            try:
                fit, cost = fit_pair(ds_lf, ds_hf)
            except Exception:
                continue
            if cost < best_cost:
                best_cost = cost
                best_fit = fit
    if best_fit is None:
        raise RuntimeError("Ни одна комбинация не аппроксимировалась")
    ds_lf.fit = ds_hf.fit = best_fit
