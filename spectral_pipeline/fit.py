from __future__ import annotations

from typing import Tuple, List, Optional
import math
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from . import DataSet, FittingResult, GHZ, PI, logger, LF_BAND, HF_BAND
from . import esprit_utils as _esprit_module
from .cwt_candidates import _cwt_gaussian_candidates, _top2_nearest
from .esprit_utils import multichannel_esprit, _esprit_freqs_and_decay
from .spectrum_analysis import (
    _fallback_peak,
    _fft_spectrum,
    _peak_in_band,
)
from .initial_guess import _load_guess, _resolve_tau_bounds, _single_sine_refine
from .numba_models import _core_signal, _numba_residuals, _numba_residuals_single

# Maximum acceptable fitting cost. Pairs with higher cost are rejected
# and treated as unsuccessful.
MAX_COST = 100


def _run_multichannel_esprit(signals: List[NDArray], fs: float, p: int = 6) -> tuple[NDArray, NDArray]:
    """Wrapper to keep monkeypatch compatibility for ESPRIT helpers."""
    _esprit_module._esprit_freqs_and_decay = _esprit_freqs_and_decay
    try:
        return multichannel_esprit(signals, fs, p)
    except TypeError:
        return multichannel_esprit(signals, fs)


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
        w[t >= borders[0]] = 1
        w[t >= borders[1]] = 1
        return w

    w_lf_raw = _piecewise_time_weights(t_lf)

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
        -PI-1e-5, -PI-1e-5
    ])
    hi = np.array([
        2, 2,
        C_lf_init + np.std(y_lf), C_hf_init + np.std(y_hf),
        A1_init * 2, A2_init * 2,
        tau1_hi, tau2_hi,
        f1_hi, f2_hi,
        PI+1e-5, PI+1e-5
    ])

    p0 = np.clip(p0, lo, hi)

    t_all = np.concatenate((t_lf, t_hf))
    y_all = np.concatenate((y_lf, y_hf))
    split_idx = len(t_lf)

    n_lf = y_lf.size
    n_hf = y_hf.size
    norm_lf = 1.0 / math.sqrt(n_lf) if n_lf > 0 else 0.0
    norm_hf = 1.0 / math.sqrt(n_hf) if n_hf > 0 else 0.0
    weights_lf = w_lf_raw * norm_lf
    weights_hf = np.full(n_hf, norm_hf, dtype=y_hf.dtype)
    weights_all = np.concatenate((weights_lf, weights_hf))

    def residuals(p):
        return _numba_residuals(p, t_all, y_all, weights_all, split_idx)

    sol = least_squares(
        residuals,
        p0,
        bounds=(lo, hi),
        method='trf',
        ftol=3e-16, xtol=3e-16, gtol=3e-16,
        max_nfev=20000,
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

    def _sigma(idx: int) -> float:
        try:
            val = float(cov[idx, idx])
        except Exception:
            return float("nan")
        if not np.isfinite(val) or val < 0:
            return float("nan")
        return math.sqrt(val)

    sigma_k_lf = _sigma(0)
    sigma_k_hf = _sigma(1)
    sigma_C_lf = _sigma(2)
    sigma_C_hf = _sigma(3)
    sigma_A1 = _sigma(4)
    sigma_A2 = _sigma(5)
    sigma_tau1 = _sigma(6)
    sigma_tau2 = _sigma(7)
    sigma_f1 = _sigma(8)
    sigma_f2 = _sigma(9)
    sigma_phi1 = _sigma(10)
    sigma_phi2 = _sigma(11)

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
        f_all, zeta_all = _run_multichannel_esprit([spec_lf_c, spec_hf_c], fs_common)
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
        f_all, zeta_all = _run_multichannel_esprit([spec_lf_c, spec_hf_c], fs_common)
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
            k_lf_err=best_fit.k_lf_err,
            k_hf_err=best_fit.k_hf_err,
            C_lf_err=best_fit.C_lf_err,
            C_hf_err=best_fit.C_hf_err,
            A1_err=best_fit.A2_err,
            A2_err=best_fit.A1_err,
            tau1_err=best_fit.tau2_err,
            tau2_err=best_fit.tau1_err,
            phi1_err=best_fit.phi2_err,
            phi2_err=best_fit.phi1_err,
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
    C_init = (
        ds.additive_const_init
        if ds.additive_const_init is not None and np.isfinite(ds.additive_const_init)
        else np.mean(y)
    )
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
        -PI-1e-5,
        -PI-1e-5,
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
        PI+1e-5,
        PI+1e-5,
    ])

    p0 = np.clip(p0, lo, hi)

    def residuals(p):
        return _numba_residuals_single(p, t, y, w)

    sol = least_squares(
        residuals,
        p0,
        bounds=(lo, hi),
        method="trf",
        ftol=3e-16,
        xtol=3e-16,
        gtol=3e-16,
        max_nfev=20000,
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

    def _sigma(idx: int) -> float:
        try:
            val = float(cov[idx, idx])
        except Exception:
            return float("nan")
        if not np.isfinite(val) or val < 0:
            return float("nan")
        return math.sqrt(val)

    sigma_k = _sigma(0)
    sigma_C = _sigma(1)
    sigma_A1 = _sigma(2)
    sigma_A2 = _sigma(3)
    sigma_tau1 = _sigma(4)
    sigma_tau2 = _sigma(5)
    sigma_f1 = _sigma(6)
    sigma_f2 = _sigma(7)
    sigma_phi1 = _sigma(8)
    sigma_phi2 = _sigma(9)

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
            k_lf_err=sigma_k,
            k_hf_err=float("nan"),
            C_lf_err=sigma_C,
            C_hf_err=float("nan"),
            A1_err=sigma_A1,
            A2_err=sigma_A2,
            tau1_err=sigma_tau1,
            tau2_err=sigma_tau2,
            phi1_err=sigma_phi1,
            phi2_err=sigma_phi2,
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
            k_lf_err=best_fit.k_lf_err,
            k_hf_err=best_fit.k_hf_err,
            C_lf_err=best_fit.C_lf_err,
            C_hf_err=best_fit.C_hf_err,
            A1_err=best_fit.A2_err,
            A2_err=best_fit.A1_err,
            tau1_err=best_fit.tau2_err,
            tau2_err=best_fit.tau1_err,
            phi1_err=best_fit.phi2_err,
            phi2_err=best_fit.phi1_err,
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
