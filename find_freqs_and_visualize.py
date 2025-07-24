
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pipeline_esprit.py

Полностью самостоятельный скрипт для обработки LF/HF:
1. load_records + crop_signal  — копировать из оригинала
2. _single_sine_refine, _esprit_freqs_and_decay — копировать из оригинала
3. fit_pair, visualize, export_freq_tables — копировать из оригинала
4. process_pair: coarse refine, ESPRIT HF, ESPRIT LF, общая аппроксимация
5. main: группировка, вызов process_pair, визуализация, экспорт
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
from numpy.fft import rfft, rfftfreq
from dataclasses import dataclass, field
import logging, math
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from openpyxl.styles import Border, Side, Alignment
from typing import Literal, Tuple, List, Dict, Optional
from scipy.optimize import least_squares, differential_evolution
from scipy.signal import welch, find_peaks, get_window, savgol_filter



# ────────────────────────── constants
GHZ = 1e9
NS  = 1e-9
PI  = math.pi
DF_MIN = 0.4 * GHZ
LF_RANGE = (8 * GHZ, 15 * GHZ)
HF_RANGE = (10 * GHZ, 80 * GHZ)
FREQ_TAG = Literal["LF", "HF"]
logger = logging.getLogger("spectral_pipeline")
logger.setLevel(logging.INFO)

# ────────────────────────── dataclasses
@dataclass(slots=True)
class RecordMeta:
    fs: float
@dataclass(slots=True)
class TimeSeries:
    t: NDArray
    s: NDArray
    meta: RecordMeta
@dataclass(slots=True)
class FittingResult:
    f1: float
    f2: float
    zeta1: float
    zeta2: float
    phi1: float
    phi2: float
    A1: float
    A2: float
    k_lf: float
    k_hf: float
    C_lf: float
    C_hf: float
    f1_err: float | None = None
    f2_err: float | None = None
@dataclass(slots=True)
class DataSet:
    field_mT: int
    temp_K: int
    tag: FREQ_TAG
    ts: TimeSeries

    # начальные оценки из Coarse + ESPRIT
    f1_init:  float = 0.0
    f2_init:  float = 0.0
    zeta1:    Optional[float] = None
    zeta2:    Optional[float] = None

    # окончательный результат
    fit: FittingResult | None = field(default_factory=lambda: None)

    # FFT-спектр
    freq_fft: NDArray | None = None
    asd_fft : NDArray | None = None

# ────────────────────────── helpers
def burg(x: NDArray, order: int = 8):
            """
            Простейшая (и быстрая) реализация AR‑Burg.
            Возвращает (ar_coeffs, err_var).
            """
            x = np.asarray(x, dtype=np.float64)
            N = x.size
            if order >= N:
                raise ValueError("order must be < len(x)")
            ef = eb = x[1:].copy()
            a = np.zeros(order + 1); a[0] = 1.0
            E = np.dot(x, x) / N
            for m in range(1, order + 1):
                num = -2.0 * np.dot(eb, ef.conj())
                den = np.dot(ef, ef.conj()) + np.dot(eb, eb.conj())
                k = num / den if den else 0.0
                a_prev = a.copy()
                a[1:m+1] += k * a_prev[m-1::-1]
                ef, eb = ef[1:] + k * eb[1:], eb[:-1] + k * ef[:-1]
                E *= 1 - k.real**2
            return -a[1:], E

def _fft_spectrum(sig: np.ndarray,
                  fs : float,
                  *,
                  window_name: str = "hamming",
                  df_target_GHz: float = 0.1
                 ) -> tuple[np.ndarray, np.ndarray]:
    """
    Амплитудный спектр (ASD) с шагом ≈ df_target_GHz (по умолчанию 1 ГГц).
    Процедура полностью повторяет process_measurement_files_v5:
        • сигнал → убираем DC
        • окно (Hann/Hamming) до паддинга
        • zero-padding до N_req = ceil(fs / (df_target_GHz*1 ГГц))
        • ASD = sqrt( 2·|F|² / (fs * Σwin²) )
    Возвращает частоты (Гц) и ASD-амплитуды.
    """
    N = sig.size
    sig = sig - sig.mean()

    # ---------- окно ----------
    if window_name:
        win = get_window(window_name, N)
        sig = sig * win
        win_sum_sq = np.sum(win ** 2)
    else:                               # «прямоугольное» окно
        win_sum_sq = float(N)

    # ---------- zero-padding до целевого df ----------
    df_target_Hz = df_target_GHz * GHZ
    N_req = int(np.ceil(fs / df_target_Hz))
    N_fft = max(N_req, N)               # без округления к 2^k — как в v5
    sig = np.pad(sig, (0, N_fft - N), mode="constant")

    # ---------- сам FFT ----------
    F = rfft(sig, n=N_fft)
    freqs = rfftfreq(N_fft, d=1/fs)

    # ---------- ASD ----------
    psd = 2.0 * np.abs(F) ** 2 / (fs * win_sum_sq + 1e-16)
    asd = np.sqrt(psd)
    asd = np.nan_to_num(asd)

    return freqs, asd


def _peak_in_band(freqs: np.ndarray,
                  amps : np.ndarray,
                  fmin_GHz: float,
                  fmax_GHz: float
                 ) -> float | None:
    """Лучший (по амплитуде) пик в заданном диапазоне, Hz; None если нет."""
    mask = (freqs >= fmin_GHz*GHZ) & (freqs <= fmax_GHz*GHZ)
    if not mask.any():
        return None
    f_band = freqs[mask]
    a_band = amps [mask]

    # порог = медиана+σ, расстояние ≥0.3 ГГц
    thr  = np.median(a_band) + 0.3 * np.std(a_band)
    df   = f_band[1] - f_band[0] if len(f_band) > 1 else 1*GHZ
    dist = int(0.3*GHZ/df)
    pk, _ = find_peaks(a_band, height=thr, distance=max(1, dist))
    if pk.size == 0:
        return None
    best_idx = pk[np.argmax(a_band[pk])]
    return float(f_band[best_idx])        # Hz

def _crop_signal(t: NDArray, s: NDArray, tag: FREQ_TAG):
    """авто‑обрезка сигнала после глобального максимума"""
    pk = int(np.argmax(s))
    minima = np.where(
        (np.diff(np.signbit(np.diff(s))) > 0) & (np.arange(len(s))[1:-1] > pk)
    )[0]
    st = minima[0] + 1 if minima.size else pk + 1
    cutoff = 0.9e-9 if tag == "LF" else 0.12e-9
    end = st + np.searchsorted(t[st:], cutoff, "right")
    return t[st:end], s[st:end]

def _fallback_peak(
    t: NDArray,
    y: NDArray,
    fs: float,
    f_range: Tuple[float, float],
    f_rough: float,
    avoid: float | None = None,       # частота, от которой держимся
    df_min: float = 0.5*GHZ,
    order_burg: int = 8,
) -> float | None:
    """
    Пытается найти ОДИН пик в диапазоне f_range:
        1) AR-Burg,  2) Welch-PSD.
    Если ничего достоверного – вернёт None.
    """
    # ---------- 1. AR-Burg ---------------------------------------------
    try:
        ar, _ = burg(y - y.mean(), order=order_burg)
        roots = np.roots(np.r_[1, -ar])
        root  = roots[np.argmax(np.abs(roots))]
        f_burg = abs(np.angle(root)) * fs / (2*np.pi)
        if f_range[0] <= f_burg <= f_range[1] and \
           (avoid is None or abs(f_burg - avoid) >= df_min):
            return float(f_burg)
    except Exception:
        pass

    # ---------- 2. Welch-PSD -------------------------------------------
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
    """Берёт ≤2 частоты, ближайшие к f0, вместе с их ζ (или None)."""
    if freqs.size == 0:
        return []
    idx = np.argsort(np.abs(freqs - f0))[:4]
    return [(float(freqs[i]), float(zetas[i])) for i in idx]

def _hankel_matrix(x: NDArray, L: int) -> NDArray:
    """
    Строит Hankel‑матрицу размера L × (K+1), где
    L = число строк, K = N−L (N — длина сигнала).
    """
    N = len(x)
    if L <= 0 or L >= N:
        raise ValueError("L must satisfy 0 < L < N")
    return np.lib.stride_tricks.sliding_window_view(x, L).T

def _esprit_freqs_and_decay(r: NDArray, fs: float, p: int = 10
                            ) -> Tuple[NDArray, NDArray]:
    """
    Оценка p комплексных тонов {f_i, ζ_i} методом ESPRIT
    (ζ — константа экспоненциального затухания, s(t)=A·e^{−ζt}cos(2πft+φ)).

    Возвращает два массива длины ≤ p:   f (Гц) и ζ (1/с).
    """
    # 1. Hankel
    L = len(r) // 2
    H = _hankel_matrix(r, L)          # форма L × K

    # 2. SVD → сигнальное подпространство
    U, _, _ = np.linalg.svd(H, full_matrices=False)
    Us = U[:, :p]                     # L × p

    # 3. Разложение на Us1 / Us2 (строчный сдвиг)
    Us1, Us2 = Us[:-1], Us[1:]
    # Psi = (Us1^H Us1)⁻¹ Us1^H Us2
    Psi = np.linalg.pinv(Us1) @ Us2

    # 4. Собственные значения
    lam = np.linalg.eigvals(Psi)
    dt = 1.0 / fs
    f = np.abs(np.angle(lam)) / (2 * np.pi * dt)        # Гц
    zeta = -np.log(np.abs(lam)) / dt                    # 1/с
    return f, zeta


def _core_signal(t: NDArray,
                 A1, A2, tau1, tau2,
                 f1, f2, phi1, phi2) -> NDArray:
    """Общий шаблон для LF/HF без масштабов и констант."""
    return (
        A1 * np.exp(-t / tau1) * np.cos(2 * np.pi * f1 * t + phi1) +
        A2 * np.exp(-t / tau2) * np.cos(2 * np.pi * f2 * t + phi2)
    )


def _single_sine_refine(t: NDArray, y: NDArray, f0: float
                        ) -> tuple[float, float, float, float]:
    """
    Фит затухающего синуса:
        y(t) = A * exp(-t/tau) * cos(2*pi*f*t + phi) + C
    Возвращает: f, phi, |A|, tau
    """
    A0 = 0.5 * (y.max() - y.min())
    tau0 = (t[-1] - t[0]) / 3

    # [A, f, phi, tau, C]
    p0 = [A0, f0, 0.0, tau0, y.mean()]

    lo = np.array([0.0, 8*GHZ, -np.pi, tau0/10, y.min() - abs(np.ptp(y))])
    hi = np.array([3*A0, 80*GHZ, np.pi, tau0*10, y.max() + abs(np.ptp(y))])

    def model(p):
        A, f, phi, tau, C = p
        return A * np.exp(-t/tau) * np.cos(2*np.pi*f*t + phi) + C

    sol = least_squares(lambda p: model(p) - y, p0, bounds=(lo, hi), method="trf")

    A_est, f_est, phi_est, tau_est, C_est = sol.x
    return f_est, phi_est, A_est, tau_est

# ────────────────────────── I/O: file loader ──────────────────────────
def load_records(root: Path) -> List[DataSet]:
    """
    Читает все *.dat‑файлы в каталоге *root* с именами вида
        *_<field>mT_<temp>K_<HF|LF>_*.dat
    и возвращает список объектов DataSet.

    Формат файла: две колонки (distance, signal).
    Время рассчитывается как  t = 2·(x − x₀) / 3e11  (сек).
    Далее сигнал обрезается _crop_signal(), ограничивается длительностью
    0.6 нс (LF) или 0.1 нс (HF) и упаковывается в DataSet.
    """
    import re

    pattern = re.compile(r"_(\d+)mT_(\d+)K_(HF|LF)_.*\.dat$", re.IGNORECASE)
    datasets: list[DataSet] = []

    for path in root.glob("*.dat"):
        m = pattern.search(path.name)
        if not m:
            continue

        field_mT, temp_K, tag = int(m.group(1)), int(m.group(2)), m.group(3).upper()

        # ───── чтение файла
        try:
            data = np.loadtxt(path, usecols=(0, 1))
        except Exception as exc:
            logger.warning("Невозможно прочитать %s: %s", path.name, exc)
            continue
        if data.ndim != 2 or data.shape[1] != 2 or data.size < 10:
            logger.warning("Пропуск %s: неверный формат/мало точек", path.name)
            continue

        x, s = data[:, 0], data[:, 1]
        x0 = x[np.argmax(s)]
        t = 2.0 * (x - x0) / 3e11  # секунды, скорость = 3·10¹¹ м/с (двойной ход)

        # ───── автообрезка
        cutoff = 0.4e-9 if tag == "LF" else 0.1e-9
        mask = (t >= 0) & (t <= cutoff)
        t, s = t[mask], s[mask]
        if len(t) < 10:
            logger.warning("Пропуск %s: слишком короткий ряд", path.name)
            continue

        dt = float(np.mean(np.diff(t)))
        if not np.isfinite(dt) or dt <= 0:
            logger.warning("Пропуск %s: некорректный шаг dt", path.name)
            continue
        fs = 1.0 / dt  # Гц

        ts = TimeSeries(t=t, s=s, meta=RecordMeta(fs=fs))
        datasets.append(DataSet(field_mT=field_mT, temp_K=temp_K, tag=tag, ts=ts))

        logger.info("Загружен %s: %d точек, fs=%.2f ГГц", path.name, len(t), fs / GHZ)

    return datasets


# ────────────────────────── fit

def fit_pair(ds_lf, ds_hf):
    """
    Нелинейная аппроксимация LF и HF сигналов одновременно.
    Использует найденные параметры для начального приближения, узкие границы
    и весовую функцию для LF.
    Требует:
        ds_lf.ts.meta.zeta1, ds_hf.ts.meta.zeta2 заданы
    """
    # Данные
    t_lf, y_lf = _crop_signal(ds_lf.ts.t, ds_lf.ts.s, tag="LF")
    t_hf, y_hf = _crop_signal(ds_hf.ts.t, ds_hf.ts.s, tag="HF")

    def _piecewise_time_weights(t: np.ndarray) -> np.ndarray:
        """
        Вес = 1.0 в первой трети записи,
              0.8 во второй,
              0.5 в последней.
        """
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

    # Пред вычисления веса для LF
    w_lf = 1 #_piecewise_time_weights(t_lf)

    # Начальные параметры
    f1_init = ds_lf.f1_init
    ζ1_init = ds_lf.zeta1

    f2_init = ds_hf.f2_init
    ζ2_init = ds_hf.zeta2

    # грубые амплитуды/фазы через refine
    _, phi1_init, A1_init, τ1_init = _single_sine_refine(t_lf, y_lf, f1_init)
    if ds_lf.zeta1 is None:
        τ1_lo, τ1_hi = 5e-11, 5e-9
    else:
        τ1_init  = 1.0 / ds_lf.zeta1
        τ1_lo, τ1_hi = τ1_init * 0.8, τ1_init * 1.2

    proto_lf_hf = A1_init * np.exp(-t_hf / τ1_init) * np.cos(2 * PI * f1_init * t_hf + phi1_init)
    _, phi2_init, A2_init, τ2_init= _single_sine_refine(t_hf, y_hf - proto_lf_hf, f2_init)
    if ds_hf.zeta2 is None:
        τ2_lo, τ2_hi = 5e-12, 5e-10
    else:
        τ2_init  = 1.0 / ds_hf.zeta2
        τ2_lo, τ2_hi = τ2_init * 0.8, τ2_init * 1.2

    # грубые масштабы и средние
    k_lf_init = 1
    k_hf_init = 1
    C_lf_init = np.mean(y_lf)
    C_hf_init = np.mean(y_hf)

    # Вектор начального приближения:
    p0 = np.array([
        k_lf_init, k_hf_init,
        C_lf_init, C_hf_init,
        A1_init,    A2_init,
        τ1_init,    τ2_init,
        f1_init,    f2_init,
        phi1_init,  phi2_init
    ])

    lo = np.array([
        0.5,   0.5,
        C_lf_init - np.std(y_lf), C_hf_init - np.std(y_hf),
        0.0,   0.0,
        τ1_lo, τ2_lo,                 # ← вместо прежних коэффициентов
        f1_init * 0.9, f2_init * 0.9,
        -PI,   -PI
    ])
    hi = np.array([
        2, 2,
        C_lf_init + np.std(y_lf), C_hf_init + np.std(y_hf),
        A1_init * 2,  A2_init * 2,
        τ1_hi, τ2_hi,
        f1_init * 1.2, f2_init * 1.2,
        PI,    PI
    ])

    # Функция ошибок
    def residuals(p):
        (k_lf, k_hf, C_lf, C_hf,
        A1, A2, τ1, τ2,
        f1_, f2_, φ1_, φ2_) = p

        core_lf = _core_signal(t_lf, A1, A2, τ1, τ2, f1_, f2_, φ1_, φ2_)
        core_hf = _core_signal(t_hf, A1, A2, τ1, τ2, f1_, f2_, φ1_, φ2_)
        res_lf = w_lf * (k_lf * core_lf + C_lf - y_lf)
        res_hf = 1 * (k_hf * core_hf + C_hf - y_hf)
        return np.concatenate([res_lf, res_hf])

    sol = least_squares(
        residuals,
        p0,
        bounds=(lo, hi),
        method='trf',
        # более строгие критерии
        ftol=1e-15, xtol=1e-15, gtol=1e-15,
        # ↑-чувствительность к малым улучшениям
        max_nfev=100000,        # ← больше итераций
        loss='soft_l1',        # ← робастная метрика
        f_scale=0.1,           # ← чувствительность soft-L1
        x_scale='jac',         # ← масштабируем переменные
        verbose=0
    )
    p = sol.x
    cost = sol.cost

    # ковариационная матрица параметров
    #  J — Якобиан, m = число точек, n = число параметров
    m, n = sol.jac.shape
    # защита от сингулярности
    try:
        cov = np.linalg.inv(sol.jac.T @ sol.jac) * 2*cost / max(m - n, 1)
    except np.linalg.LinAlgError:
        cov = np.full((n, n), np.nan)

    # индексы f1 и f2 в векторе p0            (см. порядок параметров)
    idx_f1 = 8
    idx_f2 = 9
    σ_f1 = math.sqrt(abs(cov[idx_f1, idx_f1]))
    σ_f2 = math.sqrt(abs(cov[idx_f2, idx_f2]))

    # Разбор результата
    (k_lf, k_hf, C_lf, C_hf,
    A1, A2, τ1, τ2,
    f1_fin, f2_fin, φ1_fin, φ2_fin) = p

    return FittingResult(
        f1     = f1_fin,
        f2     = f2_fin,
        zeta1  = 1/τ1,
        zeta2  = 1/τ2,
        phi1   = φ1_fin,
        phi2   = φ2_fin,
        A1     = A1,
        A2     = A2,
        k_lf   = k_lf,
        k_hf   = k_hf,
        C_lf   = C_lf,
        C_hf   = C_hf,
        f1_err = σ_f1,
        f2_err = σ_f2,
    ), cost

# ────────────────────────── main

def process_pair(ds_lf, ds_hf):
    """
    Обрабатывает пару LF/HF полностью внутри этого скрипта.
    """
    tau_guess_lf, tau_guess_hf = 3e-10, 3e-11

    t_lf, y_lf = _crop_signal(ds_lf.ts.t, ds_lf.ts.s, tag="LF")
    t_hf, y_hf = _crop_signal(ds_hf.ts.t, ds_hf.ts.s, tag="HF")

    # 1. Coarse refine LF
    f1_rough, phi1, A1, tau1 = _single_sine_refine(t_lf, y_lf, f0=10*GHZ)

    # 2. Coarse refine HF (LF-прототип)
    proto_lf = A1 * np.exp(-t_hf / tau1) * np.cos(2*np.pi * f1_rough * t_hf + phi1)
    residual = y_hf - proto_lf
    f2_rough, _, _, _ = _single_sine_refine(t_hf, residual, f0=40*GHZ)

    # 3. ESPRIT на HF
    spec_hf = y_hf - np.mean(y_hf)
    f_all_hf, zeta_all_hf = _esprit_freqs_and_decay(spec_hf, ds_hf.ts.meta.fs)
    mask_hf = (zeta_all_hf > 0) & (10*GHZ <= f_all_hf) & (f_all_hf <= 80*GHZ)
    hf_cand: List[Tuple[float, Optional[float]]]

    if np.any(mask_hf):
        hf_cand = list(zip(f_all_hf[mask_hf], zeta_all_hf[mask_hf]))
    else:
        logger.warning(f"({ds_hf.temp_K}, {ds_hf.field_mT}): вызван fallback для HF")
        f2_fallback = _fallback_peak(
            t_hf, y_hf, ds_hf.ts.meta.fs,
            (10*GHZ, 80*GHZ), f2_rough
        )
        if f2_fallback is None:
            raise RuntimeError("HF-тон не найден")
        hf_cand = [(f2_fallback, None)]

    # 4. ESPRIT на LF
    spec_lf = y_lf - np.mean(y_lf)
    f_all_lf, zeta_all_lf = _esprit_freqs_and_decay(spec_lf, ds_lf.ts.meta.fs)
    mask_lf = (zeta_all_lf > 0) & (8*GHZ <= f_all_lf) & (f_all_lf <= 12*GHZ)

    lf_cand: List[Tuple[float, Optional[float]]]
    if np.any(mask_lf):
        lf_cand = _top2_nearest(f_all_lf[mask_lf], zeta_all_lf[mask_lf], f1_rough)
    else:
        logger.warning(f"({ds_lf.temp_K}, {ds_lf.field_mT}): вызван fallback для LF")
        f1_fallback = _fallback_peak(
            t_lf, y_lf, ds_lf.ts.meta.fs,
            (8*GHZ, 12*GHZ), f1_rough,
            avoid=hf_cand[0][0] if np.any(mask_hf) else f2_fallback
        )
        if f1_fallback is None:
            raise RuntimeError("LF-тон не найден")
        lf_cand = [(f1_fallback, None)]

    # 5. ДОБАВЛЯЕМ ДВА FFT-ПИКА
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
    if start_band_HF > 30.0: start_band_HF = 30.0

    f1_hz = _peak_in_band(freqs_fft, amps_fft,  8.0, 12.0)   # 8–12 ГГц
    f2_hz = _peak_in_band(freqs_fft, amps_fft, start_band_HF, 80.0)   # 20–70 ГГц

    if f1_hz is not None:
        logger.info(f"({ds_lf.temp_K}, {ds_lf.field_mT}): найден пик  f1 = {f1_hz/1e9:.1f} ГГц (8–12)")
    else:
        logger.warning(f"({ds_lf.temp_K}, {ds_lf.field_mT}): в полосе 8–12 ГГц пиков не найдено")

    if f2_hz is not None:
        logger.info(f"({ds_lf.temp_K}, {ds_lf.field_mT}): найден пик  f2 = {f2_hz/1e9:.1f} ГГц ({start_band_HF:.0f}–70)")
    else:
        logger.warning(f"({ds_lf.temp_K}, {ds_lf.field_mT}): в полосе {start_band_HF:.0f}–70 ГГц пиков не найдено")

    # ---- добавляем уникальные FFT-пики в списки кандидатов ----
    def _append_unique(target_list, new_freq_hz):
        if new_freq_hz is None:
            return
        for old_f, _ in target_list:
            if abs(old_f - new_freq_hz) < 20e6:      # 20 МГц допуск
                return                               # уже есть
        target_list.append((new_freq_hz, None))

    # f1 лежит ниже 12 ГГц → это всё-таки LF-тон
    _append_unique(lf_cand, f1_hz)
    # f2 – HF-тон
    _append_unique(hf_cand, f2_hz)

    # 6. Общая аппроксимация через fit_pair
    best_cost = np.inf
    best_fit  = None

    for f1, z1 in lf_cand:
        for f2, z2 in hf_cand:
            ds_lf.f1_init, ds_lf.zeta1 = f1, z1
            ds_hf.f2_init, ds_hf.zeta2 = f2, z2
            try:
                fit, cost = fit_pair(ds_lf, ds_hf)
            except Exception:
                continue             # аппроксимация не сошлась

            if cost < best_cost:
                best_cost = cost
                best_fit  = fit

    if best_fit is None:
        raise RuntimeError("Ни одна комбинация не аппроксимировалась")

    ds_lf.fit = ds_hf.fit = best_fit

# ────────────────────────── plotting

def visualize(triples: list[tuple[DataSet, DataSet, dict[str, float]]]):
    if not triples:
        print("Нет данных для визуализации."); return

    by_key = {(lf.field_mT, lf.temp_K):(lf,hf) for lf,hf in triples}
    keys   = sorted(by_key)                          # (H,T)

    # ── серия для сводных графиков --------------------------------------
    freq_vs_H: dict[int, list[tuple[int,float,float]]] = {}   # T → [(H,fLF,fHF)]
    freq_vs_T: dict[int, list[tuple[int,float,float]]] = {}   # H → [(T,fLF,fHF)]

    for (H,T), (lf, hf) in by_key.items():
        p = lf.fit
        if p is None:        # фит неудачный → нет частот
            continue
        f_LF, f_HF = sorted((p.f1 / GHZ, p.f2 / GHZ))
        freq_vs_H.setdefault(T, []).append((H,f_LF,f_HF))
        freq_vs_T.setdefault(H, []).append((T,f_LF,f_HF))

    freq_vs_H = {T:sorted(v) for T,v in freq_vs_H.items() if len(v)>=2}
    freq_vs_T = {H:sorted(v) for H,v in freq_vs_T.items() if len(v)>=2}

    # ── разобьем сводные зависимости на две группы -------------------------
    series_T = sorted(freq_vs_T.items())   # f(T) @ H_const   → в ЛЕВОЙ кол.
    series_H = sorted(freq_vs_H.items())   # f(H) @ T_const   → в ПРАВОЙ кол.

    have_T = bool(series_T)
    have_H = bool(series_H)

    # ── сколько строк нужно под сводные графики -------------------------
    if have_T and have_H:
        n_rows_summary = max(len(series_T), len(series_H))
    else:                                   # только один тип → размещаем 2‑в‑ряд
        n_rows_summary = math.ceil(max(len(series_T), len(series_H)) / 2)

    rows = 1 + n_rows_summary              # + первая строка с сигналами

    # ── specs: описываем, где есть подграфики ---------------------------
    specs = [[{"type": "xy"}, {"type": "xy"}]]       # первая строка (LF | HF)
    subplot_titles = ["LF signal (selected)", "HF signal (selected)"]

    for r in range(n_rows_summary):
        row_spec = []
        # кол‑1 (f(T) @ H_const)
        if have_T and r < len(series_T):            # есть такая серия
            row_spec.append({"type": "xy"})
            subplot_titles.append(
                f"f_LF, f_HF vs T  (H = {series_T[r][0]} mT)"
            )
        else:
            row_spec.append(None)
        # кол‑2 (f(H) @ T_const)
        if have_H and r < len(series_H):
            row_spec.append({"type": "xy"})
            subplot_titles.append(
                f"f_LF, f_HF vs H  (T = {series_H[r][0]} K)"
            )
        else:
            row_spec.append(None)
        specs.append(row_spec)

    fig = make_subplots(
        rows=rows, cols=2, specs=specs,
        column_widths=[0.5, 0.5],
        horizontal_spacing=0.08, vertical_spacing=0.12,
        subplot_titles=tuple(subplot_titles)
    )

    # ── динамические трассы (Dropdown) ----------------------------------
    trace_ranges, idx = {}, 0
    for (H, T) in keys:
        ds_lf, ds_hf = by_key[(H, T)]
        p = ds_lf.fit
        s = idx

        # --- raw LF --------------------------------------------------------
        fig.add_trace(go.Scattergl(x=ds_lf.ts.t/NS, y=ds_lf.ts.s,
                                  name=f"LF raw ({H} mT, {T} K)",
                                  line=dict(width=1), visible=False, showlegend=False),
                      1, 1); idx += 1

        # --- LF fit (если есть) -------------------------------------------
        if p is not None:
            core = lambda ts: _core_signal(
                ts.t, p.A1, p.A2,
                1/p.zeta1, 1/p.zeta2,  # если нужны tau1, tau2 в нс, домножьте на 1e9
                p.f1, p.f2,
                p.phi1, p.phi2)
            fit_lf = p.k_lf * core(ds_lf.ts) + p.C_lf
            fig.add_trace(go.Scattergl(x=ds_lf.ts.t/NS, y=fit_lf,
                                      name="LF fit", line=dict(dash="dash", color="#e67e22"),
                                      visible=False),
                          1, 1); idx += 1

            # Отдельно синус HF-компоненты
            hf_only = p.k_lf * (
                p.A2 * np.exp(-ds_lf.ts.t / (1/p.zeta2)) * np.cos(2*PI*p.f2*ds_lf.ts.t + p.phi2)
            ) + p.C_lf
            fig.add_trace(go.Scattergl(
                x=ds_lf.ts.t/NS, y=hf_only,
                name="HF component only",
                line=dict(dash="dot", width=2, color="royalblue"),
                opacity=0.25,
                visible=False
            ), 1, 1); idx += 1

            # Отдельно синус LF-компоненты
            lf_only = p.k_lf * (
                p.A1 * np.exp(-ds_lf.ts.t / (1/p.zeta1)) * np.cos(2*PI*p.f1*ds_lf.ts.t + p.phi1)
            ) + p.C_lf
            fig.add_trace(go.Scattergl(
                x=ds_lf.ts.t/NS, y=lf_only,
                name="LF component only",
                line=dict(dash="dot", width=2, color="indianred"),
                opacity=0.25,
                visible=False
            ), 1, 1); idx += 1

        # --- raw HF --------------------------------------------------------
        fig.add_trace(go.Scattergl(x=ds_hf.ts.t/NS, y=ds_hf.ts.s,
                                  name=f"HF raw ({H} mT, {T} K)",
                                  line=dict(width=1), visible=False, showlegend=False),
                      1, 2); idx += 1

        # --- HF fit (если есть) -------------------------------------------
        if p is not None:
            fit_hf = p.k_hf * core(ds_hf.ts) + p.C_hf
            fig.add_trace(go.Scattergl(x=ds_hf.ts.t/NS, y=fit_hf,
                                      name="HF fit", line=dict(dash="dash", color="#2971c5"),
                                      visible=False),
                          1, 2); idx += 1

            # Отдельно синус HF-компоненты
            hf_only = p.k_hf * (
                p.A2 * np.exp(-ds_hf.ts.t / (1/p.zeta2)) * np.cos(2*PI*p.f2*ds_hf.ts.t + p.phi2)
            ) + p.C_hf
            fig.add_trace(go.Scattergl(
                x=ds_hf.ts.t/NS, y=hf_only,
                name="HF component only",
                line=dict(dash="dot", width=2, color="royalblue"),
                opacity=0.25,
                visible=False
            ), 1, 2); idx += 1

            # Отдельно синус LF-компоненты (на HF-графике)
            lf_only = p.k_hf * (
                p.A1 * np.exp(-ds_hf.ts.t / (1/p.zeta1)) * np.cos(2*PI*p.f1*ds_hf.ts.t + p.phi1)
            ) + p.C_hf
            fig.add_trace(go.Scattergl(
                x=ds_hf.ts.t/NS, y=lf_only,
                name="LF component only",
                line=dict(dash="dot", width=2, color="indianred"),
                opacity=0.25,
                visible=False
            ), 1, 2); idx += 1

        trace_ranges[(H, T)] = (s, idx)

    if keys:
        s0, e0 = trace_ranges[keys[0]]
        for i in range(s0, e0):
            fig.data[i].visible = True

    idx_static = idx   # всё, что ниже, статично

    # ── добавляем сводные графики ---------------------------------------
    #   СЛУЧАЙ A: есть и series_T, и series_H  → рисуем их в две колонки
    if have_T and have_H:
        # 1‑я колонка: series_T  (фикс. H)
        for t_idx, (H_fix, pts) in enumerate(series_T):
            row = 2 + t_idx
            T_vals, fLF, fHF = zip(*pts)
            fig.add_trace(go.Scatter(x=T_vals, y=fLF, mode='markers+lines',
                                    name=f"f_L {H_fix} mT", showlegend=False,
                                    line=dict(color='red')),
                          row=row, col=1)
            fig.add_trace(go.Scatter(x=T_vals, y=fHF, mode='markers+lines',
                                    name=f"f_HF {H_fix} mT", showlegend=False,
                                    line=dict(color='blue')),
                          row=row, col=1)
            fig.update_xaxes(title_text="T (K)", row=row, col=1)
            fig.update_yaxes(title_text="frequency (GHz)", row=row, col=1)

        #   2‑я колонка: series_H  (фикс. T), тоже сверху‑вниз
        for h_idx, (T_fix, pts) in enumerate(series_H):
            row = 2 + h_idx
            H_vals, fLF, fHF = zip(*pts)
            fig.add_trace(go.Scatter(x=H_vals, y=fLF, mode='markers+lines',
                                    name=f"f_LF {T_fix} K", showlegend=False),
                                    line=dict(color='red'),
                          row=row, col=2)
            fig.add_trace(go.Scatter(x=H_vals, y=fHF, mode='markers+lines',
                                    name=f"f_HF {T_fix} K", showlegend=False,
                                    line=dict(color='blue')),
                          row=row, col=2)
            fig.update_xaxes(title_text="H (mT)", row=row, col=2)
            fig.update_yaxes(title_text="frequency (GHz)", row=row, col=2)

    # СЛУЧАЙ B: есть ТОЛЬКО один тип зависимостей  → разложим «2‑в‑ряд»
    else:
        sole_series = series_T if have_T else series_H
        for idx_s, (fix, pts) in enumerate(sole_series):
            row = 2 + idx_s // 2
            col = 1 + idx_s % 2                # сначала пытаемся (row, col)

            # >>> NEW: если такой подграфик не создан, берём соседний
            if specs[row-1][col-1] is None:    # row‑1, col‑1  — индексы списка
                col = 2 if specs[row-1][1] is not None else 1
            # <<<

            x, fLF, fHF = zip(*pts)
            if have_T:                         # фиксирован H
                xs_label = "T (K)"; fix_lbl = f"{fix} mT"
            else:                              # фиксирована T
                xs_label = "H (mT)"; fix_lbl = f"{fix} K"

            fig.add_trace(go.Scatter(x=x, y=fLF, mode='markers+lines',
                                    name=f"f_LF {fix_lbl}", showlegend=False,
                                    line=dict(color='red')),
                          row=row, col=col)
            fig.add_trace(go.Scatter(x=x, y=fHF, mode='markers+lines',
                                    name=f"f_HF {fix_lbl}", showlegend=False,
                                    line=dict(color='blue')),
                          row=row, col=col)
            fig.update_xaxes(title_text=xs_label, row=row, col=col)
            fig.update_yaxes(title_text="frequency (GHz)", row=row, col=col)

    # ── Dropdown ---------------------------------------------------------
    buttons, n_tot = [], len(fig.data)
    for (H,T) in keys:
        vis = [False]*n_tot
        s,e = trace_ranges[(H,T)]
        for i in range(s,e): vis[i]=True
        for i in range(idx_static,n_tot): vis[i]=True  # статичные всегда
        buttons.append(dict(label=f"{H} mT, {T} K",
                            method="update", args=[{"visible":vis}]))

    fig.update_layout(updatemenus=[dict(buttons=buttons, active=0,
                                        x=0.2, y=1.15, showactive=True)])
    fig.update_layout(height=350*rows, hovermode="x unified")
    fig.update_yaxes(title_text="signal (a.u.)", row=1,col=1)
    fig.update_yaxes(title_text="signal (a.u.)", row=1,col=2)
    print("\nОтображение графика…"); fig.show()


def visualize_stacked(triples: list[tuple[DataSet, DataSet]],
                      *, title: str | None = None) -> None:
    """
    Отображает все LF/HF-сигналы и их аппроксимации вертикально сдвинутыми.
    + Сводный график f_LF и f_HF (как в visualize) в третьей колонке первой строки.
    """
    if not triples:
        return

    RAW_CLR   = "#1fbe63"
    FIT_LF    = "red"
    FIT_HF    = "blue"
    BASE_CLR  = "#606060"

    # ── что меняется: H или T?
    all_H = {ds_lf.field_mT for ds_lf, _ in triples}
    all_T = {ds_lf.temp_K   for ds_lf, _ in triples}
    if   len(all_H) == 1 and len(all_T) > 1:
        varying, var_label = "T", "K"
        key_func = lambda ds_lf: ds_lf.temp_K
    elif len(all_T) == 1 and len(all_H) > 1:
        varying, var_label = "H", "mT"
        key_func = lambda ds_lf: ds_lf.field_mT
    else:
        raise RuntimeError("Скрипт ожидает изменение только H или T.")

    triples_sorted = sorted(triples, key=lambda p: key_func(p[0]))

    # ── вертикальный шаг
    ranges = []
    for pair in triples_sorted:
        for ds in pair:
            if ds.fit is not None:
                mask = ds.ts.t > 0
                p = ds.fit
                core = _core_signal(
                    ds.ts.t, p.A1, p.A2,
                    1/p.zeta1, 1/p.zeta2,
                    p.f1, p.f2, p.phi1, p.phi2
                )
                y_fit = p.k_lf * core + p.C_lf if ds.tag == 'LF' else p.k_hf * core + p.C_hf
                ranges.append(np.ptp(y_fit[mask]))
            else:
                ranges.append(np.ptp(ds.ts.s))
    y_step = 0.8 * max(ranges + [1e-3])

    # ── собираем данные для сводных графиков
    freq_vs_H: dict[int, list[tuple[int,float,float]]] = {}
    freq_vs_T: dict[int, list[tuple[int,float,float]]] = {}
    amp_vs_H: dict[int, list[tuple[int,float,float]]] = {}
    amp_vs_T: dict[int, list[tuple[int,float,float]]] = {}
    for ds_lf, ds_hf in triples_sorted:
        if ds_lf.fit is None:
            continue
        H, T = ds_lf.field_mT, ds_lf.temp_K
        f1, f2 = sorted((ds_lf.fit.f1/GHZ, ds_lf.fit.f2/GHZ))
        freq_vs_H.setdefault(T, []).append((H, f1, f2))
        freq_vs_T.setdefault(H, []).append((T, f1, f2))
        amp_vs_H.setdefault(T, []).append((H, ds_lf.fit.A1, ds_lf.fit.A2))
        amp_vs_T.setdefault(H, []).append((T, ds_lf.fit.A1, ds_lf.fit.A2))
    freq_vs_H = {T:sorted(v) for T,v in freq_vs_H.items() if len(v) >= 2}
    freq_vs_T = {H:sorted(v) for H,v in freq_vs_T.items() if len(v) >= 2}
    amp_vs_H = {T:sorted(v) for T,v in amp_vs_H.items() if len(v) >= 2}
    amp_vs_T = {H:sorted(v) for H,v in amp_vs_T.items() if len(v) >= 2}

    have_T, have_H = bool(freq_vs_T), bool(freq_vs_H)
    # n_rows_summary больше не нужен

    # ── макет подграфиков ──────────────────────────────────────────────
    specs = [
        [  # Row 1
            {"type": "xy", "rowspan": 2},           # 1 – сигналы LF
            {"type": "xy", "rowspan": 2},           # 2 – сигналы HF
            {"type": "xy", "rowspan": 2},           # 3 – спектры
            {"type": "xy"},                         # 4 – f-сводка
        ],
        [  # Row 2
            None,
            None,
            None,
            {"type": "xy"},                         # 4 – amp-сводка
        ],
    ]
    if varying == "T":
        fixed_H = list(all_H)[0]
        titles = [
            f"LF signals (H = {fixed_H} mT)",
            f"HF signals (H = {fixed_H} mT)",
            f"Spectra (H = {fixed_H} mT)",
            f"Frequencies (H = {fixed_H} mT)",
            f"Amplitudes (H = {fixed_H} mT)"
        ]
    elif varying == "H":
        fixed_T = list(all_T)[0]
        titles = [
            f"LF signals (T = {fixed_T} K)",
            f"HF signals (T = {fixed_T} K)",
            f"Spectra (T = {fixed_T} K)",
            f"Frequencies (T = {fixed_T} K)",
            f"Amplitudes (T = {fixed_T} K)"
        ]
    else:
        titles = ["LF signals", "HF signals", "Spectra", "Frequencies", "Amplitudes"]

    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    fig = make_subplots(
        rows=2, cols=4, specs=specs,
        column_widths=[0.26, 0.26, 0.26, 0.22],
        horizontal_spacing=0.06,
        vertical_spacing=0.15,
        subplot_titles=tuple(titles)
    )

    # ── сигналы (первая строка, колонки 1 и 2)
    for idx, (ds_lf, ds_hf) in enumerate(triples_sorted):
        shift      = (idx + 1) * y_step
        var_value  = key_func(ds_lf)
        tmin_lf, tmax_lf = ds_lf.ts.t[0]/NS, ds_lf.ts.t[-1]/NS
        tmin_hf, tmax_hf = ds_hf.ts.t[0]/NS, ds_hf.ts.t[-1]/NS

        for col, (tmin, tmax) in ((1, (tmin_lf, tmax_lf)), (2, (tmin_hf, tmax_hf))):
            fig.add_trace(
                go.Scattergl(
                    x=[tmin, tmax], y=[shift, shift],
                    line=dict(width=1, color=BASE_CLR),
                    mode="lines",
                    showlegend=False, hoverinfo="skip"
                ), row=1, col=col)

        # LF raw + fit
        if ds_lf.fit: p = ds_lf.fit
        y = ds_lf.ts.s + shift
        y -= p.C_lf if ds_lf.fit else ds_lf.ts.s.mean()
        fig.add_trace(
            go.Scattergl(
                x=ds_lf.ts.t/NS, y=y,
                line=dict(width=3, color=RAW_CLR),
                name=f"{varying} = {var_value} {var_label}"
            ), 1, 1)

        if ds_lf.fit:
            core  = _core_signal(ds_lf.ts.t, p.A1, p.A2,
                                1/p.zeta1, 1/p.zeta2,
                                p.f1, p.f2, p.phi1, p.phi2)
            y_fit = p.k_lf * core + shift
            fig.add_trace(
                go.Scattergl(
                    x=ds_lf.ts.t/NS, y=y_fit,
                    line=dict(width=2, dash="dash", color=FIT_LF),
                    name=f"{varying} = {var_value} {var_label}"
                ), 1, 1)

        # HF raw + fit
        if ds_hf.fit: p = ds_hf.fit
        y = ds_hf.ts.s + shift
        y -= p.C_hf if ds_hf.fit else ds_hf.ts.s.mean()
        fig.add_trace(
            go.Scattergl(
                x=ds_hf.ts.t/NS, y=y,
                line=dict(width=3, color=RAW_CLR),
                name=f"{varying} = {var_value} {var_label}"
            ), 1, 2)

        if ds_hf.fit:
            core  = _core_signal(ds_hf.ts.t, p.A1, p.A2,
                                1/p.zeta1, 1/p.zeta2,
                                p.f1, p.f2, p.phi1, p.phi2)
            y_fit = p.k_hf * core + shift
            fig.add_trace(
                go.Scattergl(
                    x=ds_hf.ts.t/NS, y=y_fit,
                    line=dict(width=2, dash="dash", color=FIT_HF),
                    name=f"{varying} = {var_value} {var_label}"
                ), 1, 2)

        # подписи у хвоста
        fig.add_annotation(x=tmax_lf, y=shift,
                          text=f"{var_value} {var_label}",
                          showarrow=False, xanchor="left", font=dict(size=16),
                          row=1, col=1)
        fig.add_annotation(x=tmax_hf, y=shift,
                          text=f"{var_value} {var_label}",
                          showarrow=False, xanchor="left", font=dict(size=16),
                          row=1, col=2)

    # ── Сводный график в колонке 4 первой строки
    if varying == "T":
        for H_fix, pts in freq_vs_T.items():
            T_vals, fLF, fHF = zip(*pts)
            fig.add_trace(go.Scatter(x=T_vals, y=fLF, mode="markers+lines",
                                     line=dict(width=2, color='red'),
                                     name=f"f_LF, H = {H_fix} mT"),
                          row=1, col=4)
            fig.add_trace(go.Scatter(x=T_vals, y=fHF, mode="markers+lines",
                                     line=dict(width=2, color='blue'),
                                     name=f"f_HF, H = {H_fix} mT"),
                          row=1, col=4)
        fig.update_xaxes(title_text="T (K)", row=1, col=4)
        for H_fix, pts in amp_vs_T.items():
            T_vals, amp_LF, amp_HF = zip(*pts)
            fig.add_trace(go.Scatter(x=T_vals, y=amp_LF, mode="markers+lines",
                                     line=dict(width=2, color='red'),
                                     name=f"f_LF, H = {H_fix} mT"),
                          row=2, col=4)
            fig.add_trace(go.Scatter(x=T_vals, y=amp_HF, mode="markers+lines",
                                     line=dict(width=2, color='blue'),
                                     name=f"f_HF, H = {H_fix} mT"),
                          row=2, col=4)
        fig.update_xaxes(title_text="T (K)", row=2, col=4)
    else:  # varying == "H"
        for T_fix, pts in freq_vs_H.items():
            H_vals, fLF, fHF = zip(*pts)
            fig.add_trace(go.Scatter(x=H_vals, y=fLF, mode="markers+lines",
                                     line=dict(width=2, color='red'),
                                     name=f"f_LF, T = {T_fix} K"),
                          row=1, col=4)
            fig.add_trace(go.Scatter(x=H_vals, y=fHF, mode="markers+lines",
                                     line=dict(width=2, color='blue'),
                                     name=f"f_HF, T = {T_fix} K"),
                          row=1, col=4)
        fig.update_xaxes(title_text="H (mT)", row=1, col=4)
        for T_fix, pts in amp_vs_H.items():
            H_vals, amp_LF, amp_HF = zip(*pts)
            fig.add_trace(go.Scatter(x=H_vals, y=amp_LF, mode="markers+lines",
                                     line=dict(width=2, color='red'),
                                     name=f"f_LF, T = {T_fix} K"),
                          row=2, col=4)
            fig.add_trace(go.Scatter(x=H_vals, y=amp_HF, mode="markers+lines",
                                     line=dict(width=2, color='blue'),
                                     name=f"f_HF, T = {T_fix} K"),
                          row=2, col=4)
        fig.update_xaxes(title_text="H (mT)", row=2, col=4)
    fig.update_yaxes(title_text="f (GHz)", row=1, col=4)
    fig.update_yaxes(title_text="amp", row=2, col=4)

    spectra_HF: list[tuple[np.ndarray, np.ndarray, str]] = []   # (f_GHz, ASD, depth, label)
    spectra_LF: list[tuple[np.ndarray, np.ndarray, str]] = []

    for ds_lf, ds_hf in triples_sorted:
        if ds_hf.freq_fft is None or ds_lf.freq_fft is None:
            continue
        depth_val = ds_hf.temp_K if varying == "T" else ds_hf.field_mT
        spectra_HF.append((
            ds_hf.freq_fft / GHZ,                    # freq_GHz  (np.ndarray)
            ds_hf.asd_fft / np.max(ds_hf.asd_fft),      # asd_amp   (np.ndarray)
            f"{depth_val:.0f} {var_label}"           # label_str (str)
        ))
        spectra_LF.append((
            ds_lf.freq_fft / GHZ,
            ds_lf.asd_fft / np.max(ds_lf.asd_fft),
            f"{depth_val:.0f} {var_label}"
        ))

    if spectra_HF:                                     # если есть что рисовать
        spectra_HF.sort(key=lambda tpl: tpl[2])
        spectra_LF.sort(key=lambda tpl: tpl[2])
        shift_hf = 1.2 * max(np.nanmax(a_hf) for (_, a_hf, _) in spectra_HF)
        shift_lf = 1.2 * max(np.nanmax(a_lf) for (_, a_lf, _) in spectra_LF)
        shift_f = max(shift_hf, shift_lf)

        for idx, ((f_GHz, amp, lbl), (f_lf, amp_lf, _)) in enumerate(zip(spectra_HF, spectra_LF)):
            offset = (idx + 1) * shift_f
            y_vals = amp + offset
            f_GHz = f_GHz[:np.argmin(np.abs(f_GHz - 80))]

            fig.add_trace(
                go.Scattergl(
                    x=f_GHz, y=y_vals,
                    mode="lines",
                    line=dict(color=FIT_HF, width=2),
                    name=lbl, showlegend=False,
                ),
                row=1, col=3
            )

            y_vals = amp_lf + offset
            f_lf = f_lf[:np.argmin(np.abs(f_lf - 80))]

            fig.add_trace(
                go.Scattergl(
                    x=f_lf, y=y_vals,
                    mode="lines",
                    line=dict(color=FIT_LF, width=2),
                    name=lbl, showlegend=False,
                ),
                row=1, col=3
            )

            fig.add_annotation(
                x=f_GHz[-1], y=offset,
                text=lbl, xanchor="left", showarrow=False,
                font=dict(size=16), row=1, col=3
            )

            fig.add_trace(
                go.Scattergl(
                    x=[f_GHz[0], f_GHz[-1]], y=[offset, offset],
                    line=dict(width=1, color=BASE_CLR),
                    mode="lines",
                    showlegend=False, hoverinfo="skip"
                ), row=1, col=3)

    # ── оформление
    fig.update_layout(
        showlegend=False, hovermode="x unified",
        font=dict(size=16),
        width=2000, height=1000,  # 4:1 разрешение
        paper_bgcolor="white",   # фон всего полотна
        plot_bgcolor="white"     # фон каждой ячейки
    )
    for annotation in fig['layout']['annotations'][:len(titles)]:
        annotation['font'] = dict(size=22)

    fig.update_xaxes(
        showline=True, linewidth=1, linecolor="black",
        mirror=True, showticklabels=True, ticks="inside",
        showgrid=True, gridcolor="#cccccc", gridwidth=1,
        row=1, col=1, title_text="time (ns)"
    )
    fig.update_xaxes(
        showline=True, linewidth=1, linecolor="black",
        mirror=True, showticklabels=True, ticks="inside",
        showgrid=True, gridcolor="#cccccc", gridwidth=1,
        row=1, col=2, title_text="time (ns)"
    )
    fig.update_xaxes(
        showline=True, linewidth=1, linecolor="black",
        mirror=True, showticklabels=True, ticks="inside",
        showgrid=True, gridcolor="#cccccc", gridwidth=1,
        row=1, col=3, title_text="frequency (GHz)"
    )
    fig.update_yaxes(
        range=[0, shift + y_step],
        showline=True, linewidth=1, linecolor="black",
        mirror=True, showticklabels=False,
        row=1, col=1
    )
    fig.update_yaxes(
        range=[0, shift + y_step],
        showline=True, linewidth=1, linecolor="black",
        mirror=True, showticklabels=False,
        row=1, col=2
    )
    fig.update_yaxes(
        range=[0, offset + shift_f],
        showline=True, linewidth=1, linecolor="black",
        mirror=True, showticklabels=False,
        row=1, col=3
    )
    fig.update_xaxes(
        showline=True, linewidth=1, linecolor="black",
        mirror=True, showgrid=True, gridcolor="#cccccc",
        gridwidth=1, row=1, col=4
    )
    fig.update_yaxes(
        showline=True, linewidth=1, linecolor="black",
        mirror=True, showgrid=True, gridcolor="#cccccc",
        gridwidth=1, row=1, col=4
    )
    fig.update_xaxes(
        showline=True, linewidth=1, linecolor="black",
        mirror=True, showgrid=True, gridcolor="#cccccc",
        gridwidth=1, row=2, col=4
    )
    fig.update_yaxes(
        showline=True, linewidth=1, linecolor="black",
        mirror=True, showgrid=True, gridcolor="#cccccc",
        gridwidth=1, row=2, col=4, range=[0, None]
    )

    print("\nОтображение объединённого графика…"); fig.show()


def visualize_without_spectra(triples: list[tuple[DataSet, DataSet]],
                      *, title: str | None = None) -> None:
    """
    Отображает все LF/HF-сигналы и их аппроксимации вертикально сдвинутыми.
    + Сводный график f_LF и f_HF (как в visualize) в третьей колонке первой строки.
    """
    if not triples:
        return

    RAW_CLR   = "#1fbe63"
    FIT_LF    = "red"
    FIT_HF    = "blue"
    BASE_CLR  = "#606060"

    # ── что меняется: H или T?
    all_H = {ds_lf.field_mT for ds_lf, _ in triples}
    all_T = {ds_lf.temp_K   for ds_lf, _ in triples}
    if   len(all_H) == 1 and len(all_T) > 1:
        varying, var_label = "T", "K"
        key_func = lambda ds_lf: ds_lf.temp_K
    elif len(all_T) == 1 and len(all_H) > 1:
        varying, var_label = "H", "mT"
        key_func = lambda ds_lf: ds_lf.field_mT
    else:
        raise RuntimeError("Скрипт ожидает изменение только H или T.")

    triples_sorted = sorted(triples, key=lambda p: key_func(p[0]))

    # ── вертикальный шаг
    ranges = []
    for pair in triples_sorted:
        for ds in pair:
            if ds.fit is not None:
                mask = ds.ts.t > 0
                p = ds.fit
                core = _core_signal(
                    ds.ts.t, p.A1, p.A2,
                    1/p.zeta1, 1/p.zeta2,
                    p.f1, p.f2, p.phi1, p.phi2
                )
                y_fit = p.k_lf * core + p.C_lf if ds.tag == 'LF' else p.k_hf * core + p.C_hf
                ranges.append(np.ptp(y_fit[mask]))
            else:
                ranges.append(np.ptp(ds.ts.s))
    y_step = 0.8 * max(ranges + [1e-3])

    # ── собираем данные для сводных графиков
    freq_vs_H: dict[int, list[tuple[int,float,float]]] = {}
    freq_vs_T: dict[int, list[tuple[int,float,float]]] = {}
    err_vs_H: dict[int, list[tuple[int,float,float]]] = {}
    err_vs_T: dict[int, list[tuple[int,float,float]]] = {}
    amp_vs_H: dict[int, list[tuple[int,float,float]]] = {}
    amp_vs_T: dict[int, list[tuple[int,float,float]]] = {}
    for ds_lf, ds_hf in triples_sorted:
        if ds_lf.fit is None:
            continue
        H, T = ds_lf.field_mT, ds_lf.temp_K
        f1, f2 = sorted((ds_lf.fit.f1/GHZ, ds_lf.fit.f2/GHZ))
        σ1, σ2 = ds_lf.fit.f1_err / GHZ, ds_lf.fit.f2_err / GHZ
        freq_vs_H.setdefault(T, []).append((H, f1, f2))
        freq_vs_T.setdefault(H, []).append((T, f1, f2))
        err_vs_H.setdefault(T, []).append((H, σ1, σ2))
        err_vs_T.setdefault(H, []).append((T, σ1, σ2))
        amp_vs_H.setdefault(T, []).append((H, ds_lf.fit.A1, ds_lf.fit.A2))
        amp_vs_T.setdefault(H, []).append((T, ds_lf.fit.A1, ds_lf.fit.A2))
    freq_vs_H = {T:sorted(v) for T,v in freq_vs_H.items() if len(v) >= 2}
    freq_vs_T = {H:sorted(v) for H,v in freq_vs_T.items() if len(v) >= 2}
    err_vs_H = {T:sorted(v) for T,v in err_vs_H.items() if len(v) >= 2}
    err_vs_T = {H:sorted(v) for H,v in err_vs_T.items() if len(v) >= 2}
    amp_vs_H = {T:sorted(v) for T,v in amp_vs_H.items() if len(v) >= 2}
    amp_vs_T = {H:sorted(v) for H,v in amp_vs_T.items() if len(v) >= 2}

    have_T, have_H = bool(freq_vs_T), bool(freq_vs_H)
    # n_rows_summary больше не нужен

    # ── макет подграфиков ──────────────────────────────────────────────
    specs = [
        [  # Row 1
            {"type": "xy", "rowspan": 2},           # 1 – сигналы LF
            {"type": "xy", "rowspan": 2},           # 2 – сигналы HF
            {"type": "xy"},                         # 3 – f-сводка
        ],
        [  # Row 2
            None,
            None,
            None,
        ],
    ]
    if varying == "T":
        fixed_H = list(all_H)[0]
        titles = [
            f"LF signals (H = {fixed_H} mT)",
            f"HF signals (H = {fixed_H} mT)",
            f"Frequencies (H = {fixed_H} mT)"
        ]
    elif varying == "H":
        fixed_T = list(all_T)[0]
        titles = [
            f"LF signals (T = {fixed_T} K)",
            f"HF signals (T = {fixed_T} K)",
            f"Frequencies (T = {fixed_T} K)"
        ]
    else:
        titles = ["LF signals", "HF signals", "Frequencies"]

    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    fig = make_subplots(
        rows=2, cols=3, specs=specs,
        column_widths=[0.34, 0.34, 0.32],
        horizontal_spacing=0.06,
        vertical_spacing=0.15,
        subplot_titles=tuple(titles)
    )

    # ── сигналы (первая строка, колонки 1 и 2)
    for idx, (ds_lf, ds_hf) in enumerate(triples_sorted):
        shift      = (idx + 1) * y_step
        var_value  = key_func(ds_lf)
        tmin_lf, tmax_lf = ds_lf.ts.t[0]/NS, ds_lf.ts.t[-1]/NS
        tmin_hf, tmax_hf = ds_hf.ts.t[0]/NS, ds_hf.ts.t[-1]/NS

        for col, (tmin, tmax) in ((1, (tmin_lf, tmax_lf)), (2, (tmin_hf, tmax_hf))):
            fig.add_trace(
                go.Scattergl(
                    x=[tmin, tmax], y=[shift, shift],
                    line=dict(width=1, color=BASE_CLR),
                    mode="lines",
                    showlegend=False, hoverinfo="skip"
                ), row=1, col=col)

        # LF raw + fit
        if ds_lf.fit: p = ds_lf.fit
        y = ds_lf.ts.s + shift
        y -= p.C_lf if ds_lf.fit else ds_lf.ts.s.mean()
        fig.add_trace(
            go.Scattergl(
                x=ds_lf.ts.t/NS, y=y,
                line=dict(width=3, color=RAW_CLR),
                name=f"{varying} = {var_value} {var_label}"
            ), 1, 1)

        if ds_lf.fit:
            core  = _core_signal(ds_lf.ts.t, p.A1, p.A2,
                                1/p.zeta1, 1/p.zeta2,
                                p.f1, p.f2, p.phi1, p.phi2)
            y_fit = p.k_lf * core + shift
            fig.add_trace(
                go.Scattergl(
                    x=ds_lf.ts.t/NS, y=y_fit,
                    line=dict(width=2, dash="dash", color=FIT_LF),
                    name=f"{varying} = {var_value} {var_label}"
                ), 1, 1)

        # HF raw + fit
        if ds_hf.fit: p = ds_hf.fit
        y = ds_hf.ts.s + shift
        y -= p.C_hf if ds_hf.fit else ds_hf.ts.s.mean()
        fig.add_trace(
            go.Scattergl(
                x=ds_hf.ts.t/NS, y=y,
                line=dict(width=3, color=RAW_CLR),
                name=f"{varying} = {var_value} {var_label}"
            ), 1, 2)

        if ds_hf.fit:
            core  = _core_signal(ds_hf.ts.t, p.A1, p.A2,
                                1/p.zeta1, 1/p.zeta2,
                                p.f1, p.f2, p.phi1, p.phi2)
            y_fit = p.k_hf * core + shift
            fig.add_trace(
                go.Scattergl(
                    x=ds_hf.ts.t/NS, y=y_fit,
                    line=dict(width=2, dash="dash", color=FIT_HF),
                    name=f"{varying} = {var_value} {var_label}"
                ), 1, 2)

        # подписи у хвоста
        fig.add_annotation(x=tmax_lf, y=shift,
                          text=f"{var_value} {var_label}",
                          showarrow=False, xanchor="left", font=dict(size=16),
                          row=1, col=1)
        fig.add_annotation(x=tmax_hf, y=shift,
                          text=f"{var_value} {var_label}",
                          showarrow=False, xanchor="left", font=dict(size=16),
                          row=1, col=2)

    # ── Сводный график в колонке 4 первой строки
    if varying == "T":
        for H_fix, pts in freq_vs_T.items():
            T_vals, fLF, fHF = zip(*pts)
            σLF = [σ1 for _,σ1,_ in err_vs_T[H_fix]]
            σHF = [σ2 for _,_,σ2 in err_vs_T[H_fix]]
            fig.add_trace(go.Scatter(x=T_vals, y=fLF, mode="markers",
                                     error_y=dict(type="data", array=σLF, visible=True),
                                     line=dict(width=2, color='red'),
                                     marker=dict(size=12),
                                     name=f"f_LF, H = {H_fix} mT"),
                          row=1, col=3)
            fig.add_trace(go.Scatter(x=T_vals, y=fHF, mode="markers",
                                     error_y=dict(type="data", array=σHF, visible=True),
                                     line=dict(width=2, color='blue'),
                                     marker=dict(size=12),
                                     name=f"f_HF, H = {H_fix} mT"),
                          row=1, col=3)
        fig.update_xaxes(title_text="T (K)", row=1, col=3)
    else:  # varying == "H"
        for T_fix, pts in freq_vs_H.items():
            H_vals, fLF, fHF = zip(*pts)
            σLF = [σ1 for _,σ1,_ in err_vs_H[T_fix]]
            σHF = [σ2 for _,_,σ2 in err_vs_H[T_fix]]
            fig.add_trace(go.Scatter(x=H_vals, y=fLF, mode="markers",
                                     error_y=dict(type="data", array=σLF, visible=True),
                                     line=dict(width=2, color='red'),
                                     marker=dict(size=12),
                                     name=f"f_LF, T = {T_fix} K"),
                          row=1, col=3)
            fig.add_trace(go.Scatter(x=H_vals, y=fHF, mode="markers",
                                     error_y=dict(type="data", array=σHF, visible=True),
                                     line=dict(width=2, color='blue'),
                                     marker=dict(size=12),
                                     name=f"f_HF, T = {T_fix} K"),
                          row=1, col=3)
        fig.update_xaxes(title_text="H (mT)", row=1, col=3)
    fig.update_yaxes(title_text="f (GHz)", row=1, col=3)

    # ── оформление
    fig.update_layout(
        showlegend=False, hovermode="x unified",
        font=dict(size=16),
        width=2000, height=1000,  # 4:1 разрешение
        paper_bgcolor="white",   # фон всего полотна
        plot_bgcolor="white"     # фон каждой ячейки
    )
    for annotation in fig['layout']['annotations'][:len(titles)]:
        annotation['font'] = dict(size=22)

    fig.update_xaxes(
        showline=True, linewidth=1, linecolor="black",
        mirror=True, showticklabels=True, ticks="inside",
        showgrid=True, gridcolor="#cccccc", gridwidth=1,
        row=1, col=1, title_text="time (ns)"
    )
    fig.update_xaxes(
        showline=True, linewidth=1, linecolor="black",
        mirror=True, showticklabels=True, ticks="inside",
        showgrid=True, gridcolor="#cccccc", gridwidth=1,
        row=1, col=2, title_text="time (ns)"
    )
    fig.update_yaxes(
        range=[0, shift + y_step],
        showline=True, linewidth=1, linecolor="black",
        mirror=True, showticklabels=False,
        row=1, col=1
    )
    fig.update_yaxes(
        range=[0, shift + y_step],
        showline=True, linewidth=1, linecolor="black",
        mirror=True, showticklabels=False,
        row=1, col=2
    )
    fig.update_xaxes(
        showline=True, linewidth=1, linecolor="black",
        mirror=True, showgrid=True, gridcolor="#cccccc",
        gridwidth=1, row=1, col=3
    )
    fig.update_yaxes(
        showline=True, linewidth=1, linecolor="black",
        mirror=True, showgrid=True, gridcolor="#cccccc",
        gridwidth=1, row=1, col=3
    )

    print("\nОтображение объединённого графика…"); fig.show()





# ────────────────────────── export

def export_freq_tables(triples: list[tuple[DataSet, DataSet]], root: Path) -> None:
    """
    Создаёт Excel‑файл frequencies_<folder>.xlsx (в самом *root*).
    На листах «LF» и «HF» pivot‑таблицы: строки = T, столбцы = H.
    """
    # ─── собираем данные без дублей (H,T) ──────────────────────────
    recs = []
    for lf, hf in triples:
        if lf.fit is None:
            continue
        H, T = lf.field_mT, lf.temp_K
        f1, f2 = lf.fit.f1/GHZ, lf.fit.f2/GHZ
        recs.append(dict(H=H, T=T, LF=min(f1, f2), HF=max(f1, f2)))

    if not recs:
        return

    df = (
        pd.DataFrame(recs)
          .drop_duplicates(subset=["H", "T"], keep="first")
    )

    # pivot‑таблицы
    tab_LF = df.pivot(index="T", columns="H", values="LF").sort_index()
    tab_HF = df.pivot(index="T", columns="H", values="HF").sort_index()

    # файл «frequencies_<folder>.xlsx» рядом с данными
    out_path = root / f"frequencies_({root.name}).xlsx"

    with pd.ExcelWriter(out_path, engine="openpyxl") as xls:
        tab_LF.to_excel(xls, sheet_name="LF", index=True, header=True)
        tab_HF.to_excel(xls, sheet_name="HF", index=True, header=True)

        # ─── стилизуем A1 на обоих листах ─────────────────────────
        for ws in xls.book.worksheets:          # «LF» и «HF»
            cell = ws["A1"]
            cell.value = "H, mT\nT, K"          # две строки
            cell.alignment = Alignment(
                wrapText=True, horizontal="center", vertical="center"
            )
            thin = Side(style="thin")
            cell.border = Border(diagonal=thin, diagonalDown=True)

            # чуть увеличим размеры для красоты
            ws.column_dimensions["A"].width = 12
            ws.row_dimensions[1].height = 30

    # ───── вариант B: два CSV файла ───
    # tab_LF.to_csv("freq_LF.csv")
    # tab_HF.to_csv("freq_HF.csv")

# ────────────────────────── main / demo

def main(data_dir: str = '.', *, return_datasets: bool = False, do_plot: bool = True):
    root = Path(data_dir).resolve()
    datasets = load_records(root)

    # группировка по (H, T)
    grouped: Dict[Tuple[int,int], Dict[str, object]] = {}
    for ds in datasets:
        key = (ds.field_mT, ds.temp_K)
        grouped.setdefault(key, {})[ds.tag] = ds

    triples = []
    for key, pair in grouped.items():
        if 'LF' in pair and 'HF' in pair:
            ds_lf, ds_hf = pair['LF'], pair['HF']
            try:
                process_pair(ds_lf, ds_hf)  # теперь ничего не возвращает
                triples.append((ds_lf, ds_hf))  # только пара DataSet
            except Exception as e:
                print(f"Error processing {key}: {e}")

    if do_plot and triples:
        visualize_without_spectra(triples)

    # экспорт таблиц
    export_freq_tables(triples, root)
    return triples if return_datasets else None

def demo(data_dir: str | Path = "."):
    """Colab‑friendly демонстрация: рисует только один интерактивный график."""
    triples = main(data_dir, return_datasets=True, do_plot=False)
    if not triples:
        raise RuntimeError("No valid LF/HF pairs found")
    visualize_without_spectra(triples)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', nargs='?', default='.')
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args()
    main(args.data_dir, do_plot=not args.no_plot)
