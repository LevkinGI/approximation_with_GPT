from __future__ import annotations

import re
from pathlib import Path
from typing import List
import numpy as np

from . import DataSet, TimeSeries, RecordMeta, logger, GHZ, C_M_S


def _soft_lowpass(signal: np.ndarray, fs: float, cutoff: float = 100e9,
                  transition: float = 5e9) -> np.ndarray:
    """Apply a near-rectangular low-pass filter with a cosine taper.

    Frequencies above ``cutoff`` are suppressed with a cosine roll-off of
    width ``transition`` to avoid ringing.
    """

    if signal.size == 0:
        return signal

    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(signal.size, d=1.0 / fs)

    if transition <= 0:
        transition = cutoff * 0.05

    trans_lo = max(0.0, cutoff - transition)
    trans_hi = cutoff + transition

    window = np.ones_like(freqs)
    window[freqs >= trans_hi] = 0.0
    in_trans = (freqs > trans_lo) & (freqs < trans_hi)
    window[in_trans] = 0.5 * (1 + np.cos(np.pi * (freqs[in_trans] - trans_lo) / transition))

    return np.fft.irfft(spectrum * window, n=signal.size)


def load_records(root: Path) -> List[DataSet]:
    """Читает все *.dat файлы в каталоге *root* (или *root/data*)
    и возвращает список DataSet."""
    pattern = re.compile(r"_(\d+)mT_(\d+)K_(HF|LF)_.*\.dat$", re.IGNORECASE)
    datasets: List[DataSet] = []
    data_dir = root / "data" if (root / "data").is_dir() else root
    logger.info("Поиск данных в %s", data_dir)
    for path in sorted(data_dir.glob("*.dat")):
        m = pattern.search(path.name)
        if not m:
            logger.warning("Пропуск %s: имя не соответствует шаблону", path.name)
            continue
        field_mT, temp_K, tag = int(m.group(1)), int(m.group(2)), m.group(3).upper()
        try:
            data = np.loadtxt(path, usecols=(0, 1, 2))
        except Exception as exc:
            logger.warning("Невозможно прочитать %s: %s", path.name, exc)
            continue
        if data.ndim != 2 or data.shape[1] != 3 or data.size < 15:
            logger.warning("Пропуск %s: неверный формат/мало точек", path.name)
            continue
        x, s, noise = data[:, 0], data[:, 1], data[:, 2]
        x0 = x[np.argmax(s)]
        t_all = 2.0 * (x - x0) / C_M_S  # секунды

        # Обрезаем сигнал сразу после первого минимума справа от пика
        pk = int(np.argmax(s))
        minima = np.where(
            (np.diff(np.signbit(np.diff(s))) > 0)
            & (np.arange(len(s))[1:-1] > pk)
        )[0]
        st = minima[0] + 1 if minima.size else pk + 1
        t = t_all[st:]
        s = s[st:]
        noise = noise[st:]

        # Для LF дополнительно ограничиваем длительность 0.7 нс
        if tag == "LF":
            cutoff = 0.7e-9
            end = np.searchsorted(t, t[0] + cutoff, "right")
            t, s, noise = t[:end], s[:end], noise[:end]

        if len(t) < 10:
            logger.warning("Пропуск %s: слишком короткий ряд", path.name)
            continue
        dt = float(np.mean(np.diff(t)))
        if not np.isfinite(dt) or dt <= 0:
            logger.warning("Пропуск %s: некорректный шаг dt", path.name)
            continue
        fs = 1.0 / dt

        # мягкий срез частот выше 100 ГГц в обоих сигналах
        s = _soft_lowpass(s, fs)
        noise = _soft_lowpass(noise, fs)

        ts = TimeSeries(t=t, s=s, meta=RecordMeta(fs=fs), noise=noise)
        datasets.append(DataSet(field_mT=field_mT, temp_K=temp_K, tag=tag,
                               ts=ts, root=data_dir))
        logger.info("Загружен %s: %d точек, fs=%.2f ГГц", path.name, len(t), fs / GHZ)
    logger.info("Загружено %d наборов", len(datasets))
    return datasets
