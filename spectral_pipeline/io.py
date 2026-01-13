from __future__ import annotations

import re
from pathlib import Path
from typing import List
import numpy as np

from . import DataSet, TimeSeries, RecordMeta, logger, GHZ, C_M_S


def _replace_spike_segment(x: np.ndarray, s: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """Replace values in ``s`` where ``x`` is between ``lower`` and ``upper``.

    The replacement value is taken from the first point after ``upper`` if it
    exists, otherwise from the last point before ``lower``. Falls back to the
    original values if neither is available.
    """
    mask = (lower < x) & (x < upper)
    if not np.any(mask):
        return s
    after = s[x >= upper]
    before = s[x <= lower]
    if after.size:
        fill = after[0]
    elif before.size:
        fill = before[-1]
    else:
        fill = s[mask][0]
    s_copy = s.copy()
    s_copy[mask] = fill
    return s_copy


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
            data = np.loadtxt(path, usecols=(0, 1))
        except Exception as exc:
            logger.warning("Невозможно прочитать %s: %s", path.name, exc)
            continue
        if data.ndim != 2 or data.shape[1] != 2 or data.size < 10:
            logger.warning("Пропуск %s: неверный формат/мало точек", path.name)
            continue
        x, s = data[:, 0], data[:, 1]
        x0 = x[np.argmax(s)]
        t_all = 2.0 * (x - x0) / C_M_S  # секунды

        # Вырезаем выбросы
        s = _replace_spike_segment(x, s, 136.9, 137.05)
        s = _replace_spike_segment(x, s, 142.9, 143.05)
        s = _replace_spike_segment(x, s, 132.85, 132.95)
        s = _replace_spike_segment(x, s, 133.15, 133.25)

        pk = int(np.argmax(s))
        minima_all = np.where(np.diff(np.signbit(np.diff(s))) < 0)[0] + 1
        left_minima = minima_all[minima_all < pk]
        right_minima = minima_all[minima_all > pk]
        left_min_idx = int(left_minima[-1]) if left_minima.size else None
        right_min_idx = int(right_minima[0]) if right_minima.size else None

        baseline_end = left_min_idx if left_min_idx is not None else pk
        if baseline_end <= 0:
            baseline_end = max(pk, 1)
        additive_const = float(np.median(s[:baseline_end])) if s.size else float("nan")
        if not np.isfinite(additive_const):
            additive_const = None

        # Обрезаем сигнал сразу после первого минимума справа от пика
        st_candidates = []
        if right_min_idx is not None:
            st_candidates.append(right_min_idx + 1)
        else:
            st_candidates.append(pk + 10)
        st_candidates.append(pk + 2 if tag == "LF" else pk + 5)
        st = int(np.min(st_candidates))
        t = t_all[st:]
        s = s[st:]

        # Для LF дополнительно ограничиваем длительность 0.7 нс
        if tag == "LF":
            cutoff = 0.5e-9
            end = np.searchsorted(t, cutoff, "right")
            t, s = t[:end], s[:end]

        # Для HF дополнительно ограничиваем длительность 0.08 нс
        if tag == "HF":
            cutoff = 0.09e-9
            end = np.searchsorted(t, cutoff, "right")
            t, s = t[:end], s[:end]

        if len(t) < 10:
            logger.warning("Пропуск %s: слишком короткий ряд", path.name)
            continue
        dt = float(np.mean(np.diff(t)))
        if not np.isfinite(dt) or dt <= 0:
            logger.warning("Пропуск %s: некорректный шаг dt", path.name)
            continue
        fs = 1.0 / dt
        ts = TimeSeries(t=t, s=s, meta=RecordMeta(fs=fs))
        datasets.append(DataSet(field_mT=field_mT, temp_K=temp_K, tag=tag,
                               ts=ts, root=data_dir, additive_const_init=additive_const))
        logger.info("Загружен %s: %d точек, fs=%.2f ГГц", path.name, len(t), fs / GHZ)
    logger.info("Загружено %d наборов", len(datasets))
    return datasets
