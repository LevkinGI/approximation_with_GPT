from __future__ import annotations

import re
from pathlib import Path
from typing import List
import numpy as np

from . import DataSet, TimeSeries, RecordMeta, logger, GHZ, C_M_S


def load_records(root: Path) -> List[DataSet]:
    """Читает все *.dat файлы в каталоге *root* (или *root/data*)
    и возвращает список DataSet."""
    pattern = re.compile(r"_(\d+)mT_(\d+)K_(HF|LF)_.*\.dat$", re.IGNORECASE)
    datasets: List[DataSet] = []
    data_dir = root / "data" if (root / "data").is_dir() else root
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
        t = 2.0 * (x - x0) / C_M_S  # секунды
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
        fs = 1.0 / dt
        ts = TimeSeries(t=t, s=s, meta=RecordMeta(fs=fs))
        datasets.append(DataSet(field_mT=field_mT, temp_K=temp_K, tag=tag, ts=ts))
        logger.info("Загружен %s: %d точек, fs=%.2f ГГц", path.name, len(t), fs / GHZ)
    return datasets
