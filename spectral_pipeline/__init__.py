from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional
import numpy as np
from numpy.typing import NDArray
import logging
from logging.handlers import RotatingFileHandler
import math

log_dir = Path(__file__).resolve().parent.parent / "logs"
log_dir.mkdir(exist_ok=True)
LOG_PATH = log_dir / "pipeline.log"

logger = logging.getLogger("spectral_pipeline")
if not logger.handlers:
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream_h = logging.StreamHandler()
    stream_h.setFormatter(fmt)
    file_h = RotatingFileHandler(LOG_PATH, maxBytes=1_000_000, backupCount=5, encoding="utf-8")
    file_h.setFormatter(fmt)
    logger.addHandler(stream_h)
    logger.addHandler(file_h)
    logger.setLevel(logging.INFO)
    logger.propagate = False

GHZ = 1e9
NS = 1e-9
PI = math.pi
# скорость света, мм/с (используется при расчёте времени)
C_M_S = 3e11

# диапазоны частот для НЧ и ВЧ сигналов
# поиск осуществляется в общем диапазоне 0–40 ГГц
LF_BAND = (0.0, 40 * GHZ)
HF_BAND = (0.0, 40 * GHZ)
FREQ_TAG = Literal["LF", "HF"]

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
    k_scale: float
    C_lf: float
    C_hf: float
    f1_err: float | None = None
    f2_err: float | None = None
    cost: float | None = None

@dataclass(slots=True)
class DataSet:
    """Набор данных одного измерения.

    field_mT: магнитное поле (мТл)
    temp_K : температура (К)
    tag    : диапазон сигнала ('LF' или 'HF')
    ts     : временной ряд

    f1_init, f2_init – грубые оценки частот
    zeta1, zeta2     – оценки затухания
    fit              – результат аппроксимации
    freq_fft, asd_fft – спектр сигнала
    """
    field_mT: int
    temp_K: int
    tag: FREQ_TAG
    ts: TimeSeries
    root: Path | None = None

    # начальные оценки из Coarse + ESPRIT
    f1_init: float = 0.0
    f2_init: float = 0.0
    zeta1: Optional[float] = None
    zeta2: Optional[float] = None

    # окончательный результат
    fit: FittingResult | None = field(default_factory=lambda: None)

    # FFT-спектр
    freq_fft: NDArray | None = None
    asd_fft : NDArray | None = None

__all__ = [
    "DataSet", "FittingResult", "RecordMeta", "TimeSeries",
    "GHZ", "NS", "PI", "FREQ_TAG", "logger", "LOG_PATH",
    "C_M_S", "LF_BAND", "HF_BAND",
]
