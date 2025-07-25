from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional
import numpy as np
from numpy.typing import NDArray
import logging
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("spectral_pipeline")
logger.setLevel(logging.INFO)

GHZ = 1e9
NS = 1e-9
PI = math.pi
# скорость света, мм/с (используется при расчёте времени)
C_M_S = 3e11

# диапазоны частот НЧ и ВЧ
LF_BAND = (8 * GHZ, 12 * GHZ)
HF_BAND = (20 * GHZ, 80 * GHZ)
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
    k_lf: float
    k_hf: float
    C_lf: float
    C_hf: float
    f1_err: float | None = None
    f2_err: float | None = None

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
    "GHZ", "NS", "PI", "FREQ_TAG", "logger",
    "C_M_S", "LF_BAND", "HF_BAND",
]
