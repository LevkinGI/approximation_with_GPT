from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from . import GHZ

OutlierInterval = Tuple[float, float]
BandHz = Tuple[float, float]
Bounds = Tuple[float, float]


@dataclass(frozen=True, slots=True)
class ApproximationConfig:
    """Единые настройки аппроксимации и препроцессинга."""

    # Оркестрация пайплайна
    use_theory_guess: bool = True
    force_lf_only: bool = False

    # Полосы поиска частот
    lf_band_hz: BandHz = (0.0, 50 * GHZ)
    hf_band_hz: BandHz = (0.0, 50 * GHZ)

    # Ограничения параметров мод
    equal_amplitudes: bool = True
    equal_phases: bool = True
    # Применяется только если equal_phases=True.
    zero_phases_if_equal: bool = True

    # Настройки обрезки/очистки сигнала
    outlier_intervals: tuple[OutlierInterval, ...] = (
        (136.9, 137.05),
        (142.9, 143.05),
        (132.85, 132.95),
        (133.15, 133.25),
    )
    cutoff_lf_s: float = 0.3e-9
    cutoff_hf_s: float = 0.08e-9

    # Границы и начальные диапазоны для fit
    init_freq_lo_mul: float = 0.9
    init_freq_hi_mul: float = 1.2
    tau1_bounds_s: Bounds = (5e-11, 5e-9)
    tau2_bounds_s: Bounds = (5e-12, 5e-10)

    # Критерий успешной аппроксимации
    max_cost: float = 100.0

    # Настройки least_squares
    ftol: float = 3e-16
    xtol: float = 3e-16
    gtol: float = 3e-16
    max_nfev: int = 5000
    loss: str = "soft_l1"
    f_scale: float = 0.1
    x_scale: str = "jac"


DEFAULT_APPROXIMATION_CONFIG = ApproximationConfig()
