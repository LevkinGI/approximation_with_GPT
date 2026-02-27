from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ApproximationConfig:
    """Настройки аппроксимации для двухмодовой модели."""

    # Ограничения параметров мод
    equal_amplitudes: bool = True
    equal_phases: bool = True
    # Применяется только если equal_phases=True.
    # При True и использовании синусной формы фазы фиксируются в 0.
    zero_phases_if_equal: bool = True

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
