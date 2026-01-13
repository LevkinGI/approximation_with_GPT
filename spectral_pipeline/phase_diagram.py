from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.io import loadmat

from . import logger


PHASE_WEIGHT = 10.0


@dataclass(slots=True)
class PhaseDiagram:
    temps: NDArray
    fields: NDArray
    theta: NDArray


def _load_mat_array(path: Path) -> NDArray | None:
    try:
        data = loadmat(path)
    except Exception as exc:
        logger.warning("Не удалось прочитать %s: %s", path, exc)
        return None
    for key, val in data.items():
        if key.startswith("__"):
            continue
        if isinstance(val, np.ndarray):
            return np.array(val, dtype=float)
    logger.warning("В %s не найдено массивов данных", path)
    return None


def load_experimental_phase_diagram(root: Path) -> PhaseDiagram | None:
    data_dir = root / "data" if (root / "data").is_dir() else root
    temp_path = data_dir / "Temp_exper.mat"
    field_path = data_dir / "H_exper.mat"
    theta_path = data_dir / "teta_exper.mat"
    missing = [p for p in (temp_path, field_path, theta_path) if not p.exists()]
    if missing:
        logger.info("Экспериментальная фазовая диаграмма не найдена: %s", missing)
        return None
    temp = _load_mat_array(temp_path)
    field = _load_mat_array(field_path)
    theta = _load_mat_array(theta_path)
    if temp is None or field is None or theta is None:
        return None
    if temp.shape != field.shape or temp.shape != theta.shape:
        logger.warning(
            "Несовпадающие формы фазовой диаграммы: T=%s, H=%s, θ=%s",
            temp.shape,
            field.shape,
            theta.shape,
        )
        return None
    return PhaseDiagram(temps=temp, fields=field, theta=theta)


def _feature_matrix(temps: NDArray, fields: NDArray) -> NDArray:
    t = temps.ravel()
    h = fields.ravel()
    return np.column_stack(
        (
            np.ones_like(t),
            t,
            h,
            t * h,
            t**2,
            h**2,
        )
    )


def compute_phases(temps: NDArray, fields: NDArray, coeffs: NDArray) -> NDArray:
    """Возвращает теоретическую фазовую диаграмму θ(T, H)."""
    features = _feature_matrix(temps, fields)
    coeffs = np.array(coeffs, dtype=float)
    theta = features @ coeffs
    return theta.reshape(temps.shape)


def fit_phase_coeffs(diagram: PhaseDiagram, *, weight: float = PHASE_WEIGHT) -> NDArray:
    temps = diagram.temps
    fields = diagram.fields
    theta = diagram.theta
    mask = np.isfinite(temps) & np.isfinite(fields) & np.isfinite(theta)
    if not np.any(mask):
        raise ValueError("Нет корректных данных для аппроксимации фазовой диаграммы.")
    temps = temps[mask]
    fields = fields[mask]
    theta = theta[mask]
    features = _feature_matrix(temps, fields)
    scale = float(weight)
    coeffs, *_ = np.linalg.lstsq(features * scale, theta * scale, rcond=None)
    return coeffs
