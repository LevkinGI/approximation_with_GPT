from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, sosfiltfilt

from . import GHZ, logger

BANDPASS_LOW_GHZ = 2.0
BANDPASS_HIGH_GHZ = 80.0
_BUTTER_ORDER = 5
_HIGH_CUTOFF_MARGIN = 0.99


def bandpass_filter_signal(
    signal: NDArray,
    fs: float,
    *,
    low_GHz: float = BANDPASS_LOW_GHZ,
    high_GHz: float = BANDPASS_HIGH_GHZ,
    order: int = _BUTTER_ORDER,
) -> NDArray:
    """Return a zero-phase band-pass filtered copy of ``signal``.

    Parameters
    ----------
    signal:
        Input time-domain signal.
    fs:
        Sampling frequency in Hz.
    low_GHz, high_GHz:
        Pass-band limits expressed in GHz. The defaults correspond to the
        2–80 ГГц range required by the processing pipeline.
    order:
        Butterworth filter order.

    Notes
    -----
    If the sampling frequency is insufficient to realise the requested
    pass-band, the original signal (converted to ``float64``) is returned
    unchanged.
    """

    arr = np.asarray(signal, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("signal must be one-dimensional")

    nyq = 0.5 * fs
    low_hz = low_GHz * GHZ
    high_hz = high_GHz * GHZ

    if nyq <= low_hz:
        logger.debug(
            "Пропуск полосовой фильтрации: Nyquist=%.1f ГГц <= нижней границы %.1f ГГц",
            nyq / GHZ,
            low_GHz,
        )
        return arr

    if high_hz >= nyq:
        adjusted = nyq * _HIGH_CUTOFF_MARGIN
        logger.debug(
            "Высокая граница %.1f ГГц обрезана до %.1f ГГц (Nyquist=%.1f ГГц)",
            high_GHz,
            adjusted / GHZ,
            nyq / GHZ,
        )
        high_hz = adjusted

    if high_hz <= low_hz:
        logger.debug(
            "Пропуск полосовой фильтрации: верхняя граница %.1f ГГц <= %.1f ГГц",
            high_hz / GHZ,
            low_hz / GHZ,
        )
        return arr

    try:
        sos = butter(order, [low_hz, high_hz], btype="band", fs=fs, output="sos")
    except ValueError as exc:
        logger.debug(
            "Пропуск полосовой фильтрации %.1f–%.1f ГГц: %s",
            low_GHz,
            high_hz / GHZ,
            exc,
        )
        return arr

    padlen = min(arr.size - 1, 3 * (sos.shape[0] - 1))
    if padlen < 0:
        padlen = 0
    filtered = sosfiltfilt(sos, arr, padlen=padlen)
    logger.debug(
        "Применена полосовая фильтрация %.1f–%.1f ГГц (fs=%.1f ГГц)",
        low_GHz,
        high_hz / GHZ,
        fs / GHZ,
    )
    return filtered


__all__ = ["bandpass_filter_signal", "BANDPASS_LOW_GHZ", "BANDPASS_HIGH_GHZ"]
