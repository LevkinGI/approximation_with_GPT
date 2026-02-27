from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Protocol, Tuple
import logging

import numpy as np

from . import DataSet, logger, LOG_PATH
from .approximation_config import ApproximationConfig, DEFAULT_APPROXIMATION_CONFIG


class PairProcessor(Protocol):
    def __call__(
        self,
        ds_lf: DataSet,
        ds_hf: DataSet,
        *,
        approximation_config: ApproximationConfig,
    ) -> object | None: ...


class LfProcessor(Protocol):
    def __call__(
        self,
        ds_lf: DataSet,
        *,
        approximation_config: ApproximationConfig,
    ) -> object | None: ...


class Plotter(Protocol):
    def __call__(
        self,
        triples: List[Tuple[DataSet, DataSet]],
        *,
        use_theory_guess: bool,
        approximation_config: ApproximationConfig,
    ) -> object: ...


class Exporter(Protocol):
    def __call__(self, triples: List[Tuple[DataSet, DataSet]], root: Path, outfile: Path | None = None) -> object: ...


LoaderFn = Callable[[Path, ApproximationConfig], List[DataSet]]
CrossingFinder = Callable[[Path, int, int], tuple[str, float] | None]


@dataclass(slots=True)
class PipelineHooks:
    loader: LoaderFn
    pair_processor: PairProcessor
    lf_only_processor: LfProcessor
    plotter: Plotter
    exporter: Exporter
    crossing_finder: CrossingFinder


def _group_by_conditions(datasets: List[DataSet]) -> Dict[Tuple[int, int], Dict[str, DataSet]]:
    grouped: Dict[Tuple[int, int], Dict[str, DataSet]] = {}
    for ds in datasets:
        key = (ds.field_mT, ds.temp_K)
        grouped.setdefault(key, {})[ds.tag] = ds
    return grouped


def _should_use_lf_only(crossing: tuple[str, float] | None, ds_lf: DataSet) -> bool:
    if crossing is None:
        return False
    axis, val = crossing
    if axis == "T" and ds_lf.temp_K >= val:
        return True
    if axis == "H" and ds_lf.field_mT >= val:
        return True
    return False


def _process_pair(
    key: Tuple[int, int],
    ds_lf: DataSet,
    ds_hf: DataSet,
    *,
    hooks: PipelineHooks,
    approximation_config: ApproximationConfig,
) -> object | None:
    try:
        try:
            return hooks.pair_processor(ds_lf, ds_hf, approximation_config=approximation_config)
        except TypeError:
            return hooks.pair_processor(ds_lf, ds_hf, use_theory_guess=approximation_config.use_theory_guess)
    except Exception as exc:  # pragma: no cover - passthrough to logger
        logger.error("Ошибка обработки %s: %s", key, exc)
        return None


def _process_lf_only(
    key: Tuple[int, int],
    ds_lf: DataSet,
    *,
    hooks: PipelineHooks,
    approximation_config: ApproximationConfig,
) -> object | None:
    try:
        try:
            return hooks.lf_only_processor(ds_lf, approximation_config=approximation_config)
        except TypeError:
            return hooks.lf_only_processor(ds_lf, use_theory_guess=approximation_config.use_theory_guess)
    except Exception as exc:  # pragma: no cover - passthrough to logger
        logger.error("Ошибка обработки %s: %s", key, exc)
        return None


@lru_cache(maxsize=None)
def find_crossing(root: Path, field_mT: int, temp_K: int):
    """Return axis type and value where HF and LF curves intersect.

    Searches first-approximation files ``H_{field}.npy`` or ``T_{temp}.npy``
    located in ``root``.  If intersection is not found, returns ``None``.
    """

    path_root = Path(root)
    path_H = path_root / f"H_{field_mT}.npy"
    path_T = path_root / f"T_{temp_K}.npy"
    arr = None
    axis_name = None
    if path_H.exists():
        arr = np.load(path_H)
        axis_name = "T"
    elif path_T.exists():
        arr = np.load(path_T)
        axis_name = "H"
    if arr is None or arr.shape[0] < 3:
        return None
    axis, hf, lf = arr[0], arr[1], arr[2]
    diff = hf - lf
    idx = np.where(diff <= 0)[0]
    if idx.size:
        return axis_name, float(axis[idx[0]])
    return None


def _fit_pairs(
    grouped: Dict[Tuple[int, int], Dict[str, DataSet]],
    root: Path,
    *,
    hooks: PipelineHooks,
    approximation_config: ApproximationConfig,
) -> Tuple[List[Tuple[DataSet, DataSet]], int]:
    triples: List[Tuple[DataSet, DataSet]] = []
    success_count = 0

    for key, pair in grouped.items():
        if "LF" not in pair:
            continue
        ds_lf = pair["LF"]
        ds_hf = pair.get("HF")

        use_lf_only = approximation_config.force_lf_only

        if use_lf_only or ds_hf is None:
            fit = _process_lf_only(key, ds_lf, hooks=hooks, approximation_config=approximation_config)
            ds_hf = ds_hf or ds_lf
        else:
            fit = _process_pair(key, ds_lf, ds_hf, hooks=hooks, approximation_config=approximation_config)

        if fit is not None:
            success_count += 1
            triples.append((ds_lf, ds_hf))

    return triples, success_count


def run_pipeline(
    data_dir: str = ".",
    *,
    return_datasets: bool = False,
    do_plot: bool = True,
    excel_path: str | None = None,
    log_level: str | None = None,
    use_theory_guess: bool | None = None,
    approximation_config: ApproximationConfig | None = None,
    hooks: PipelineHooks,
):
    cfg = approximation_config or DEFAULT_APPROXIMATION_CONFIG
    resolved_log_level = log_level if log_level is not None else cfg.log_level
    level = getattr(logging, resolved_log_level.upper(), logging.INFO)
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)
    logger.info("Лог-файл: %s", LOG_PATH)

    resolved_use_theory_guess = cfg.use_theory_guess if use_theory_guess is None else use_theory_guess
    active_cfg = replace(cfg, use_theory_guess=resolved_use_theory_guess)

    root = Path(data_dir).resolve()
    logger.info("Начало обработки каталога %s", root)

    try:
        datasets = hooks.loader(root, active_cfg)
    except TypeError:
        datasets = hooks.loader(root)
    if not datasets:
        logger.error("В каталоге %s отсутствуют файлы .dat", root)
        return None
    logger.info("Загружено %d файлов", len(datasets))

    grouped = _group_by_conditions(datasets)
    triples, success_count = _fit_pairs(
        grouped, root, hooks=hooks, approximation_config=active_cfg
    )

    logger.info("Успешно аппроксимировано пар: %d", success_count)

    if do_plot and success_count:
        try:
            hooks.plotter(
                triples,
                use_theory_guess=resolved_use_theory_guess,
                approximation_config=active_cfg,
            )
        except TypeError:
            hooks.plotter(triples, use_theory_guess=resolved_use_theory_guess)

    out_excel = Path(excel_path) if excel_path else None
    if success_count:
        hooks.exporter(triples, root, outfile=out_excel)

    logger.info("Завершение обработки каталога %s", root)
    return triples if return_datasets else None
