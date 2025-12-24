from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Dict
import logging
from functools import lru_cache

import numpy as np
import pandas as pd
from openpyxl.styles import Border, Side, Alignment

from . import DataSet, GHZ, NS, logger, LOG_PATH
from .io import load_records
from .fit import process_pair, process_lf_only
from .plotting import visualize_stacked


@lru_cache(maxsize=None)
def _find_crossing(root: str, field_mT: int, temp_K: int):
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


_PARAM_ROWS = (
    (
        "Частота НЧ, ГГц",
        lambda fit: fit.f1 / GHZ,
        lambda fit: fit.f1_err / GHZ if fit.f1_err is not None else np.nan,
    ),
    (
        "Частота ВЧ, ГГц",
        lambda fit: fit.f2 / GHZ,
        lambda fit: fit.f2_err / GHZ if fit.f2_err is not None else np.nan,
    ),
    (
        "Время затухания НЧ, нс",
        lambda fit: (1.0 / fit.zeta1) / NS if fit.zeta1 else np.nan,
        lambda fit: fit.tau1_err / NS if fit.tau1_err is not None else np.nan,
    ),
    (
        "Время затухания ВЧ, нс",
        lambda fit: (1.0 / fit.zeta2) / NS if fit.zeta2 else np.nan,
        lambda fit: fit.tau2_err / NS if fit.tau2_err is not None else np.nan,
    ),
    (
        "Начальная фаза НЧ, рад",
        lambda fit: fit.phi1,
        lambda fit: fit.phi1_err if fit.phi1_err is not None else np.nan,
    ),
    (
        "Начальная фаза ВЧ, рад",
        lambda fit: fit.phi2,
        lambda fit: fit.phi2_err if fit.phi2_err is not None else np.nan,
    ),
    ("Амплитуда НЧ", lambda fit: fit.A1, lambda fit: fit.A1_err if fit.A1_err is not None else np.nan),
    ("Амплитуда ВЧ", lambda fit: fit.A2, lambda fit: fit.A2_err if fit.A2_err is not None else np.nan),
    ("k НЧ", lambda fit: fit.k_lf, lambda fit: fit.k_lf_err if fit.k_lf_err is not None else np.nan),
    ("k ВЧ", lambda fit: fit.k_hf, lambda fit: fit.k_hf_err if fit.k_hf_err is not None else np.nan),
    ("Константа НЧ", lambda fit: fit.C_lf, lambda fit: fit.C_lf_err if fit.C_lf_err is not None else np.nan),
    ("Константа ВЧ", lambda fit: fit.C_hf, lambda fit: fit.C_hf_err if fit.C_hf_err is not None else np.nan),
)


def _prepare_axis(recs: List[dict]):
    fields = sorted({rec["H"] for rec in recs})
    temps = sorted({rec["T"] for rec in recs})
    axis_type = "H"
    axis_label = "H"
    axis_unit = "mT"
    if len(temps) > len(fields):
        axis_type = "T"
        axis_label = "T"
        axis_unit = "K"
    elif len(temps) == len(fields) and len(temps) > 1:
        axis_type = "H"
    grouped: Dict[int, List[dict]] = {}
    key_name = "H" if axis_type == "H" else "T"
    other_name = "T" if axis_type == "H" else "H"
    for rec in recs:
        key = rec[key_name]
        grouped.setdefault(key, []).append(rec)
    selected: Dict[int, dict] = {}
    for key, rows in grouped.items():
        selected[key] = rows[0]
        if len(rows) > 1:
            others = sorted({row[other_name] for row in rows})
            logger.warning(
                "Для %s=%s найдено %d записей с разными %s: %s. Используется первая.",
                key_name, key, len(rows), other_name, others,
            )
    axis_values = sorted(selected.keys())
    return axis_label, axis_unit, axis_values, selected


def export_freq_tables(triples: List[Tuple[DataSet, DataSet]], root: Path,
                       outfile: Path | None = None) -> None:
    logger.info("Экспорт таблицы параметров аппроксимации")
    recs = []
    for lf, _ in triples:
        if lf.fit is None:
            continue
        recs.append({
            "H": lf.field_mT,
            "T": lf.temp_K,
            "fit": lf.fit,
        })
    if not recs:
        logger.warning("Нет данных для экспорта таблицы")
        return
    axis_label, axis_unit, axis_vals, selected = _prepare_axis(recs)
    if not axis_vals:
        logger.warning("Нет данных после группировки для экспорта таблицы")
        return
    rows = []
    for name, val_fn, err_fn in _PARAM_ROWS:
        rows.append((name, val_fn))
        rows.append((f"Погр. {name}", err_fn))
    data = {}
    for val in axis_vals:
        fit = selected[val]["fit"] if val in selected else None
        col = []
        for _, fn in rows:
            if fit is None:
                col.append(np.nan)
            else:
                try:
                    col.append(fn(fit))
                except Exception:
                    col.append(np.nan)
        data[val] = col
    df = pd.DataFrame(data, index=[name for name, _ in rows])
    df.index.name = f"{axis_label}, {axis_unit}"
    sheet_name = "parameters"
    out_path = outfile if outfile else root / f"approximation_({root.name}).xlsx"
    try:
        with pd.ExcelWriter(out_path, engine="openpyxl") as xls:
            df.to_excel(xls, sheet_name=sheet_name, index=True, header=True)
            ws = xls.book[sheet_name]
            cell = ws["A1"]
            cell.value = f"{axis_label}, {axis_unit}"
            cell.alignment = Alignment(wrapText=True, horizontal="center", vertical="center")
            thin = Side(style="thin")
            cell.border = Border(diagonal=thin, diagonalDown=True)
            ws.column_dimensions["A"].width = 25
            ws.row_dimensions[1].height = 30
        logger.info("Таблица сохранена в %s", out_path)
    except Exception as exc:
        logger.error("Не удалось сохранить %s: %s", out_path, exc)


def main(
    data_dir: str = '.',
    *,
    return_datasets: bool = False,
    do_plot: bool = True,
    excel_path: str | None = None,
    log_level: str = "DEBUG",
    use_theory_guess: bool = True,
):
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)
    for h in logger.handlers:
        h.setLevel(level)
    logger.info("Лог-файл: %s", LOG_PATH)
    root = Path(data_dir).resolve()
    logger.info("Начало обработки каталога %s", root)
    datasets = load_records(root)
    if not datasets:
        logger.error("В каталоге %s отсутствуют файлы .dat", root)
        return None
    logger.info("Загружено %d файлов", len(datasets))
    grouped: Dict[Tuple[int, int], Dict[str, object]] = {}
    for ds in datasets:
        key = (ds.field_mT, ds.temp_K)
        grouped.setdefault(key, {})[ds.tag] = ds
    triples: List[Tuple[DataSet, DataSet]] = []
    success_count = 0
    for key, pair in grouped.items():
        if 'LF' not in pair:
            continue
        ds_lf = pair['LF']
        ds_hf = pair.get('HF')
        cross = _find_crossing(str(root), ds_lf.field_mT, ds_lf.temp_K)
        use_lf_only = False
        if cross is not None:
            axis, val = cross
            if axis == 'T' and ds_lf.temp_K >= val:
                use_lf_only = True
            if axis == 'H' and ds_lf.field_mT >= val:
                use_lf_only = True
        if use_lf_only or ds_hf is None:
            try:
                fit = process_lf_only(ds_lf, use_theory_guess=use_theory_guess)
            except Exception as e:
                logger.error("Ошибка обработки %s: %s", key, e)
            else:
                if fit is not None:
                    success_count += 1
                    ds_hf = ds_hf or ds_lf
                    triples.append((ds_lf, ds_hf))
        else:
            try:
                fit = process_pair(ds_lf, ds_hf, use_theory_guess=use_theory_guess)
            except Exception as e:
                logger.error("Ошибка обработки %s: %s", key, e)
            else:
                if fit is not None:
                    success_count += 1
                    triples.append((ds_lf, ds_hf))
    logger.info("Успешно аппроксимировано пар: %d", success_count)
    if do_plot and success_count:
        visualize_stacked(triples, use_theory_guess=use_theory_guess)
    out_excel = Path(excel_path) if excel_path else None
    if success_count:
        export_freq_tables(triples, root, outfile=out_excel)
    logger.info("Завершение обработки каталога %s", root)
    return triples if return_datasets else None


def demo(data_dir: str | Path = ".", *, use_theory_guess: bool = True):
    triples = main(
        data_dir,
        return_datasets=True,
        do_plot=False,
        use_theory_guess=use_theory_guess,
    )
    if not triples:
        raise RuntimeError("Не найдено корректных пар LF/HF")
    visualize_stacked(triples, use_theory_guess=use_theory_guess)
    print("График открыт в браузере")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', nargs='?', default='.')
    parser.add_argument('--no-plot', action='store_true')
    parser.add_argument('--excel', help='путь к выходному xlsx')
    parser.add_argument('--log-level', default='DEBUG', help='уровень логирования')
    parser.add_argument(
        '--use-theory-guess',
        dest='use_theory_guess',
        action='store_true',
        help='использовать теоретические значения в качестве первого приближения',
    )
    parser.add_argument(
        '--no-use-theory-guess',
        dest='use_theory_guess',
        action='store_false',
        help='не использовать теоретические значения при подборе',
    )
    parser.set_defaults(use_theory_guess=True)
    args = parser.parse_args()
    main(args.data_dir, do_plot=not args.no_plot, excel_path=args.excel,
         log_level=args.log_level, use_theory_guess=args.use_theory_guess)
