from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Dict
import logging
from functools import lru_cache

import numpy as np
import pandas as pd
from openpyxl.styles import Border, Side, Alignment

from . import DataSet, GHZ, logger, LOG_PATH
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


def export_freq_tables(triples: List[Tuple[DataSet, DataSet]], root: Path,
                       outfile: Path | None = None) -> None:
    logger.info("Экспорт таблиц частот")
    recs = []
    for lf, hf in triples:
        if lf.fit is None:
            continue
        H, T = lf.field_mT, lf.temp_K
        f1, f2 = lf.fit.f1/GHZ, lf.fit.f2/GHZ
        recs.append(dict(H=H, T=T, LF=f1, HF=f2))
    if not recs:
        logger.warning("Нет данных для экспорта таблиц")
        return
    df = (
        pd.DataFrame(recs)
          .drop_duplicates(subset=["H", "T"], keep="first")
    )
    tab_LF = df.pivot(index="T", columns="H", values="LF").sort_index()
    tab_HF = df.pivot(index="T", columns="H", values="HF").sort_index()
    out_path = outfile if outfile else root / f"frequencies_({root.name}).xlsx"
    try:
        with pd.ExcelWriter(out_path, engine="openpyxl") as xls:
            tab_LF.to_excel(xls, sheet_name="LF", index=True, header=True)
            tab_HF.to_excel(xls, sheet_name="HF", index=True, header=True)
            for ws in xls.book.worksheets:
                cell = ws["A1"]
                cell.value = "H, mT\nT, K"
                cell.alignment = Alignment(wrapText=True, horizontal="center", vertical="center")
                thin = Side(style="thin")
                cell.border = Border(diagonal=thin, diagonalDown=True)
                ws.column_dimensions["A"].width = 12
                ws.row_dimensions[1].height = 30
        logger.info("Таблицы сохранены в %s", out_path)
    except Exception as exc:
        logger.error("Не удалось сохранить %s: %s", out_path, exc)
        return


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
