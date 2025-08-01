from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Dict
import logging
import pandas as pd
from openpyxl.styles import Border, Side, Alignment

from . import DataSet, GHZ, logger
from .io import load_records
from .fit import process_pair
from .plotting import visualize_without_spectra, visualize_stacked


def export_freq_tables(triples: List[Tuple[DataSet, DataSet]], root: Path,
                       outfile: Path | None = None) -> None:
    logger.info("Экспорт таблиц частот")
    recs = []
    for lf, hf in triples:
        if lf.fit is None:
            continue
        H, T = lf.field_mT, lf.temp_K
        f1, f2 = lf.fit.f1/GHZ, lf.fit.f2/GHZ
        recs.append(dict(H=H, T=T, LF=min(f1, f2), HF=max(f1, f2)))
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


def main(data_dir: str = '.', *, return_datasets: bool = False,
         do_plot: bool = True, excel_path: str | None = None,
         log_level: str = "INFO"):
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)
    for h in logger.handlers:
        h.setLevel(level)
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
    triples = []
    for key, pair in grouped.items():
        if 'LF' in pair and 'HF' in pair:
            ds_lf, ds_hf = pair['LF'], pair['HF']
            try:
                process_pair(ds_lf, ds_hf)
                triples.append((ds_lf, ds_hf))
            except Exception as e:
                logger.error("Ошибка обработки %s: %s", key, e)
    logger.info("Успешно обработано пар: %d", len(triples))
    if do_plot and triples:
        visualize_stacked(triples)
    out_excel = Path(excel_path) if excel_path else None
    export_freq_tables(triples, root, outfile=out_excel)
    logger.info("Завершение обработки каталога %s", root)
    return triples if return_datasets else None


def demo(data_dir: str | Path = "."):
    triples = main(data_dir, return_datasets=True, do_plot=False)
    if not triples:
        raise RuntimeError("Не найдено корректных пар LF/HF")
    visualize_stacked(triples)
    print("График открыт в браузере")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', nargs='?', default='.')
    parser.add_argument('--no-plot', action='store_true')
    parser.add_argument('--excel', help='путь к выходному xlsx')
    parser.add_argument('--log-level', default='INFO', help='уровень логирования')
    args = parser.parse_args()
    main(args.data_dir, do_plot=not args.no_plot, excel_path=args.excel,
         log_level=args.log_level)
