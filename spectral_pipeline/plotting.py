from __future__ import annotations

from typing import List, Tuple
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import DataSet, GHZ, NS, logger
from .fit import _core_signal


def _add_signal_trace(fig, ds: DataSet, shift: float, row: int, col: int,
                      *, raw_color: str, fit_color: str,
                      label: str, base_color: str = "#606060") -> None:
    tmin, tmax = ds.ts.t[0] / NS, ds.ts.t[-1] / NS
    fig.add_trace(
        go.Scattergl(x=[tmin, tmax], y=[shift, shift],
                     line=dict(width=1, color=base_color),
                     mode="lines", showlegend=False, hoverinfo="skip"),
        row=row, col=col)

    y = ds.ts.s + shift
    if ds.fit:
        p = ds.fit
        y -= p.C_lf if ds.tag == "LF" else p.C_hf
    else:
        y -= ds.ts.s.mean()

    fig.add_trace(
        go.Scattergl(x=ds.ts.t/NS, y=y,
                     line=dict(width=3, color=raw_color),
                     name=label),
        row=row, col=col)

    if ds.fit:
        p = ds.fit
        core = _core_signal(ds.ts.t, p.A1, p.A2,
                            1/p.zeta1, 1/p.zeta2,
                            p.f1, p.f2, p.phi1, p.phi2)
        scale = p.k_lf if ds.tag == "LF" else p.k_hf
        y_fit = scale * core + shift
        fig.add_trace(
            go.Scattergl(x=ds.ts.t/NS, y=y_fit,
                         line=dict(width=2, dash="dash", color=fit_color),
                         name=label),
            row=row, col=col)


def visualize(triples: List[Tuple[DataSet, DataSet, dict[str, float]]]):
    if not triples:
        print("Нет данных для визуализации.")
        return
    by_key = {(lf.field_mT, lf.temp_K): (lf, hf) for lf, hf, _ in triples}
    keys = sorted(by_key)
    fig = make_subplots(rows=1, cols=2)
    first_key = keys[0]
    ds_lf, ds_hf = by_key[first_key]
    _add_signal_trace(fig, ds_lf, 0.0, 1, 1,
                      raw_color="#1fbe63", fit_color="red",
                      label=f"LF raw ({ds_lf.field_mT} mT, {ds_lf.temp_K} K)")
    _add_signal_trace(fig, ds_hf, 0.0, 1, 2,
                      raw_color="#1fbe63", fit_color="blue",
                      label=f"HF raw ({ds_hf.field_mT} mT, {ds_hf.temp_K} K)")
    fig.update_xaxes(title_text="время (нс)", row=1, col=1)
    fig.update_xaxes(title_text="время (нс)", row=1, col=2)
    fig.update_yaxes(title_text="signal (a.u.)", row=1, col=1)
    fig.update_yaxes(title_text="signal (a.u.)", row=1, col=2)
    print("\nОтображение графика…")
    fig.show()


def visualize_stacked(triples: List[Tuple[DataSet, DataSet]], *, title: str | None = None,
                      outfile: str | None = None) -> None:
    if not triples:
        return
    fig = make_subplots(rows=1, cols=2)
    y_step = 1.0
    for idx, (ds_lf, ds_hf) in enumerate(triples):
        shift = (idx + 1) * y_step
        _add_signal_trace(fig, ds_lf, shift, 1, 1,
                          raw_color="#1fbe63", fit_color="red",
                          label=f"LF {ds_lf.field_mT} mT")
        _add_signal_trace(fig, ds_hf, shift, 1, 2,
                          raw_color="#1fbe63", fit_color="blue",
                          label=f"HF {ds_hf.field_mT} mT")
    fig.update_xaxes(title_text="время (нс)", row=1, col=1)
    fig.update_xaxes(title_text="время (нс)", row=1, col=2)
    fig.update_yaxes(showticklabels=False, row=1, col=1)
    fig.update_yaxes(showticklabels=False, row=1, col=2)
    if title:
        fig.update_layout(title_text=title)
    fig.update_layout(showlegend=False)
    if outfile:
        fig.write_html(outfile)
        print(f"HTML сохранён в {outfile}")
    else:
        print("\nОтображение объединённого графика…")
        fig.show()


def visualize_without_spectra(triples: List[Tuple[DataSet, DataSet]], *, outfile: str | None = None) -> None:
    visualize_stacked(triples, outfile=outfile)
