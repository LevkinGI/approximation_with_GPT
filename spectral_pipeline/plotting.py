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


def visualize_stacked(
    triples: List[Tuple[DataSet, DataSet]], *, title: str | None = None,
    outfile: str | None = None
) -> None:
    """Рисует все LF/HF сигналы и их аппроксимации с вертикальными смещениями
    и добавляет сводные графики спектров и частот.
    """

    if not triples:
        return

    RAW_CLR = "#1fbe63"
    FIT_LF = "red"
    FIT_HF = "blue"
    BASE_CLR = "#606060"

    all_H = {ds_lf.field_mT for ds_lf, _ in triples}
    all_T = {ds_lf.temp_K for ds_lf, _ in triples}
    if len(all_H) == 1 and len(all_T) > 1:
        varying, var_label = "T", "K"
        key_func = lambda ds_lf: ds_lf.temp_K
    elif len(all_T) == 1 and len(all_H) > 1:
        varying, var_label = "H", "mT"
        key_func = lambda ds_lf: ds_lf.field_mT
    else:
        raise RuntimeError("Скрипт ожидает изменение только H или T.")

    triples_sorted = sorted(triples, key=lambda p: key_func(p[0]))

    ranges = []
    for pair in triples_sorted:
        for ds in pair:
            if ds.fit is not None:
                mask = ds.ts.t > 0
                p = ds.fit
                core = _core_signal(
                    ds.ts.t, p.A1, p.A2,
                    1 / p.zeta1, 1 / p.zeta2,
                    p.f1, p.f2, p.phi1, p.phi2,
                )
                y_fit = (
                    p.k_lf * core + p.C_lf
                    if ds.tag == "LF"
                    else p.k_hf * core + p.C_hf
                )
                ranges.append(np.ptp(y_fit[mask]))
            else:
                ranges.append(np.ptp(ds.ts.s))

    y_step = 0.8 * max(ranges + [1e-3])

    freq_vs_H: dict[int, list[tuple[int, float, float]]] = {}
    freq_vs_T: dict[int, list[tuple[int, float, float]]] = {}
    amp_vs_H: dict[int, list[tuple[int, float, float]]] = {}
    amp_vs_T: dict[int, list[tuple[int, float, float]]] = {}
    for ds_lf, ds_hf in triples_sorted:
        if ds_lf.fit is None:
            continue
        H, T = ds_lf.field_mT, ds_lf.temp_K
        f1, f2 = sorted((ds_lf.fit.f1 / GHZ, ds_lf.fit.f2 / GHZ))
        freq_vs_H.setdefault(T, []).append((H, f1, f2))
        freq_vs_T.setdefault(H, []).append((T, f1, f2))
        amp_vs_H.setdefault(T, []).append((H, ds_lf.fit.A1, ds_lf.fit.A2))
        amp_vs_T.setdefault(H, []).append((T, ds_lf.fit.A1, ds_lf.fit.A2))

    freq_vs_H = {T: sorted(v) for T, v in freq_vs_H.items() if len(v) >= 2}
    freq_vs_T = {H: sorted(v) for H, v in freq_vs_T.items() if len(v) >= 2}
    amp_vs_H = {T: sorted(v) for T, v in amp_vs_H.items() if len(v) >= 2}
    amp_vs_T = {H: sorted(v) for H, v in amp_vs_T.items() if len(v) >= 2}

    specs = [
        [
            {"type": "xy", "rowspan": 2},
            {"type": "xy", "rowspan": 2},
            {"type": "xy", "rowspan": 2},
            {"type": "xy"},
        ],
        [None, None, None, {"type": "xy"}],
    ]

    if varying == "T":
        fixed_H = list(all_H)[0]
        titles = [
            f"LF signals (H = {fixed_H} mT)",
            f"HF signals (H = {fixed_H} mT)",
            f"Spectra (H = {fixed_H} mT)",
            f"Frequencies (H = {fixed_H} mT)",
            f"Amplitudes (H = {fixed_H} mT)",
        ]
    else:  # varying == "H"
        fixed_T = list(all_T)[0]
        titles = [
            f"LF signals (T = {fixed_T} K)",
            f"HF signals (T = {fixed_T} K)",
            f"Spectra (T = {fixed_T} K)",
            f"Frequencies (T = {fixed_T} K)",
            f"Amplitudes (T = {fixed_T} K)",
        ]

    fig = make_subplots(
        rows=2,
        cols=4,
        specs=specs,
        column_widths=[0.26, 0.26, 0.26, 0.22],
        horizontal_spacing=0.06,
        vertical_spacing=0.15,
        subplot_titles=tuple(titles),
    )

    for idx, (ds_lf, ds_hf) in enumerate(triples_sorted):
        shift = (idx + 1) * y_step
        var_value = key_func(ds_lf)
        tmin_lf, tmax_lf = ds_lf.ts.t[0] / NS, ds_lf.ts.t[-1] / NS
        tmin_hf, tmax_hf = ds_hf.ts.t[0] / NS, ds_hf.ts.t[-1] / NS

        for col, (tmin, tmax) in ((1, (tmin_lf, tmax_lf)), (2, (tmin_hf, tmax_hf))):
            fig.add_trace(
                go.Scattergl(
                    x=[tmin, tmax],
                    y=[shift, shift],
                    line=dict(width=1, color=BASE_CLR),
                    mode="lines",
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=col,
            )

        if ds_lf.fit:
            p = ds_lf.fit
        y = ds_lf.ts.s + shift
        y -= p.C_lf if ds_lf.fit else ds_lf.ts.s.mean()
        fig.add_trace(
            go.Scattergl(
                x=ds_lf.ts.t / NS,
                y=y,
                line=dict(width=3, color=RAW_CLR),
                name=f"{varying} = {var_value} {var_label}",
            ),
            1,
            1,
        )

        if ds_lf.fit:
            core = _core_signal(
                ds_lf.ts.t,
                p.A1,
                p.A2,
                1 / p.zeta1,
                1 / p.zeta2,
                p.f1,
                p.f2,
                p.phi1,
                p.phi2,
            )
            y_fit = p.k_lf * core + shift
            fig.add_trace(
                go.Scattergl(
                    x=ds_lf.ts.t / NS,
                    y=y_fit,
                    line=dict(width=2, dash="dash", color=FIT_LF),
                    name=f"{varying} = {var_value} {var_label}",
                ),
                1,
                1,
            )

        if ds_hf.fit:
            p = ds_hf.fit
        y = ds_hf.ts.s + shift
        y -= p.C_hf if ds_hf.fit else ds_hf.ts.s.mean()
        fig.add_trace(
            go.Scattergl(
                x=ds_hf.ts.t / NS,
                y=y,
                line=dict(width=3, color=RAW_CLR),
                name=f"{varying} = {var_value} {var_label}",
            ),
            1,
            2,
        )

        if ds_hf.fit:
            core = _core_signal(
                ds_hf.ts.t,
                p.A1,
                p.A2,
                1 / p.zeta1,
                1 / p.zeta2,
                p.f1,
                p.f2,
                p.phi1,
                p.phi2,
            )
            y_fit = p.k_hf * core + shift
            fig.add_trace(
                go.Scattergl(
                    x=ds_hf.ts.t / NS,
                    y=y_fit,
                    line=dict(width=2, dash="dash", color=FIT_HF),
                    name=f"{varying} = {var_value} {var_label}",
                ),
                1,
                2,
            )

        fig.add_annotation(
            x=tmax_lf,
            y=shift,
            text=f"{var_value} {var_label}",
            showarrow=False,
            xanchor="left",
            font=dict(size=16),
            row=1,
            col=1,
        )
        fig.add_annotation(
            x=tmax_hf,
            y=shift,
            text=f"{var_value} {var_label}",
            showarrow=False,
            xanchor="left",
            font=dict(size=16),
            row=1,
            col=2,
        )

    if varying == "T":
        for H_fix, pts in freq_vs_T.items():
            T_vals, fLF, fHF = zip(*pts)
            fig.add_trace(
                go.Scatter(x=T_vals, y=fLF, mode="markers+lines", line=dict(width=2, color="red"), name=f"f_LF, H = {H_fix} mT"),
                row=1,
                col=4,
            )
            fig.add_trace(
                go.Scatter(x=T_vals, y=fHF, mode="markers+lines", line=dict(width=2, color="blue"), name=f"f_HF, H = {H_fix} mT"),
                row=1,
                col=4,
            )
        fig.update_xaxes(title_text="T (K)", row=1, col=4)
        for H_fix, pts in amp_vs_T.items():
            T_vals, amp_LF, amp_HF = zip(*pts)
            fig.add_trace(
                go.Scatter(x=T_vals, y=amp_LF, mode="markers+lines", line=dict(width=2, color="red"), name=f"f_LF, H = {H_fix} mT"),
                row=2,
                col=4,
            )
            fig.add_trace(
                go.Scatter(x=T_vals, y=amp_HF, mode="markers+lines", line=dict(width=2, color="blue"), name=f"f_HF, H = {H_fix} mT"),
                row=2,
                col=4,
            )
        fig.update_xaxes(title_text="T (K)", row=2, col=4)
    else:
        for T_fix, pts in freq_vs_H.items():
            H_vals, fLF, fHF = zip(*pts)
            fig.add_trace(
                go.Scatter(x=H_vals, y=fLF, mode="markers+lines", line=dict(width=2, color="red"), name=f"f_LF, T = {T_fix} K"),
                row=1,
                col=4,
            )
            fig.add_trace(
                go.Scatter(x=H_vals, y=fHF, mode="markers+lines", line=dict(width=2, color="blue"), name=f"f_HF, T = {T_fix} K"),
                row=1,
                col=4,
            )
        fig.update_xaxes(title_text="H (mT)", row=1, col=4)
        for T_fix, pts in amp_vs_H.items():
            H_vals, amp_LF, amp_HF = zip(*pts)
            fig.add_trace(
                go.Scatter(x=H_vals, y=amp_LF, mode="markers+lines", line=dict(width=2, color="red"), name=f"f_LF, T = {T_fix} K"),
                row=2,
                col=4,
            )
            fig.add_trace(
                go.Scatter(x=H_vals, y=amp_HF, mode="markers+lines", line=dict(width=2, color="blue"), name=f"f_HF, T = {T_fix} K"),
                row=2,
                col=4,
            )
        fig.update_xaxes(title_text="H (mT)", row=2, col=4)
    fig.update_yaxes(title_text="f (GHz)", row=1, col=4)
    fig.update_yaxes(title_text="amp", row=2, col=4)

    spectra_HF: list[tuple[np.ndarray, np.ndarray, str]] = []
    spectra_LF: list[tuple[np.ndarray, np.ndarray, str]] = []

    for ds_lf, ds_hf in triples_sorted:
        if ds_hf.freq_fft is None or ds_lf.freq_fft is None:
            continue
        depth_val = ds_hf.temp_K if varying == "T" else ds_hf.field_mT
        spectra_HF.append(
            (
                ds_hf.freq_fft / GHZ,
                ds_hf.asd_fft / np.max(ds_hf.asd_fft),
                f"{depth_val:.0f} {var_label}",
            )
        )
        spectra_LF.append(
            (
                ds_lf.freq_fft / GHZ,
                ds_lf.asd_fft / np.max(ds_lf.asd_fft),
                f"{depth_val:.0f} {var_label}",
            )
        )

    if spectra_HF:
        spectra_HF.sort(key=lambda tpl: tpl[2])
        spectra_LF.sort(key=lambda tpl: tpl[2])
        shift_hf = 1.2 * max(np.nanmax(a_hf) for (_, a_hf, _) in spectra_HF)
        shift_lf = 1.2 * max(np.nanmax(a_lf) for (_, a_lf, _) in spectra_LF)
        shift_f = max(shift_hf, shift_lf)

        for idx, ((f_GHz, amp, lbl), (f_lf, amp_lf, _)) in enumerate(
            zip(spectra_HF, spectra_LF)
        ):
            offset = (idx + 1) * shift_f
            y_vals = amp + offset
            f_GHz = f_GHz[: np.argmin(np.abs(f_GHz - 80))]

            fig.add_trace(
                go.Scattergl(
                    x=f_GHz,
                    y=y_vals,
                    mode="lines",
                    line=dict(color=FIT_HF, width=2),
                    name=lbl,
                    showlegend=False,
                ),
                row=1,
                col=3,
            )

            y_vals = amp_lf + offset
            f_lf = f_lf[: np.argmin(np.abs(f_lf - 80))]

            fig.add_trace(
                go.Scattergl(
                    x=f_lf,
                    y=y_vals,
                    mode="lines",
                    line=dict(color=FIT_LF, width=2),
                    name=lbl,
                    showlegend=False,
                ),
                row=1,
                col=3,
            )

            fig.add_annotation(
                x=f_GHz[-1],
                y=offset,
                text=lbl,
                xanchor="left",
                showarrow=False,
                font=dict(size=16),
                row=1,
                col=3,
            )

            fig.add_trace(
                go.Scattergl(
                    x=[f_GHz[0], f_GHz[-1]],
                    y=[offset, offset],
                    line=dict(width=1, color=BASE_CLR),
                    mode="lines",
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=3,
            )

    fig.update_layout(
        showlegend=False,
        hovermode="x unified",
        font=dict(size=16),
        width=2000,
        height=1000,
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    for annotation in fig["layout"]["annotations"][: len(titles)]:
        annotation["font"] = dict(size=22)

    fig.update_xaxes(
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
        showticklabels=True,
        ticks="inside",
        showgrid=True,
        gridcolor="#cccccc",
        gridwidth=1,
        row=1,
        col=1,
        title_text="time (ns)",
    )
    fig.update_xaxes(
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
        showticklabels=True,
        ticks="inside",
        showgrid=True,
        gridcolor="#cccccc",
        gridwidth=1,
        row=1,
        col=2,
        title_text="time (ns)",
    )
    fig.update_xaxes(
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
        showticklabels=True,
        ticks="inside",
        showgrid=True,
        gridcolor="#cccccc",
        gridwidth=1,
        row=1,
        col=3,
        title_text="frequency (GHz)",
    )
    fig.update_yaxes(
        range=[0, shift + y_step],
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
        showticklabels=False,
        row=1,
        col=1,
    )
    fig.update_yaxes(
        range=[0, shift + y_step],
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
        showticklabels=False,
        row=1,
        col=2,
    )
    fig.update_yaxes(
        range=[0, offset + shift_f],
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
        showticklabels=False,
        row=1,
        col=3,
    )
    fig.update_xaxes(
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
        showgrid=True,
        gridcolor="#cccccc",
        gridwidth=1,
        row=1,
        col=4,
    )
    fig.update_yaxes(
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
        showgrid=True,
        gridcolor="#cccccc",
        gridwidth=1,
        row=1,
        col=4,
    )
    fig.update_xaxes(
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
        showgrid=True,
        gridcolor="#cccccc",
        gridwidth=1,
        row=2,
        col=4,
    )
    fig.update_yaxes(
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
        showgrid=True,
        gridcolor="#cccccc",
        gridwidth=1,
        row=2,
        col=4,
        range=[0, None],
    )

    if outfile:
        fig.write_html(outfile)
        print(f"HTML сохранён в {outfile}")
    else:
        print("\nОтображение объединённого графика…")
        fig.show()


def visualize_without_spectra(triples: List[Tuple[DataSet, DataSet]], *, outfile: str | None = None) -> None:
    visualize_stacked(triples, outfile=outfile)
