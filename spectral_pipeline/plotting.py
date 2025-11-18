from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import DataSet, GHZ, NS, logger
from .fit import _core_signal



@dataclass(slots=True)
class TheoryCurves:
    axis: np.ndarray
    freq_hf: np.ndarray
    freq_lf: np.ndarray
    tau_hf: np.ndarray | None = None
    tau_lf: np.ndarray | None = None


def _load_guess_curves(
    directory: Path, field_mT: int, temp_K: int
) -> TheoryCurves | None:
    """Return theoretical curves (frequency and, if present, decay)."""
    path_H = directory / f"H_{field_mT}.npy"
    path_T = directory / f"T_{temp_K}.npy"
    if path_H.exists():
        path = path_H
    elif path_T.exists():
        path = path_T
    else:
        return None
    try:
        arr = np.load(path)
    except Exception as exc:
        logger.warning("Не удалось загрузить %s: %s", path, exc)
        return None
    if arr.ndim != 2 or arr.shape[0] < 3:
        return None
    axis = arr[0]
    hf = arr[1]
    lf = arr[2]
    tau_hf = tau_lf = None
    if arr.shape[0] >= 5:
        tau_hf = arr[3]
        tau_lf = arr[4]
    return TheoryCurves(axis=axis, freq_hf=hf, freq_lf=lf, tau_hf=tau_hf, tau_lf=tau_lf)


def visualize_stacked(
    triples: List[Tuple[DataSet, DataSet]],
    *,
    title: str | None = None,
    outfile: str | None = None,
    use_theory_guess: bool = True,
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
    tau_vs_H: dict[int, list[tuple[int, float, float]]] = {}
    tau_vs_T: dict[int, list[tuple[int, float, float]]] = {}
    amp_vs_H: dict[int, list[tuple[int, float, float]]] = {}
    amp_vs_T: dict[int, list[tuple[int, float, float]]] = {}
    k_vs_H: dict[int, list[tuple[int, float, float]]] = {}
    k_vs_T: dict[int, list[tuple[int, float, float]]] = {}
    phi_vs_H: dict[int, list[tuple[int, float, float]]] = {}
    phi_vs_T: dict[int, list[tuple[int, float, float]]] = {}
    C_vs_H: dict[int, list[tuple[int, float, float]]] = {}
    C_vs_T: dict[int, list[tuple[int, float, float]]] = {}
    for ds_lf, ds_hf in triples_sorted:
        if ds_lf.fit is None:
            continue
        H, T = ds_lf.field_mT, ds_lf.temp_K
        f1, f2 = sorted((ds_lf.fit.f1 / GHZ, ds_lf.fit.f2 / GHZ))
        tau1 = (1.0 / ds_lf.fit.zeta1) / NS
        tau2 = (1.0 / ds_lf.fit.zeta2) / NS
        freq_vs_H.setdefault(T, []).append((H, f1, f2))
        freq_vs_T.setdefault(H, []).append((T, f1, f2))
        tau_vs_H.setdefault(T, []).append((H, tau1, tau2))
        tau_vs_T.setdefault(H, []).append((T, tau1, tau2))
        amp_vs_H.setdefault(T, []).append((H, ds_lf.fit.A1, ds_lf.fit.A2))
        amp_vs_T.setdefault(H, []).append((T, ds_lf.fit.A1, ds_lf.fit.A2))
        k_vs_H.setdefault(T, []).append((H, ds_lf.fit.k_lf, ds_lf.fit.k_hf))
        k_vs_T.setdefault(H, []).append((T, ds_lf.fit.k_lf, ds_lf.fit.k_hf))
        phi_vs_H.setdefault(T, []).append((H, ds_lf.fit.phi1, ds_lf.fit.phi2))
        phi_vs_T.setdefault(H, []).append((T, ds_lf.fit.phi1, ds_lf.fit.phi2))
        C_vs_H.setdefault(T, []).append((H, ds_lf.fit.C_lf, ds_lf.fit.C_hf))
        C_vs_T.setdefault(H, []).append((T, ds_lf.fit.C_lf, ds_lf.fit.C_hf))

    freq_vs_H = {T: sorted(v) for T, v in freq_vs_H.items() if len(v) >= 2}
    freq_vs_T = {H: sorted(v) for H, v in freq_vs_T.items() if len(v) >= 2}
    tau_vs_H = {T: sorted(v) for T, v in tau_vs_H.items() if len(v) >= 2}
    tau_vs_T = {H: sorted(v) for H, v in tau_vs_T.items() if len(v) >= 2}
    amp_vs_H = {T: sorted(v) for T, v in amp_vs_H.items() if len(v) >= 2}
    amp_vs_T = {H: sorted(v) for H, v in amp_vs_T.items() if len(v) >= 2}
    k_vs_H = {T: sorted(v) for T, v in k_vs_H.items() if len(v) >= 2}
    k_vs_T = {H: sorted(v) for H, v in k_vs_T.items() if len(v) >= 2}
    phi_vs_H = {T: sorted(v) for T, v in phi_vs_H.items() if len(v) >= 2}
    phi_vs_T = {H: sorted(v) for H, v in phi_vs_T.items() if len(v) >= 2}
    C_vs_H = {T: sorted(v) for T, v in phi_vs_H.items() if len(v) >= 2}
    C_vs_T = {H: sorted(v) for H, v in phi_vs_T.items() if len(v) >= 2}

    # теоретические кривые из файлов первого приближения
    theory_curves = None
    first_ds = triples_sorted[0][0]
    if first_ds.root:
        theory_curves = _load_guess_curves(
            first_ds.root,
            first_ds.field_mT,
            first_ds.temp_K,
        )
    theory_label_suffix = "" if use_theory_guess else " (plot only)"

    specs = [
        [
            {"type": "xy", "rowspan": 3},
            {"type": "xy", "rowspan": 3},
            {"type": "xy", "rowspan": 3},
            {"type": "xy"},
        ],
        [None, None, None, {"type": "xy"}],
        [None, None, None, {"type": "xy"}],
    ]

    if varying == "T":
        fixed_H = list(all_H)[0]
        titles = [
            f"LF signals (H = {fixed_H} mT)",
            f"HF signals (H = {fixed_H} mT)",
            f"Spectra (H = {fixed_H} mT)",
        ]
    else:  # varying == "H"
        fixed_T = list(all_T)[0]
        titles = [
            f"LF signals (T = {fixed_T} K)",
            f"HF signals (T = {fixed_T} K)",
            f"Spectra (T = {fixed_T} K)",
        ]

    fig = make_subplots(
        rows=3,
        cols=4,
        specs=specs,
        column_widths=[0.26, 0.26, 0.26, 0.22],
        horizontal_spacing=0.06,
        vertical_spacing=0.05,
        shared_xaxes=True,
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
            y_lf = p.k_lf * (p.A1 * np.exp(-ds_lf.ts.t * p.zeta1) * np.cos(2*np.pi*p.f1*ds_lf.ts.t + p.phi1)) + shift
            y_hf = p.k_lf * (p.A2 * np.exp(-ds_lf.ts.t * p.zeta2) * np.cos(2*np.pi*p.f2*ds_lf.ts.t + p.phi2)) + shift
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
            fig.add_trace(
                go.Scattergl(
                    x=ds_lf.ts.t / NS,
                    y=y_lf,
                    opacity=0.5,
                    mode="lines",
                    line=dict(width=2, color='rgb(27,158,119)'),
                    name=f"{varying} = {var_value} {var_label}",
                ),
                1,
                1,
            )
            fig.add_trace(
                go.Scattergl(
                    x=ds_lf.ts.t / NS,
                    y=y_hf,
                    opacity=0.5,
                    mode="lines",
                    line=dict(width=2, color='rgb(166,118,29)'),
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
            y_lf = p.k_hf * (p.A1 * np.exp(-ds_hf.ts.t * p.zeta1) * np.cos(2*np.pi*p.f1*ds_hf.ts.t + p.phi1)) + shift
            y_hf = p.k_hf * (p.A2 * np.exp(-ds_hf.ts.t * p.zeta2) * np.cos(2*np.pi*p.f2*ds_hf.ts.t + p.phi2)) + shift
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
            fig.add_trace(
                go.Scattergl(
                    x=ds_hf.ts.t / NS,
                    y=y_lf,
                    opacity=0.5,
                    mode="lines",
                    line=dict(width=2, color='rgb(27,158,119)'),
                    name=f"{varying} = {var_value} {var_label}",
                ),
                1,
                2,
            )
            fig.add_trace(
                go.Scattergl(
                    x=ds_hf.ts.t / NS,
                    y=y_hf,
                    opacity=0.5,
                    mode="lines",
                    line=dict(width=2, color='rgb(166,118,29)'),
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
                go.Scatter(
                    x=T_vals,
                    y=fLF,
                    mode="markers",
                    line=dict(width=2, color="red"),
                    marker=dict(size=9, color="red"),
                    name=f"f_LF, H = {H_fix} mT",
                ),
                row=1,
                col=4,
            )
            fig.add_trace(
                go.Scatter(
                    x=T_vals,
                    y=fHF,
                    mode="markers",
                    line=dict(width=2, color="blue"),
                    marker=dict(size=9, color="blue"),
                    name=f"f_HF, H = {H_fix} mT",
                ),
                row=1,
                col=4,
            )
        for H_fix, pts in tau_vs_T.items():
            T_vals, tau_LF, tau_HF = zip(*pts)
            fig.add_trace(
                go.Scatter(
                    x=T_vals,
                    y=tau_LF,
                    mode="markers",
                    line=dict(width=2, color="red"),
                    marker=dict(size=9, color="red"),
                    name=f"tau_LF, H = {H_fix} mT",
                ),
                row=2,
                col=4,
            )
            fig.add_trace(
                go.Scatter(
                    x=T_vals,
                    y=tau_HF,
                    mode="markers",
                    line=dict(width=2, color="blue"),
                    marker=dict(size=9, color="blue"),
                    name=f"tau_HF, H = {H_fix} mT",
                ),
                row=2,
                col=4,
            )
        for H_fix, pts in C_vs_T.items():
            T_vals, amp_LF, amp_HF = zip(*pts)
            fig.add_trace(
                go.Scatter(
                    x=T_vals,
                    y=amp_LF,
                    mode="markers",
                    line=dict(width=2, color="red"),
                    marker=dict(size=9, color="red"),
                    name=f"A_LF, H = {H_fix} mT",
                ),
                row=3,
                col=4,
            )
            fig.add_trace(
                go.Scatter(
                    x=T_vals,
                    y=amp_HF,
                    mode="markers",
                    line=dict(width=2, color="blue"),
                    marker=dict(size=9, color="blue"),
                    name=f"A_HF, H = {H_fix} mT",
                ),
                row=3,
                col=4,
            )
        fig.update_xaxes(title_text="Temperature (K)", row=3, col=4)
    else:
        for T_fix, pts in freq_vs_H.items():
            H_vals, fLF, fHF = zip(*pts)
            fig.add_trace(
                go.Scatter(
                    x=H_vals,
                    y=fLF,
                    mode="markers",
                    line=dict(width=2, color="red"),
                    marker=dict(size=9, color="red"),
                    name=f"f_LF, T = {T_fix} K",
                ),
                row=1,
                col=4,
            )
            fig.add_trace(
                go.Scatter(
                    x=H_vals,
                    y=fHF,
                    mode="markers",
                    line=dict(width=2, color="blue"),
                    marker=dict(size=9, color="blue"),
                    name=f"f_HF, T = {T_fix} K",
                ),
                row=1,
                col=4,
            )
        for T_fix, pts in tau_vs_H.items():
            H_vals, tau_LF, tau_HF = zip(*pts)
            fig.add_trace(
                go.Scatter(
                    x=H_vals,
                    y=tau_LF,
                    mode="markers",
                    line=dict(width=2, color="red"),
                    marker=dict(size=9, color="red"),
                    name=f"tau_LF, T = {T_fix} K",
                ),
                row=2,
                col=4,
            )
            fig.add_trace(
                go.Scatter(
                    x=H_vals,
                    y=tau_HF,
                    mode="markers",
                    line=dict(width=2, color="blue"),
                    marker=dict(size=9, color="blue"),
                    name=f"tau_HF, T = {T_fix} K",
                ),
                row=2,
                col=4,
            )
        for T_fix, pts in C_vs_H.items():
            H_vals, amp_LF, amp_HF = zip(*pts)
            fig.add_trace(
                go.Scatter(
                    x=H_vals,
                    y=amp_LF,
                    mode="markers",
                    line=dict(width=2, color="red"),
                    marker=dict(size=9, color="red"),
                    name=f"A_LF, T = {T_fix} K",
                ),
                row=3,
                col=4,
            )
            fig.add_trace(
                go.Scatter(
                    x=H_vals,
                    y=amp_HF,
                    mode="markers",
                    line=dict(width=2, color="blue"),
                    marker=dict(size=9, color="blue"),
                    name=f"A_HF, T = {T_fix} K",
                ),
                row=3,
                col=4,
            )
        fig.update_xaxes(title_text="Magnetic field (mT)", row=3, col=4)
    fig.update_yaxes(title_text="Frequency (GHz)", row=1, col=4)
    fig.update_yaxes(title_text="Decay time (ns)", row=2, col=4)
    fig.update_yaxes(title_text="Amplitude", row=3, col=4)

    if theory_curves is not None:
        pass
        axis = theory_curves.axis
        hf_th = theory_curves.freq_hf
        lf_th = theory_curves.freq_lf
        var_vals = [key_func(ds_lf) for ds_lf, _ in triples_sorted]
        lo, hi = min(var_vals), max(var_vals)
        mask = (axis >= lo) & (axis <= hi)
        axis = axis[mask]
        hf_th = hf_th[mask]
        lf_th = lf_th[mask]
        if axis.size:
            col_idx = 4
            fig.add_trace(
                go.Scatter(
                    x=axis,
                    y=lf_th,
                    mode="lines",
                    line=dict(color="red"),
                    name=f"LF theory{theory_label_suffix}",
                ),
                row=1,
                col=col_idx,
            )
            fig.add_trace(
                go.Scatter(
                    x=axis,
                    y=hf_th,
                    mode="lines",
                    line=dict(color="blue"),
                    name=f"HF theory{theory_label_suffix}",
                ),
                row=1,
                col=col_idx,
            )
            if theory_curves.tau_lf is not None and theory_curves.tau_hf is not None:
                tau_lf = theory_curves.tau_lf[mask]
                tau_hf = theory_curves.tau_hf[mask]
                fig.add_trace(
                    go.Scatter(
                        x=axis,
                        y=tau_lf,
                        mode="lines",
                        line=dict(color="red"),
                        name=f"tau_LF theory{theory_label_suffix}",
                    ),
                    row=2,
                    col=col_idx,
                )
                fig.add_trace(
                    go.Scatter(
                        x=axis,
                        y=tau_hf,
                        mode="lines",
                        line=dict(color="blue"),
                        name=f"tau_HF theory{theory_label_suffix}",
                    ),
                    row=2,
                    col=col_idx,
                )

    spectra_HF: list[tuple[np.ndarray, np.ndarray, str]] = []
    spectra_LF: list[tuple[np.ndarray, np.ndarray, str]] = []
    offset = 0.0
    shift_f = 1.0

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
        height=1200,
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
        title_text="Time (ns)",
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
        title_text="Time (ns)",
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
        title_text="Frequency (GHz)",
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
    for r in (2, 3):
        fig.update_xaxes(
            showline=True,
            linewidth=1,
            linecolor="black",
            mirror=True,
            showgrid=True,
            gridcolor="#cccccc",
            gridwidth=1,
            row=r,
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
            row=r,
            col=4,
        )
    fig.update_xaxes(title_font=dict(size=28), tickfont=dict(size=24))
    fig.update_yaxes(title_font=dict(size=28), tickfont=dict(size=24))

    if outfile:
        fig.write_html(outfile)
        print(f"HTML сохранён в {outfile}")
    else:
        print("\nОтображение объединённого графика…")
        fig.show()
