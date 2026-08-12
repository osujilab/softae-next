"""Overview figure: grid of mini-Nyquist thumbnails, one per EISEntry."""

from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from softae.analysis.circuit_fitting import predict_fit_curve
from softae.analysis.eis_entry import EISEntry

# Wong (2011) colour-blind-safe palette
_WONG = [
    "#000000", "#E69F00", "#56B4E9", "#009E73",
    "#F0E442", "#0072B2", "#D55E00", "#CC79A7",
]
_FIT_COLOR = "#E69F00"
_DATA_COLOR = "#000000"


def _run_color_map(entries: list[EISEntry]) -> dict[str | None, str]:
    seen: list[str | None] = []
    for e in entries:
        if e.run_id not in seen:
            seen.append(e.run_id)
    extended = _WONG + [f"hsl({i*37%360},70%,45%)" for i in range(len(_WONG), 40)]
    return {rid: extended[k % len(extended)] for k, rid in enumerate(seen)}


def build_overview_figure(
    entries: list[EISEntry],
    cols: int = 4,
) -> go.Figure:
    """Build a grid of mini-Nyquist plots, one per entry.

    Each cell shows the measured data (scatter) plus a circuit-fit
    overlay (line) if available.  Sigma is annotated below each plot.

    Parameters
    ----------
    entries : list[EISEntry]
        Entries to display.
    cols : int
        Number of columns in the grid.

    Returns
    -------
    go.Figure
    """
    n = len(entries)
    if n == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No data — select a run in the sidebar",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="#888"),
        )
        fig.update_layout(
            paper_bgcolor="#fafafa",
            plot_bgcolor="#fafafa",
            margin=dict(l=10, r=10, t=10, b=10),
        )
        return fig

    rows = math.ceil(n / cols)
    color_map = _run_color_map(entries)

    subplot_titles = []
    for e in entries:
        sigma_str = f"σ = {e.sigma:.3e} S/cm" if e.sigma is not None else "σ = —"
        subplot_titles.append(f"{e.label[:20]}<br><sup>{sigma_str}</sup>")

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=subplot_titles,
        shared_xaxes=False,
        shared_yaxes=False,
        horizontal_spacing=0.06,
        vertical_spacing=0.12,
    )

    for idx, entry in enumerate(entries):
        row = idx // cols + 1
        col = idx % cols + 1
        color = color_map[entry.run_id]

        fig.add_trace(
            go.Scatter(
                x=entry.eis.z_real,
                y=entry.eis.z_imag_neg,
                mode="markers",
                marker=dict(size=4, color=color),
                name=entry.label,
                showlegend=False,
                customdata=[[idx]],
                hovertemplate=(
                    f"<b>{entry.label}</b><br>"
                    "Z′ = %{x:.3g} Ω<br>"
                    "−Z″ = %{y:.3g} Ω<extra></extra>"
                ),
            ),
            row=row, col=col,
        )

        # Circuit-fit overlay
        if entry.fit is not None and entry.fit.success:
            z_fit = predict_fit_curve(entry.fit, entry.eis.frequency)
            if z_fit is not None:
                fig.add_trace(
                    go.Scatter(
                        x=z_fit.real,
                        y=-z_fit.imag,
                        mode="lines",
                        line=dict(color=_FIT_COLOR, width=1.5),
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=row, col=col,
                )

    # Equal-aspect axes per cell
    for idx in range(n):
        row = idx // cols + 1
        col = idx % cols + 1
        axis_i = "" if (row == 1 and col == 1) else str(idx + 1)
        entry = entries[idx]
        x_range = [entry.eis.z_real.min(), entry.eis.z_real.max()]
        y_range = [entry.eis.z_imag_neg.min(), entry.eis.z_imag_neg.max()]
        span = max(x_range[1] - x_range[0], y_range[1] - y_range[0], 1.0)
        x_center = (x_range[0] + x_range[1]) / 2
        y_center = (y_range[0] + y_range[1]) / 2
        fig.update_layout(
            **{
                f"xaxis{axis_i}": dict(
                    range=[x_center - span * 0.55, x_center + span * 0.55],
                    showticklabels=False,
                    showgrid=True,
                    gridcolor="#eee",
                ),
                f"yaxis{axis_i}": dict(
                    range=[y_center - span * 0.55, y_center + span * 0.55],
                    showticklabels=False,
                    showgrid=True,
                    gridcolor="#eee",
                ),
            }
        )

    cell_h = max(180, 740 // rows)
    fig.update_layout(
        height=max(300, rows * cell_h),
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="#fafafa",
        plot_bgcolor="#fafafa",
        font=dict(size=11),
        title_font_size=11,
    )
    # Style subplot title annotations to be compact
    for ann in fig.layout.annotations:
        ann.font.size = 10

    return fig
