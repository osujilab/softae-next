"""Conductivity summary figure: σ scatter/trend colored by run ID."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from softae.analysis.eis_entry import EISEntry

_WONG = [
    "#000000", "#E69F00", "#56B4E9", "#009E73",
    "#F0E442", "#0072B2", "#D55E00", "#CC79A7",
]
_NO_FIT_COLOR = "#cccccc"


def _run_color_map(entries: list[EISEntry]) -> dict[str | None, str]:
    seen: list[str | None] = []
    for e in entries:
        if e.run_id not in seen:
            seen.append(e.run_id)
    extended = _WONG + [f"hsl({i*37%360},70%,45%)" for i in range(len(_WONG), 40)]
    return {rid: extended[k % len(extended)] for k, rid in enumerate(seen)}


def build_conductivity_figure(entries: list[EISEntry]) -> go.Figure:
    """Log-scale conductivity scatter plot.

    Each data point represents one EISEntry.  Points are coloured by
    ``run_id``.  Clicking a point triggers an ``Inspection`` tab switch
    via ``customdata`` → callback in ``callbacks.py``.

    Points without a sigma value are rendered as hollow grey circles at
    a nominal floor (1 × 10⁻¹⁰ S/cm) so they remain visible.

    Parameters
    ----------
    entries : list[EISEntry]

    Returns
    -------
    go.Figure
    """
    fig = go.Figure()

    if not entries:
        fig.add_annotation(
            text="No data — select a run in the sidebar",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="#888"),
        )
        fig.update_layout(
            paper_bgcolor="#fafafa",
            plot_bgcolor="#fafafa",
            margin=dict(l=10, r=10, t=30, b=10),
        )
        return fig

    color_map = _run_color_map(entries)

    # Group by run_id for a single legend entry per run
    seen_runs: set[str | None] = set()
    for idx, entry in enumerate(entries):
        in_legend = entry.run_id not in seen_runs
        seen_runs.add(entry.run_id)
        color = color_map[entry.run_id]

        if entry.sigma is not None:
            fig.add_trace(
                go.Scatter(
                    x=[idx],
                    y=[entry.sigma],
                    mode="markers",
                    marker=dict(
                        size=10,
                        color=color,
                        symbol="circle",
                        line=dict(width=1, color="#fff"),
                    ),
                    name=entry.run_id or "unknown",
                    legendgroup=str(entry.run_id),
                    showlegend=in_legend,
                    customdata=[[idx]],
                    hovertemplate=(
                        f"<b>{entry.label}</b><br>"
                        "σ = %{y:.4e} S/cm<br>"
                        f"R₀ = {entry.fit.R0:.4g} Ω  R₁ = {entry.fit.R1:.4g} Ω<br>"
                        f"Run: {entry.run_id or '—'}<extra></extra>"
                        if entry.fit
                        else (
                            f"<b>{entry.label}</b><br>"
                            "σ = %{y:.4e} S/cm<extra></extra>"
                        )
                    ),
                )
            )
        else:
            # No sigma — hollow grey marker at floor
            fig.add_trace(
                go.Scatter(
                    x=[idx],
                    y=[1e-10],
                    mode="markers",
                    marker=dict(
                        size=10,
                        color="rgba(200,200,200,0)",
                        symbol="circle",
                        line=dict(width=1.5, color=_NO_FIT_COLOR),
                    ),
                    name=entry.run_id or "unknown",
                    legendgroup=str(entry.run_id),
                    showlegend=False,
                    customdata=[[idx]],
                    hovertemplate=(
                        f"<b>{entry.label}</b><br>"
                        "σ = — (no fit)<extra></extra>"
                    ),
                )
            )

    tick_labels = [e.label[:14] for e in entries]

    fig.update_layout(
        yaxis=dict(
            type="log",
            title="σ (S/cm)",
            showgrid=True,
            gridcolor="#e5e5e5",
        ),
        xaxis=dict(
            tickvals=list(range(len(entries))),
            ticktext=tick_labels,
            tickangle=-45,
            showgrid=False,
            title="Sample",
        ),
        title=dict(text="Ionic Conductivity Summary", font=dict(size=14)),
        legend=dict(
            orientation="v",
            title=dict(text="Run ID"),
            x=1.01, y=1,
            xanchor="left",
        ),
        height=480,
        margin=dict(l=10, r=160, t=50, b=100),
        paper_bgcolor="white",
        plot_bgcolor="#fafafa",
        font=dict(size=12),
        hovermode="closest",
        clickmode="event+select",
    )
    return fig
