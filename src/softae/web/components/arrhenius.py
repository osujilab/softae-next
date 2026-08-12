"""Arrhenius panel figure builder.

When arrhenius_data is empty (Project 2 not yet run), returns a
placeholder figure.  When data is present (populated by the
``ArrheniusSweep`` module, spec: ``temp_eis_arrhenius_spec.md``), renders:

  Left  — ln(σ) vs 1000/T  scatter + linear-fit overlay per channel
  Right — Eₐ bar chart per channel (eV or kJ/mol)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_WONG = [
    "#000000", "#E69F00", "#56B4E9", "#009E73",
    "#F0E442", "#0072B2", "#D55E00", "#CC79A7",
]


def build_arrhenius_figure(
    arrhenius_data: list[dict[str, Any]],
    unit: str = "eV",
) -> go.Figure:
    """Build the Arrhenius panel.

    Parameters
    ----------
    arrhenius_data : list[dict]
        Each dict must contain keys:
          ``channel``, ``temperatures_C``, ``conductivities``,
          ``Ea_eV``, ``Ea_kJ_per_mol``, ``ln_A``, ``R_squared``,
          ``fit_success``.
        Produced by ``DataStore.query_arrhenius()`` or the JSON sidecar.
    unit : str
        ``"eV"`` or ``"kJ/mol"`` for the Eₐ bar chart y-axis.

    Returns
    -------
    go.Figure
    """
    if not arrhenius_data:
        return _placeholder_figure()

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["ln(σ) vs 1000/T", f"Activation Energy ({unit})"],
        horizontal_spacing=0.14,
    )

    for k, row_data in enumerate(arrhenius_data):
        color = _WONG[k % len(_WONG)]
        ch = row_data.get("channel", k + 1)
        temps_C = row_data.get("temperatures_C") or []
        sigmas = row_data.get("conductivities") or []
        Ea = row_data.get("Ea_eV") if unit == "eV" else row_data.get("Ea_kJ_per_mol")
        ln_A = row_data.get("ln_A")
        R2 = row_data.get("R_squared")
        success = row_data.get("fit_success", False)

        if temps_C and sigmas:
            T_K = np.array([t + 273.15 for t in temps_C])
            sigma_arr = np.array(sigmas, dtype=float)
            valid = ~np.isnan(sigma_arr) & (sigma_arr > 0)
            x_inv = 1000.0 / T_K[valid]
            y_ln = np.log(sigma_arr[valid])

            fig.add_trace(
                go.Scatter(
                    x=x_inv,
                    y=y_ln,
                    mode="markers",
                    marker=dict(size=8, color=color),
                    name=f"Ch{ch:02d}",
                    legendgroup=f"ch{ch}",
                    hovertemplate=(
                        f"<b>Ch{ch:02d}</b><br>"
                        "1000/T = %{x:.4f} K⁻¹<br>"
                        "ln(σ) = %{y:.4f}<extra></extra>"
                    ),
                ),
                row=1, col=1,
            )

            # Linear fit overlay
            if success and ln_A is not None and Ea is not None and len(x_inv) >= 2:
                KB_eV = 8.617333e-5
                slope = (-Ea / KB_eV / 1000.0) if unit == "eV" else (-Ea / (8.314e-3 * 1000.0))
                x_line = np.linspace(x_inv.min(), x_inv.max(), 80)
                y_line = ln_A + slope * x_line
                r2_str = f"R² = {R2:.4f}" if R2 is not None else ""
                fig.add_trace(
                    go.Scatter(
                        x=x_line,
                        y=y_line,
                        mode="lines",
                        line=dict(color=color, width=2, dash="dash"),
                        name=f"Ch{ch:02d} fit",
                        legendgroup=f"ch{ch}",
                        showlegend=False,
                        hovertemplate=(
                            f"<b>Ch{ch:02d} fit</b><br>"
                            f"Eₐ = {Ea:.4f} {unit}<br>"
                            f"{r2_str}<extra></extra>"
                        ),
                    ),
                    row=1, col=1,
                )

        # Ea bar chart
        if Ea is not None:
            fig.add_trace(
                go.Bar(
                    x=[f"Ch{ch:02d}"],
                    y=[Ea],
                    marker_color=color,
                    name=f"Ch{ch:02d}",
                    legendgroup=f"ch{ch}",
                    showlegend=False,
                    hovertemplate=(
                        f"<b>Ch{ch:02d}</b><br>"
                        f"Eₐ = {Ea:.4f} {unit}<br>"
                        f"R² = {R2:.4f}<extra></extra>"
                        if R2 is not None else
                        f"<b>Ch{ch:02d}</b><br>Eₐ = {Ea:.4f} {unit}<extra></extra>"
                    ),
                ),
                row=1, col=2,
            )

    fig.update_xaxes(title_text="1000/T (K⁻¹)", row=1, col=1, showgrid=True, gridcolor="#e5e5e5")
    fig.update_yaxes(title_text="ln(σ) [ln(S/cm)]", row=1, col=1, showgrid=True, gridcolor="#e5e5e5")
    fig.update_xaxes(title_text="Channel", row=1, col=2)
    fig.update_yaxes(title_text=f"Eₐ ({unit})", row=1, col=2, showgrid=True, gridcolor="#e5e5e5")

    fig.update_layout(
        height=480,
        legend=dict(orientation="v", x=1.02, y=1),
        margin=dict(l=10, r=150, t=50, b=40),
        paper_bgcolor="white",
        plot_bgcolor="#fafafa",
        font=dict(size=12),
        barmode="group",
    )
    return fig


def _placeholder_figure() -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=(
            "<b>No Arrhenius data yet</b><br>"
            "<span style='font-size:13px;color:#666'>"
            "Run a temperature-stepped EIS sweep to populate this panel.</span>"
        ),
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=15, color="#444"),
        align="center",
    )
    fig.update_layout(
        height=480,
        paper_bgcolor="#fafafa",
        plot_bgcolor="#fafafa",
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig
