"""Inspection figure: 4-panel Nyquist / Bode / Z'-residual / -Z''-residual."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from softae.analysis.circuit_fitting import predict_fit_curve
from softae.analysis.eis_entry import EISEntry

_DATA_COLOR  = "#000000"
_ZIMAG_COLOR = "#D55E00"
_PHASE_COLOR = "#0072B2"
_FIT_COLOR   = "#E69F00"
_RESID_COLOR = "#555555"


def _predict_fit(entry: EISEntry) -> np.ndarray | None:
    """Return complex impedance array from circuit fit, or None."""
    if entry.fit is None or not entry.fit.success:
        return None
    return predict_fit_curve(entry.fit, entry.eis.frequency)


def build_inspection_figure(entry: EISEntry) -> go.Figure:
    """Four-panel inspection plot for one EISEntry.

    Layout (2 × 2):
      [0,0] Nyquist           [0,1] Bode (|Z| + phase)
      [1,0] Z′ residuals      [1,1] −Z″ residuals

    Parameters
    ----------
    entry : EISEntry

    Returns
    -------
    go.Figure
    """
    freq   = entry.eis.frequency
    z_real = entry.eis.z_real
    z_imag = entry.eis.z_imag_neg

    z_fit = _predict_fit(entry)

    # Dual y-axis for Bode needs secondary_y
    fig = make_subplots(
        rows=2, cols=2,
        specs=[
            [{"secondary_y": False}, {"secondary_y": True}],
            [{"secondary_y": False}, {"secondary_y": False}],
        ],
        subplot_titles=[
            "Nyquist",
            "Bode",
            "Z′ Residuals (%)",
            "−Z″ Residuals (%)",
        ],
        horizontal_spacing=0.12,
        vertical_spacing=0.18,
    )

    # ── Nyquist ──────────────────────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=z_real, y=z_imag,
            mode="markers",
            marker=dict(color=_DATA_COLOR, size=6, symbol="circle"),
            name="Measured",
            hovertemplate="Z′ = %{x:.4g} Ω<br>−Z″ = %{y:.4g} Ω<extra></extra>",
        ),
        row=1, col=1,
    )
    if z_fit is not None:
        fig.add_trace(
            go.Scatter(
                x=z_fit.real, y=-z_fit.imag,
                mode="lines",
                line=dict(color=_FIT_COLOR, width=2),
                name="Model fit",
                hovertemplate="Z′ = %{x:.4g} Ω<br>−Z″ = %{y:.4g} Ω<extra>Fit</extra>",
            ),
            row=1, col=1,
        )

    # ── Bode — Z′ and −Z″  (primary y, log–log) ──────────────────────────
    fig.add_trace(
        go.Scatter(
            x=freq, y=z_real,
            mode="markers",
            marker=dict(color=_DATA_COLOR, size=5, symbol="circle"),
            name="Z′",
            hovertemplate="f = %{x:.4g} Hz<br>Z′ = %{y:.4g} Ω<extra></extra>",
        ),
        row=1, col=2, secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=freq, y=z_imag,
            mode="markers",
            marker=dict(color=_ZIMAG_COLOR, size=5, symbol="square"),
            name="−Z″",
            hovertemplate="f = %{x:.4g} Hz<br>−Z″ = %{y:.4g} Ω<extra></extra>",
        ),
        row=1, col=2, secondary_y=False,
    )
    phase_deg = -entry.eis.phase  # positive peak convention
    fig.add_trace(
        go.Scatter(
            x=freq, y=phase_deg,
            mode="markers",
            marker=dict(color=_PHASE_COLOR, size=5, symbol="triangle-up"),
            name="−Phase",
            hovertemplate="f = %{x:.4g} Hz<br>−φ = %{y:.2f}°<extra></extra>",
        ),
        row=1, col=2, secondary_y=True,
    )
    if z_fit is not None:
        z_fit_real = z_fit.real
        z_fit_imag_neg = -z_fit.imag
        z_fit_phase = -np.angle(z_fit, deg=True)
        fig.add_trace(
            go.Scatter(
                x=freq, y=z_fit_real,
                mode="lines", line=dict(color=_FIT_COLOR, width=2, dash="solid"),
                name="Z′ fit", showlegend=False, hoverinfo="skip",
            ),
            row=1, col=2, secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=freq, y=z_fit_imag_neg,
                mode="lines", line=dict(color=_FIT_COLOR, width=2, dash="dash"),
                name="−Z″ fit", showlegend=False, hoverinfo="skip",
            ),
            row=1, col=2, secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=freq, y=z_fit_phase,
                mode="lines", line=dict(color=_FIT_COLOR, width=2, dash="dot"),
                name="Phase fit", showlegend=False, hoverinfo="skip",
            ),
            row=1, col=2, secondary_y=True,
        )

    # ── Residuals ─────────────────────────────────────────────────────────
    if z_fit is not None:
        from softae.analysis.circuit_fitting import compute_fit_residuals

        resid_real, resid_imag = compute_fit_residuals(entry.eis, z_fit)

        for row_, col_, resid, title in [
            (2, 1, resid_real, "Z′"),
            (2, 2, resid_imag, "−Z″"),
        ]:
            fig.add_trace(
                go.Bar(
                    x=np.log10(freq),
                    y=resid,
                    marker_color=_RESID_COLOR,
                    name=f"{title} residual",
                    showlegend=False,
                    hovertemplate=(
                        "log₁₀(f) = %{x:.2f}<br>"
                        f"{title} residual = " + "%{y:.2f}%<extra></extra>"
                    ),
                ),
                row=row_, col=col_,
            )
            fig.add_hline(y=0, line_color="red", line_width=1, row=row_, col=col_)
    else:
        for row_, col_ in [(2, 1), (2, 2)]:
            fig.add_annotation(
                text="No fit available",
                xref=f"x{2 + (col_-1)}" if row_ == 2 else "x",
                yref=f"y{2 + (col_-1)}" if row_ == 2 else "y",
                x=0.5, y=0.5, showarrow=False,
                font=dict(color="#888", size=12),
                row=row_, col=col_,
            )

    # ── Axis styling ───────────────────────────────────────────────────────
    # Nyquist — equal aspect
    if len(z_real) > 0:
        all_x = list(z_real)
        all_y = list(z_imag)
        if z_fit is not None:
            all_x += list(z_fit.real)
            all_y += list(-z_fit.imag)
        span = max(max(all_x) - min(all_x), max(all_y) - min(all_y), 1.0) * 1.1
        xc = (max(all_x) + min(all_x)) / 2
        yc = (max(all_y) + min(all_y)) / 2
        fig.update_xaxes(range=[xc - span / 2, xc + span / 2], row=1, col=1, title_text="Z′ (Ω)")
        fig.update_yaxes(range=[yc - span / 2, yc + span / 2], row=1, col=1, title_text="−Z″ (Ω)")

    # Bode — log frequency x-axis
    fig.update_xaxes(type="log", title_text="Frequency (Hz)", row=1, col=2)
    fig.update_yaxes(type="log", title_text="Z′, −Z″ (Ω)", row=1, col=2, secondary_y=False)
    fig.update_yaxes(title_text="−Phase (°)", row=1, col=2, secondary_y=True)

    # Residual axes
    fig.update_xaxes(title_text="log₁₀(f / Hz)", row=2, col=1)
    fig.update_xaxes(title_text="log₁₀(f / Hz)", row=2, col=2)
    fig.update_yaxes(title_text="Residual (%)", row=2, col=1)
    fig.update_yaxes(title_text="Residual (%)", row=2, col=2)

    # Title
    title = entry.label
    if entry.fit is not None:
        title += f"  [{entry.fit.model_name}]"
        if not entry.fit.success:
            title += "  ⚠ fit failed"
    if entry.sigma is not None:
        title += f"  |  σ = {entry.sigma:.3e} S/cm"

    fig.update_layout(
        height=640,
        title=dict(text=title, font=dict(size=13, color="#222")),
        legend=dict(orientation="h", y=-0.04, x=0),
        margin=dict(l=10, r=10, t=60, b=40),
        paper_bgcolor="white",
        plot_bgcolor="#fafafa",
        font=dict(size=12),
    )
    # Grid lines for all subplots
    fig.update_xaxes(showgrid=True, gridcolor="#e5e5e5")
    fig.update_yaxes(showgrid=True, gridcolor="#e5e5e5")

    return fig
