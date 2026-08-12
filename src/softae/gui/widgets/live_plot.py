"""Lightweight live-updating plot widget using matplotlib embedded in Qt."""

from __future__ import annotations

import math
from typing import Sequence

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtWidgets import QVBoxLayout, QWidget


class LivePlotWidget(QWidget):
    """A single-axis rolling time-series plot with optional setpoint overlay.

    Optionally renders a second process-variable trace (``second_values``)
    on the same axes — e.g. a chamber temperature alongside a stage
    temperature — with its own legend entry and colour.

    Parameters
    ----------
    title : str
        Axes title.
    color : str
        Line colour for the primary process-variable trace.
    y_label : str
        Y-axis label.
    max_points : int
        Maximum visible points (older points scroll off).
    pv_label : str
        Legend label for the primary PV trace.
    sp_label : str
        Legend label for the setpoint overlay.
    second_color : str, optional
        Line colour for the optional second PV trace.
    second_label : str
        Legend label for the optional second PV trace.
    """

    def __init__(
        self,
        title: str = "",
        color: str = "blue",
        y_label: str = "",
        max_points: int = 300,
        *,
        pv_label: str = "PV",
        sp_label: str = "SP",
        second_color: str | None = None,
        second_label: str = "PV2",
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._color = color
        self._max_points = max_points
        self._pv_label = pv_label
        self._sp_label = sp_label
        self._second_color = second_color or "green"
        self._second_label = second_label
        self._legend_done = False

        self._fig = Figure(figsize=(5, 2), dpi=100)
        self._ax = self._fig.add_subplot(111)
        self._ax.set_title(title, fontsize=10)
        if y_label:
            self._ax.set_ylabel(y_label)
        self._fig.tight_layout()

        self._canvas = FigureCanvas(self._fig)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas)

        self._line = None
        self._sp_line = None  # setpoint overlay
        self._second_line = None  # optional second PV trace

    def update_data(
        self,
        values: Sequence[float],
        setpoint_values: Sequence[float] | None = None,
        second_values: Sequence[float] | None = None,
    ) -> None:
        """Redraw the plot with *values*, an optional *setpoint_values*
        overlay, and an optional *second_values* PV trace.

        All supplied sequences should have the same length (most recent at
        right).  ``NaN`` entries in any sequence are drawn as gaps and are
        ignored when computing the y-axis limits.
        """
        # --- Primary PV line ---
        if self._line is None:
            (self._line,) = self._ax.plot(
                values, color=self._color, linewidth=1, label=self._pv_label
            )
        else:
            self._line.set_ydata(values)
            self._line.set_xdata(range(len(values)))

        # --- Second PV line (e.g. Chamber PV) ---
        if second_values is not None:
            if self._second_line is None:
                (self._second_line,) = self._ax.plot(
                    second_values,
                    color=self._second_color,
                    linewidth=1,
                    label=self._second_label,
                )
            else:
                self._second_line.set_ydata(second_values)
                self._second_line.set_xdata(range(len(second_values)))

        # --- Setpoint line ---
        if setpoint_values is not None:
            if self._sp_line is None:
                (self._sp_line,) = self._ax.plot(
                    setpoint_values,
                    color="grey",
                    linewidth=1,
                    linestyle="--",
                    alpha=0.7,
                    label=self._sp_label,
                )
            else:
                self._sp_line.set_ydata(setpoint_values)
                self._sp_line.set_xdata(range(len(setpoint_values)))

        # --- Legend (created once a secondary trace exists) ---
        if not self._legend_done and (
            self._sp_line is not None or self._second_line is not None
        ):
            self._ax.legend(fontsize=7, loc="upper left")
            self._legend_done = True

        # --- Axes limits (NaN-aware) ---
        self._ax.set_xlim(0, max(len(values) - 1, 1))
        all_vals = [v for v in values if v is not None and not math.isnan(v)]
        if setpoint_values:
            all_vals += [v for v in setpoint_values if v is not None and not math.isnan(v)]
        if second_values:
            all_vals += [v for v in second_values if v is not None and not math.isnan(v)]
        if all_vals:
            margin = max(abs(max(all_vals) - min(all_vals)) * 0.1, 0.5)
            self._ax.set_ylim(min(all_vals) - margin, max(all_vals) + margin)

        self._canvas.draw_idle()
