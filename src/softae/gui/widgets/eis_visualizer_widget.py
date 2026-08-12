"""EIS measurement visualizer widget.

Provides a three-pane (Overview / Inspection / Conductivity) GUI for
browsing EIS measurements and circuit-fit results stored in a DataStore
or supplied as a static list.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import QObject, QThread, QTimer, Qt, Signal
from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from softae.analysis.circuit_fitting import (  # noqa: F401
    FitResult,
    predict_fit_curve,
)
from softae.analysis.eis_data import EISResult
# EISEntry now lives canonically in softae.analysis; re-exported here so existing
# ``from softae.gui.widgets.eis_visualizer_widget import EISEntry`` keeps working.
from softae.analysis.eis_entry import EISEntry  # noqa: F401

# Default electrode geometry (L, t, w) in cm — the historical literal used
# across the analysis pipeline when no per-sample geometry is known.
DEFAULT_GEOMETRY: tuple[float, float, float] = (0.2, 0.175, 0.2)

# ---------------------------------------------------------------------------
# Wong (2011) colour-blind-safe palette constants
# ---------------------------------------------------------------------------

_Z_REAL_COLOR = "#000000"
_Z_IMAG_COLOR = "#D55E00"
_PHASE_COLOR  = "#0072B2"
_FIT_COLOR    = "#E69F00"
_RESID_COLOR  = "#555555"
_WONG_CYCLE   = [
    "#000000", "#E69F00", "#56B4E9", "#009E73",
    "#F0E442", "#0072B2", "#D55E00", "#CC79A7",
]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class _NonScrollCanvas(FigureCanvas):
    """FigureCanvas that passes wheel events to the parent widget and
    erases its full background before every paint.

    ``FigureCanvasQTAgg`` sets ``WA_OpaquePaintEvent`` which tells Qt to
    skip the normal background erase, assuming matplotlib will paint every
    pixel.  Matplotlib only paints the figure patch and axes, leaving stale
    pixels in the margin area.  Overriding ``paintEvent`` with an explicit
    ``fillRect`` before delegating to the parent eliminates all residual
    artifacts.
    """

    def wheelEvent(self, event: Any) -> None:  # type: ignore[override]
        event.ignore()

    def paintEvent(self, event: Any) -> None:  # type: ignore[override]
        from PySide6.QtGui import QPainter
        painter = QPainter(self)
        painter.fillRect(self.rect(), self.palette().window())
        painter.end()
        super().paintEvent(event)


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------


def _fitresult_from_fit_row(
    fit_row: dict,
) -> tuple[FitResult, float | None, tuple[float, float, float] | None]:
    """Rebuild a :class:`FitResult` (+ stored sigma + geometry) from a DB fit row.

    Shared by :class:`DataStoreSource` and the Analysis tab's list loader so both
    paths reconstruct persisted fits identically.
    """
    raw_params = fit_row.get("parameters_json", "[]")
    try:
        params_list = (
            json.loads(raw_params) if isinstance(raw_params, str) else raw_params
        )
        if isinstance(params_list, list) and len(params_list) > 0:
            param_arr = np.array(params_list, dtype=float)
        else:
            param_arr = np.array([float(fit_row["R0"] or 0), float(fit_row["R1"] or 0)])
    except Exception:
        param_arr = np.array([float(fit_row["R0"] or 0), float(fit_row["R1"] or 0)])

    fit = FitResult(
        model_name=fit_row["model_name"],
        parameters=param_arr,
        R0=float(fit_row["R0"] or 0),
        R1=float(fit_row["R1"] or 0),
        R0_guess=float(fit_row["R0"] or 0),
        R1_guess=float(fit_row["R1"] or 0),
        z_indices=[0, 1],
        success=bool(fit_row["success"]),
        error_msg=fit_row["error_msg"] or "",
    )
    raw_sigma = fit_row.get("sigma_S_per_cm")
    sigma = float(raw_sigma) if raw_sigma is not None else None

    geom: tuple[float, float, float] | None = None
    L, t, w = (
        fit_row.get("electrode_L_cm"),
        fit_row.get("electrode_t_cm"),
        fit_row.get("electrode_w_cm"),
    )
    if L is not None and t is not None and w is not None:
        geom = (float(L), float(t), float(w))
    return fit, sigma, geom


def fit_entry(
    entry: EISEntry, model_name: str, L: float, t: float, w: float
) -> EISEntry:
    """Fit *entry*'s spectrum with *model_name* and geometry (L, t, w) in cm.

    Mutates and returns the entry: sets ``fit`` and ``geometry`` and, on a
    successful fit, the geometry-derived ``sigma`` (``None`` if σ is non-finite
    or the fit did not converge). Raises if the ``impedance`` backend is missing
    or *model_name* is unknown — the caller surfaces that to the user.

    The single choke point for the EIS Browser: embedded pane and pop-out window
    both fit through here, so routing it through
    :func:`~softae.analysis.eis.engine.analyze_spectrum` (``engine`` left unset, so
    ``[eis] engine`` decides) migrates the whole browser at once.
    """
    from softae.analysis.eis.engine import analyze_spectrum
    from softae.gui.eis_sigma import gui_cell, report_sigma

    report = analyze_spectrum(entry.eis, cell=gui_cell(L, t, w),
                              model_name=model_name)
    entry.fit = report.fit
    entry.geometry = (float(L), float(t), float(w))
    sigma = report_sigma(report)
    entry.sigma = float(sigma) if np.isfinite(sigma) else None
    return entry


class _SpectrumFitWorker(QThread):
    """Fit one spectrum off the UI thread; emit ``(entry, "")`` or ``(None, err)``."""

    done = Signal(object, str)

    def __init__(
        self,
        entry: EISEntry,
        model_name: str,
        L: float,
        t: float,
        w: float,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._entry = entry
        self._model_name = model_name
        self._L, self._t, self._w = L, t, w

    def run(self) -> None:  # noqa: D401 (QThread override)
        try:
            fit_entry(self._entry, self._model_name, self._L, self._t, self._w)
            self.done.emit(self._entry, "")
        except Exception as exc:
            self.done.emit(None, str(exc))


class EISDataSource(Protocol):
    """Protocol for objects that supply ``EISEntry`` collections."""

    def get_entries(self) -> list[EISEntry]: ...


# ---------------------------------------------------------------------------
# Concrete data sources
# ---------------------------------------------------------------------------


class ListEISSource:
    """Static list-backed data source."""

    def __init__(self, entries: list[EISEntry]) -> None:
        self._entries = entries

    def get_entries(self) -> list[EISEntry]:
        return self._entries


class DataStoreSource:
    """Loads EIS measurements and fits from a :class:`DataStore` instance."""

    def __init__(self, store: Any, run_id: str | None = None) -> None:
        from softae.core.data_store import DataStore  # noqa: F401 — deferred
        self._store = store
        self._run_id = run_id

    def get_entries(self) -> list[EISEntry]:
        rows = self._store.query_measurements(run_id=self._run_id)
        entries: list[EISEntry] = []
        for row in rows:
            if row["eis_file_path"] is None:
                continue
            # Resolve path: stored as relative to project_dir or absolute.
            eis_path_str: str = row["eis_file_path"]
            p = Path(eis_path_str)
            if not p.is_absolute():
                p = self._store.project_dir / p
            try:
                eis = EISResult.load(str(p))
            except Exception:
                continue

            fit: FitResult | None = None
            entry_sigma: float | None = None
            geom: tuple[float, float, float] | None = None
            fit_rows = self._store.query_fits(measurement_id=row["measurement_id"])
            if fit_rows:
                fit, entry_sigma, geom = _fitresult_from_fit_row(fit_rows[-1])

            run_id_val: str = row["run_id"]
            label = f"Ch{row['channel']:02d} — {run_id_val[:16]}"
            entries.append(
                EISEntry(
                    label=label,
                    eis=eis,
                    fit=fit,
                    sigma=entry_sigma,
                    run_id=run_id_val,
                    measurement_id=row["measurement_id"],
                    geometry=geom,
                )
            )
        return entries


class PollableSource(QObject):
    """Wraps any ``EISDataSource`` with a periodic polling timer.

    Emits :attr:`data_changed` on each timer tick so connected widgets
    know to call ``refresh()``.
    """

    data_changed = Signal()

    def __init__(
        self,
        base: EISDataSource,
        interval_ms: int = 5000,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._base = base
        self._timer = QTimer(self)
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self.data_changed.emit)

    def start(self) -> None:
        """Start the polling timer."""
        self._timer.start()

    def stop(self) -> None:
        """Stop the polling timer."""
        self._timer.stop()

    def get_entries(self) -> list[EISEntry]:
        return self._base.get_entries()


# ---------------------------------------------------------------------------
# Internal pane: Overview
# ---------------------------------------------------------------------------


class _OverviewPane(QScrollArea):
    """Scrollable grid of mini-Nyquist thumbnail canvases."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._cols: int = 4
        self._cell_figs: list[Figure] = []
        self._cell_canvases: list[_NonScrollCanvas] = []
        self._dirty: set[int] = set()
        self._entries: list[EISEntry] = []

        # Debounce resize events: only update cell sizes after the user
        # stops dragging the window edge (150 ms of no further resize events).
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(self._update_cell_size)

        self._content = QWidget()
        self._content.setAutoFillBackground(True)  # erase stale pixels before repaint
        self._grid = QGridLayout(self._content)
        self._grid.setContentsMargins(4, 4, 4, 4)
        self._grid.setSpacing(4)
        self.setWidget(self._content)
        self.setWidgetResizable(True)
        self.viewport().setAutoFillBackground(True)  # prevents bleed during scroll

    # ── public ──────────────────────────────────────────────────────────────

    def refresh(self, entries: list[EISEntry]) -> None:
        """Add canvases for any new entries, rebuild grid, and render dirty."""
        self._entries = entries
        n_existing = len(self._cell_figs)
        for i in range(n_existing, len(entries)):
            fig = Figure(figsize=(2.5, 2.0), dpi=80)
            canvas = _NonScrollCanvas(fig)
            canvas.setFixedSize(250, 200)  # initial size; updated by _update_cell_size
            self._cell_figs.append(fig)
            self._cell_canvases.append(canvas)
            self._dirty.add(i)

        self._rebuild_grid()
        self._update_cell_size()

    def set_columns(self, n: int) -> None:
        """Change column count and re-layout."""
        self._cols = n
        self._rebuild_grid()
        self._update_cell_size()

    def resizeEvent(self, event: Any) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        # Restart the debounce timer so rapid events collapse into one call.
        self._resize_timer.start(150)

    # ── private ─────────────────────────────────────────────────────────────

    def _update_cell_size(self) -> None:
        """Recompute and apply cell sizes based on current viewport width.

        Cells that changed size are re-rendered synchronously (``draw()``)
        so that Qt never shows a stale/stretched bitmap.
        """
        if not self._cell_canvases or not self._entries:
            return
        vp_w = self.viewport().width()
        if vp_w < 20:
            return
        spacing = 4 * max(0, self._cols - 1)
        cell_w = max(80, (vp_w - 8 - spacing) // self._cols)
        cell_h = int(cell_w * 0.8)
        resized: list[int] = []
        for i, (fig, canvas) in enumerate(zip(self._cell_figs, self._cell_canvases)):
            if canvas.width() != cell_w or canvas.height() != cell_h:
                canvas.setFixedSize(cell_w, cell_h)
                fig.set_size_inches(cell_w / fig.dpi, cell_h / fig.dpi)
                resized.append(i)
        if resized:
            self._dirty.update(resized)
            self._render_dirty(self._entries)
            # Synchronous draw eliminates stale-bitmap artifacts that
            # occur when the widget geometry changes before the next
            # event-loop idle tick processes draw_idle() requests.
            for i in resized:
                self._cell_canvases[i].draw()

    def _rebuild_grid(self) -> None:
        """Remove all canvases from grid and re-add in row-major order."""
        for canvas in self._cell_canvases:
            self._grid.removeWidget(canvas)
            canvas.hide()
        for i, canvas in enumerate(self._cell_canvases):
            row = i // self._cols
            col = i % self._cols
            self._grid.addWidget(canvas, row, col)
            canvas.show()

    def _render_dirty(self, entries: list[EISEntry]) -> None:
        """Render only the Figures marked dirty, then clear the dirty set."""
        for i in sorted(self._dirty):
            if i >= len(entries):
                continue
            entry = entries[i]
            fig = self._cell_figs[i]
            fig.clear()
            cell_h_px = self._cell_canvases[i].height()
            _fscale = max(0.7, min(2.0, cell_h_px / 200.0))
            fontsize_title = max(5, round(7 * _fscale))
            fontsize_sigma = max(6, round(8 * _fscale))
            ax_nyq    = fig.add_subplot(2, 1, 1)
            ax_footer = fig.add_subplot(2, 1, 2)

            ax_nyq.scatter(
                entry.eis.z_real,
                entry.eis.z_imag_neg,
                color=_Z_REAL_COLOR,
                s=15,
                marker="o",
                zorder=2,
            )

            if entry.fit is not None and entry.fit.success:
                z_fit = predict_fit_curve(entry.fit, entry.eis.frequency)
                if z_fit is not None:
                    ax_nyq.plot(
                        z_fit.real,
                        -z_fit.imag,
                        color=_FIT_COLOR,
                        lw=1.5,
                        zorder=4,
                    )

            ax_nyq.set_xlabel("Z\u2032 (\u03a9)", fontsize=max(5, round(5 * _fscale)))
            ax_nyq.set_ylabel("\u2212Z\u2033 (\u03a9)", fontsize=max(5, round(5 * _fscale)))
            ax_nyq.tick_params(axis="both", labelsize=max(4, round(5 * _fscale)))
            ax_nyq.set_aspect("equal", "datalim")
            ax_nyq.set_title(entry.label[:25], fontsize=fontsize_title)

            ax_footer.set_axis_off()
            sigma_txt = (
                f"\u03c3 = {entry.sigma:.3e} S/cm"
                if entry.sigma is not None
                else "\u03c3 = \u2014"
            )
            ax_footer.text(
                0.5, 0.5, sigma_txt,
                ha="center", va="center",
                transform=ax_footer.transAxes,
                fontsize=fontsize_sigma,
            )

            fig.tight_layout(pad=0.3)
            self._cell_canvases[i].draw()  # synchronous — never leaves a stale backbuffer

        self._dirty.clear()


# ---------------------------------------------------------------------------
# Internal pane: Inspection
# ---------------------------------------------------------------------------


class _InspectionPane(QWidget):
    """Side-by-side list + 4-panel EIS plot + fit metrics table."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        on_fit_saved: Callable[[EISEntry, float, float, float], None] | None = None,
        default_geometry: tuple[float, float, float] = DEFAULT_GEOMETRY,
    ) -> None:
        super().__init__(parent)
        self._entries: list[EISEntry] = []
        # Invoked after a successful in-browser re-fit so the owner can persist
        # it (e.g. DataStore.record_fit). Signature: (entry, L, t, w) in cm.
        self.on_fit_saved = on_fit_saved
        self._default_geometry = default_geometry
        self._fit_worker: _SpectrumFitWorker | None = None

        # Permanent display figure — avoids canvas ownership issues.
        self._fig = Figure(figsize=(10, 7), dpi=80)
        self._canvas = _NonScrollCanvas(self._fig)

        # Entry list
        self._list = QListWidget()
        self._list.currentRowChanged.connect(self._show_entry)

        # Fit metrics table inside a group box
        self._metrics_table = QTableWidget(0, 2)
        self._metrics_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        self._metrics_table.horizontalHeader().setStretchLastSection(True)
        self._metrics_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )

        metrics_box = QGroupBox("Fit Metrics")
        metrics_layout = QVBoxLayout(metrics_box)
        metrics_layout.addWidget(self._metrics_table)

        # Per-spectrum fit controls: model + electrode geometry (may differ per
        # sample), fitted on demand and (if wired) persisted via on_fit_saved.
        fit_box = self._build_fit_controls()

        # Right panel: canvas + metrics + fit controls
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self._canvas, stretch=1)
        right_layout.addWidget(metrics_box)
        right_layout.addWidget(fit_box)

        # Splitter: list (1) | right panel (4)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._list)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 4)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

    # ── public ──────────────────────────────────────────────────────────────

    def refresh(self, entries: list[EISEntry]) -> None:
        """Sync list widget to *entries* while preserving the current selection."""
        cur = self._list.currentRow()
        self._entries = entries

        if len(entries) < self._list.count():
            # Entries were removed — full repopulate
            self._list.clear()
            for e in entries:
                self._list.addItem(e.label)
        else:
            # Append only new items
            for i in range(self._list.count(), len(entries)):
                self._list.addItem(entries[i].label)

        if cur >= 0 and cur < len(entries):
            self._list.setCurrentRow(cur)
        elif len(entries) > 0 and self._list.currentRow() < 0:
            self._list.setCurrentRow(0)

    # ── private ─────────────────────────────────────────────────────────────

    def _show_entry(self, index: int) -> None:
        if index < 0 or index >= len(self._entries):
            self._fig.clear()
            ax = self._fig.add_subplot(111)
            ax.text(
                0.5, 0.5, "No entry selected",
                ha="center", va="center",
                transform=ax.transAxes,
                color="#888888", fontsize=12,
            )
            ax.set_axis_off()
            self._canvas.draw_idle()
            self._metrics_table.setRowCount(0)
            self._set_fit_controls_enabled(False)
            return

        entry = self._entries[index]
        self._draw_entry(entry)
        self._update_metrics_table(entry)
        self._sync_fit_controls(entry)

    def _draw_entry(self, entry: EISEntry) -> None:
        """Inline reproduction of plot_eis_fit writing to self._fig."""
        from softae.analysis.circuit_fitting import compute_fit_residuals

        freq   = entry.eis.frequency
        z_real = entry.eis.z_real
        z_imag = entry.eis.z_imag_neg

        # Build fit complex array if possible
        z_fit_complex: np.ndarray | None = None
        if entry.fit is not None and entry.fit.success:
            z_fit_complex = predict_fit_curve(entry.fit, freq)

        self._fig.clear()
        self._fig.set_layout_engine("constrained")
        _suptitle = entry.label
        if entry.fit is not None:
            _suptitle += f"  [{entry.fit.model_name}]"
            if not entry.fit.success:
                _suptitle += "  \u26a0 fit failed"
        self._fig.suptitle(_suptitle, fontsize=11, fontweight="bold")
        gs = self._fig.add_gridspec(
            2, 2, hspace=0.4, wspace=0.35, height_ratios=[2, 1]
        )
        ax_nyq      = self._fig.add_subplot(gs[0, 0])
        ax_bode     = self._fig.add_subplot(gs[0, 1])
        ax_res_real = self._fig.add_subplot(gs[1, 0])
        ax_res_imag = self._fig.add_subplot(gs[1, 1])

        # ── Nyquist ──────────────────────────────────────────────────────
        ax_nyq.scatter(
            z_real, z_imag,
            color=_Z_REAL_COLOR, s=30, zorder=2, marker="o", label="Measured"
        )
        if z_fit_complex is not None:
            ax_nyq.plot(
                z_fit_complex.real, -z_fit_complex.imag,
                color=_FIT_COLOR, linewidth=2, zorder=4, label="Model fit"
            )
        ax_nyq.set_xlabel("Z\u2032 (\u03a9)")
        ax_nyq.set_ylabel("\u2212Z\u2033 (\u03a9)")
        ax_nyq.set_title("Nyquist")
        ax_nyq.set_aspect("equal", adjustable="datalim")
        ax_nyq.grid(True, linestyle="--", alpha=0.5)
        ax_nyq.legend(fontsize=9)

        # ── Bode ─────────────────────────────────────────────────────────
        ax_bode_phase = ax_bode.twinx()
        ax_bode.scatter(
            freq, z_real, color=_Z_REAL_COLOR, s=20, marker="o",
            zorder=2, label="Z\u2032 meas."
        )
        ax_bode.scatter(
            freq, z_imag, color=_Z_IMAG_COLOR, s=20, marker="s",
            zorder=2, label="\u2212Z\u2033 meas."
        )
        ax_bode.set_xscale("log")
        ax_bode.set_yscale("log")
        ax_bode.set_xlabel("Frequency (Hz)")
        ax_bode.set_ylabel("Z\u2032, \u2212Z\u2033 (\u03a9)")

        phase_measured = -entry.eis.phase
        ax_bode_phase.scatter(
            freq, phase_measured, color=_PHASE_COLOR, s=15,
            marker="^", zorder=2, label="Phase meas."
        )
        ax_bode_phase.set_ylabel("\u2212Phase (\u00b0)", color=_PHASE_COLOR)
        ax_bode_phase.tick_params(axis="y", labelcolor=_PHASE_COLOR)

        if z_fit_complex is not None:
            z_fit_real     =  np.real(z_fit_complex)
            z_fit_imag_neg = -np.imag(z_fit_complex)
            z_fit_phase    = -np.angle(z_fit_complex, deg=True)
            ax_bode.plot(freq, z_fit_real,     color=_FIT_COLOR, lw=2, ls="-",  zorder=4, label="Z\u2032 fit")
            ax_bode.plot(freq, z_fit_imag_neg, color=_FIT_COLOR, lw=2, ls="--", zorder=4, label="\u2212Z\u2033 fit")
            ax_bode_phase.plot(freq, z_fit_phase, color=_FIT_COLOR, lw=2, ls=":", zorder=4, label="Phase fit")

        ax_bode.set_title("Bode")
        ax_bode.grid(True, which="both", linestyle="--", alpha=0.4)
        lines1, labels1 = ax_bode.get_legend_handles_labels()
        lines2, labels2 = ax_bode_phase.get_legend_handles_labels()
        ax_bode.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

        # ── Residuals ────────────────────────────────────────────────────
        if z_fit_complex is not None:
            resid_real, resid_imag = compute_fit_residuals(entry.eis, z_fit_complex)

            ax_res_real.axhline(0, color="red", linewidth=1.0, zorder=1)
            ml_r, sl_r, _ = ax_res_real.stem(
                freq, resid_real,
                linefmt=_RESID_COLOR, markerfmt="o", basefmt=" "
            )
            ml_r.set_color(_RESID_COLOR)
            ml_r.set_markersize(4)
            try:
                sl_r.set_linewidth(0.8)
            except AttributeError:
                sl_r.set_linewidths(0.8)
            ax_res_real.set_xscale("log")
            ax_res_real.set_xlabel("Frequency (Hz)")
            ax_res_real.set_ylabel("Residual (%)")
            ax_res_real.set_title("Z' Residuals")
            ax_res_real.grid(True, linestyle="--", alpha=0.4)

            ax_res_imag.axhline(0, color="red", linewidth=1.0, zorder=1)
            ml_i, sl_i, _ = ax_res_imag.stem(
                freq, resid_imag,
                linefmt=_RESID_COLOR, markerfmt="o", basefmt=" "
            )
            ml_i.set_color(_RESID_COLOR)
            ml_i.set_markersize(4)
            try:
                sl_i.set_linewidth(0.8)
            except AttributeError:
                sl_i.set_linewidths(0.8)
            ax_res_imag.set_xscale("log")
            ax_res_imag.set_xlabel("Frequency (Hz)")
            ax_res_imag.set_ylabel("Residual (%)")
            ax_res_imag.set_title("-Z'' Residuals")
            ax_res_imag.grid(True, linestyle="--", alpha=0.4)
        else:
            for ax, ttl in (
                (ax_res_real, "Z' Residuals"),
                (ax_res_imag, "-Z'' Residuals"),
            ):
                ax.text(
                    0.5, 0.5, "No fit available",
                    ha="center", va="center",
                    transform=ax.transAxes,
                    color="#888888", fontsize=10,
                )
                ax.set_title(ttl)
                ax.set_axis_off()

        self._canvas.draw_idle()

    def _update_metrics_table(self, entry: EISEntry) -> None:
        self._metrics_table.setRowCount(0)
        if entry.fit is None:
            self._metrics_table.setRowCount(1)
            self._metrics_table.setItem(0, 0, QTableWidgetItem("No fit"))
            self._metrics_table.setItem(0, 1, QTableWidgetItem("\u2014"))
            return

        rows = [
            ("Model",     entry.fit.model_name),
            ("R0 (\u03a9)",   f"{entry.fit.R0:.4g}"),
            ("R1 (\u03a9)",   f"{entry.fit.R1:.4g}"),
            ("\u03c3 (S/cm)", f"{entry.sigma:.4e}" if entry.sigma is not None else "\u2014"),
            ("Success",   str(entry.fit.success)),
            ("Error",     entry.fit.error_msg or "\u2014"),
        ]
        self._metrics_table.setRowCount(len(rows))
        for r, (param, val) in enumerate(rows):
            self._metrics_table.setItem(r, 0, QTableWidgetItem(param))
            self._metrics_table.setItem(r, 1, QTableWidgetItem(val))

    # ── Per-spectrum fit controls ────────────────────────────────────────────

    def _build_fit_controls(self) -> QGroupBox:
        """Model + electrode-geometry inputs and a Fit button for one spectrum."""
        from softae.analysis.circuit_fitting import CIRCUIT_MODELS

        box = QGroupBox("Fit this spectrum")
        lay = QVBoxLayout(box)

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model:"))
        self._combo_model = QComboBox()
        self._combo_model.addItems(list(CIRCUIT_MODELS.keys()))
        model_row.addWidget(self._combo_model)
        model_row.addStretch()
        lay.addLayout(model_row)

        def _geom_spin(value: float) -> QDoubleSpinBox:
            s = QDoubleSpinBox()
            s.setRange(0.0001, 1000.0)
            s.setDecimals(6)  # single-micron (0.0001 cm) resolution
            s.setSingleStep(0.01)
            s.setValue(value)
            return s

        L0, t0, w0 = self._default_geometry
        geom_row = QHBoxLayout()
        geom_row.addWidget(QLabel("L (cm):"))
        self._spin_L = _geom_spin(L0)
        geom_row.addWidget(self._spin_L)
        geom_row.addWidget(QLabel("t (cm):"))
        self._spin_t = _geom_spin(t0)
        geom_row.addWidget(self._spin_t)
        geom_row.addWidget(QLabel("w (cm):"))
        self._spin_w = _geom_spin(w0)
        geom_row.addWidget(self._spin_w)
        geom_row.addStretch()
        lay.addLayout(geom_row)

        action_row = QHBoxLayout()
        self._btn_fit = QPushButton("Fit")
        self._btn_fit.setToolTip(
            "Fit the selected spectrum with the chosen model and electrode "
            "geometry, then (if backed by a DataStore) save the result."
        )
        self._btn_fit.clicked.connect(self._on_fit_clicked)
        self._btn_fit.setEnabled(False)
        action_row.addWidget(self._btn_fit)
        self._fit_status = QLabel("")
        self._fit_status.setStyleSheet("color: gray;")
        action_row.addWidget(self._fit_status)
        action_row.addStretch()
        lay.addLayout(action_row)
        return box

    def _set_fit_controls_enabled(self, enabled: bool) -> None:
        # Never re-enable Fit mid-run (a worker owns it until it finishes).
        self._btn_fit.setEnabled(enabled and self._fit_worker is None)

    def _current_entry(self) -> EISEntry | None:
        idx = self._list.currentRow()
        return self._entries[idx] if 0 <= idx < len(self._entries) else None

    def _sync_fit_controls(self, entry: EISEntry) -> None:
        """Prefill model + geometry from the entry (its prior fit, else default)."""
        if entry.fit is not None:
            midx = self._combo_model.findText(entry.fit.model_name)
            if midx >= 0:
                self._combo_model.setCurrentIndex(midx)
        L, t, w = entry.geometry or self._default_geometry
        self._spin_L.setValue(L)
        self._spin_t.setValue(t)
        self._spin_w.setValue(w)
        self._fit_status.setText("")
        self._set_fit_controls_enabled(True)

    def _on_fit_clicked(self) -> None:
        entry = self._current_entry()
        if entry is None or self._fit_worker is not None:
            return
        model = self._combo_model.currentText()
        L, t, w = self._spin_L.value(), self._spin_t.value(), self._spin_w.value()
        self._btn_fit.setEnabled(False)
        self._fit_status.setText("Fitting…")
        worker = _SpectrumFitWorker(entry, model, L, t, w, self)
        worker.done.connect(self._on_fit_done)
        self._fit_worker = worker
        worker.start()

    def _on_fit_done(self, entry: EISEntry | None, error: str) -> None:
        self._fit_worker = None
        self._btn_fit.setEnabled(self._current_entry() is not None)
        if entry is None:
            self._fit_status.setText(f"Fit failed: {error[:80]}")
            return

        # Redraw only if this entry is still the one on screen.
        if entry is self._current_entry():
            self._draw_entry(entry)
            self._update_metrics_table(entry)

        ok = entry.fit is not None and entry.fit.success
        if ok and entry.sigma is not None:
            self._fit_status.setText(f"Fit ok · σ = {entry.sigma:.3e} S/cm")
        elif ok:
            self._fit_status.setText("Fit ok (σ unavailable)")
        else:
            self._fit_status.setText("Fit did not converge")

        if callable(self.on_fit_saved) and entry.fit is not None and entry.geometry:
            L, t, w = entry.geometry
            try:
                self.on_fit_saved(entry, L, t, w)
            except Exception:
                self._fit_status.setText(
                    self._fit_status.text() + "  (save failed)"
                )


# ---------------------------------------------------------------------------
# Internal pane: Conductivity
# ---------------------------------------------------------------------------


class _ConductivityPane(QWidget):
    """Log-scale conductivity trend plot with point picker."""

    def __init__(
        self,
        on_pick_jump: Callable[[int], None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._entries: list[EISEntry] = []
        self._on_pick_jump = on_pick_jump

        self._fig = Figure(figsize=(8, 5), dpi=80)
        self._canvas = _NonScrollCanvas(self._fig)
        self._canvas.mpl_connect("pick_event", self._on_pick)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas)

    def refresh(self, entries: list[EISEntry]) -> None:
        self._entries = entries
        self._redraw()

    def _redraw(self) -> None:
        self._fig.clear()
        self._fig.set_layout_engine("constrained")
        ax = self._fig.add_subplot(111)
        ax.set_yscale("log")
        ax.set_xlabel("Sample index")
        ax.set_ylabel("\u03c3 (S/cm)")
        ax.set_title("Estimated Ionic Conductivity")

        # Build run_id → color mapping
        seen_ids: list[str | None] = []
        for e in self._entries:
            if e.run_id not in seen_ids:
                seen_ids.append(e.run_id)
        # Extend palette with tab10 colours if needed
        extended_palette = _WONG_CYCLE + [f"C{i}" for i in range(len(_WONG_CYCLE), 20)]
        color_map: dict[str | None, str] = {
            rid: extended_palette[k % len(extended_palette)]
            for k, rid in enumerate(seen_ids)
        }

        for i, entry in enumerate(self._entries):
            color = color_map[entry.run_id]
            if entry.sigma is not None:
                ax.semilogy(
                    i, entry.sigma, "o",
                    color=color, picker=5, gid=str(i), markersize=7,
                )
            else:
                ax.plot(
                    i, 1e-10, "o",
                    color="#aaaaaa", markerfacecolor="none", markersize=7,
                )

        # Legend — one proxy artist per run_id
        import matplotlib.patches as mpatches
        legend_handles = [
            mpatches.Patch(
                color=color_map[rid],
                label=rid if rid is not None else "unknown",
            )
            for rid in seen_ids
        ]
        if legend_handles:
            ax.legend(handles=legend_handles, fontsize=8)

        if self._entries:
            ax.set_xticks(range(len(self._entries)))
            ax.set_xticklabels(
                [e.label[:12] for e in self._entries],
                rotation=45, ha="right", fontsize=7,
            )

        self._canvas.draw_idle()

    def _on_pick(self, event: Any) -> None:
        try:
            idx = int(event.artist.get_gid())
            self._on_pick_jump(idx)
        except (ValueError, TypeError, AttributeError):
            pass


# ---------------------------------------------------------------------------
# Grid dialog
# ---------------------------------------------------------------------------


class _GridDialog(QDialog):
    """Simple dialog to adjust the column count of the Overview grid."""

    def __init__(
        self, overview: _OverviewPane, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Adjust Grid")
        self._overview = overview

        self._spinbox = QSpinBox()
        self._spinbox.setRange(1, 12)
        self._spinbox.setValue(overview._cols)

        form = QFormLayout()
        form.addRow("Plots per row:", self._spinbox)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)

        main_layout = QVBoxLayout(self)
        main_layout.addLayout(form)
        main_layout.addWidget(buttons)

    def _on_accept(self) -> None:
        self._overview.set_columns(self._spinbox.value())
        self.accept()


# ---------------------------------------------------------------------------
# Main widget
# ---------------------------------------------------------------------------


class EISVisualizerWidget(QWidget):
    """Three-pane EIS browser: Overview · Inspection · Conductivity."""

    def __init__(
        self,
        source: EISDataSource,
        *,
        geometry: tuple[float, float, float] = DEFAULT_GEOMETRY,
        poll_interval_ms: int = 0,
        on_fit_saved: Callable[[EISEntry, float, float, float], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._source = source
        self._geometry = geometry
        self._entries: list[EISEntry] = []
        self._poll_timer: QTimer | None = None

        # Build the three panes
        self._overview     = _OverviewPane()
        self._inspection   = _InspectionPane(
            on_fit_saved=on_fit_saved, default_geometry=geometry
        )
        self._conductivity = _ConductivityPane(
            on_pick_jump=self._jump_to_inspection
        )

        # Stacked widget
        self._stack = QStackedWidget()
        self._stack.addWidget(self._overview)      # index 0
        self._stack.addWidget(self._inspection)    # index 1
        self._stack.addWidget(self._conductivity)  # index 2

        # Toolbar
        toolbar = QToolBar()
        toolbar.setMovable(False)
        action_group = QActionGroup(self)
        action_group.setExclusive(True)

        self._act_overview = QAction("Overview", self)
        self._act_overview.setCheckable(True)
        self._act_overview.setChecked(True)
        action_group.addAction(self._act_overview)
        toolbar.addAction(self._act_overview)
        self._act_overview.toggled.connect(
            lambda v: v and self._set_mode(0)
        )

        self._act_inspection = QAction("Inspection", self)
        self._act_inspection.setCheckable(True)
        action_group.addAction(self._act_inspection)
        toolbar.addAction(self._act_inspection)
        self._act_inspection.toggled.connect(
            lambda v: v and self._set_mode(1)
        )

        self._act_conductivity = QAction("Conductivity", self)
        self._act_conductivity.setCheckable(True)
        action_group.addAction(self._act_conductivity)
        toolbar.addAction(self._act_conductivity)
        self._act_conductivity.toggled.connect(
            lambda v: v and self._set_mode(2)
        )

        toolbar.addSeparator()

        btn_grid = QPushButton("Adjust Grid\u2026")
        btn_grid.clicked.connect(self._open_grid_dialog)
        toolbar.addWidget(btn_grid)

        btn_refresh = QPushButton("\u21ba Refresh")
        btn_refresh.clicked.connect(self.refresh)
        toolbar.addWidget(btn_refresh)

        # Status bar
        self._status_label = QLabel("")

        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_layout.addWidget(toolbar)
        main_layout.addWidget(self._stack, stretch=1)
        main_layout.addWidget(self._status_label)

        if poll_interval_ms > 0:
            self.start_polling(poll_interval_ms)

        self.refresh()

    # ── public API ───────────────────────────────────────────────────────────

    def set_source(self, source: EISDataSource) -> None:
        """Replace the data source and refresh."""
        self._source = source
        self.refresh()

    def set_geometry(self, L: float, t: float, w: float) -> None:
        """Update L/t/w geometry and refresh the overview."""
        self._geometry = (L, t, w)
        self._overview.refresh(self._entries)

    def refresh(self) -> None:
        """Re-query the source and update all three panes."""
        try:
            self._entries = self._source.get_entries()
        except Exception:
            self._entries = []

        self._overview.refresh(self._entries)
        self._inspection.refresh(self._entries)
        self._conductivity.refresh(self._entries)

        now = datetime.now().strftime("%H:%M:%S")
        self._status_label.setText(
            f"{len(self._entries)} entries \u00b7 refreshed {now}"
        )

    def start_polling(self, interval_ms: int) -> None:
        """Create (or reuse) an internal QTimer connected to refresh()."""
        if self._poll_timer is None:
            self._poll_timer = QTimer(self)
            self._poll_timer.timeout.connect(self.refresh)
        self._poll_timer.setInterval(interval_ms)
        self._poll_timer.start()

    def stop_polling(self) -> None:
        """Stop and release the polling timer."""
        if self._poll_timer is not None:
            self._poll_timer.stop()
            self._poll_timer = None

    # ── private ─────────────────────────────────────────────────────────────

    def _set_mode(self, index: int) -> None:
        self._stack.setCurrentIndex(index)
        if index == 0:
            self._overview.refresh(self._entries)
        elif index == 1:
            self._inspection.refresh(self._entries)
        elif index == 2:
            self._conductivity.refresh(self._entries)

    def _open_grid_dialog(self) -> None:
        dlg = _GridDialog(self._overview, parent=self)
        dlg.exec()

    def _jump_to_inspection(self, idx: int) -> None:
        """Switch to Inspection pane and select entry at *idx*."""
        self._act_inspection.setChecked(True)
        self._set_mode(1)
        if 0 <= idx < len(self._entries):
            self._inspection._list.setCurrentRow(idx)


# ---------------------------------------------------------------------------
# Standalone window
# ---------------------------------------------------------------------------


class EISVisualizerWindow(QMainWindow):
    """EIS visualizer as a detached top-level window.

    Wraps :class:`EISVisualizerWidget` in a ``QMainWindow`` — useful for
    launching a separate browser from any tab or from a notebook/script.

    Usage
    -----
    From a tab::

        win = EISVisualizerWindow(ListEISSource(entries))
        win.show()

    As a blocking script launcher::

        EISVisualizerWindow.open(ListEISSource(entries))
    """

    def __init__(
        self,
        source: EISDataSource,
        *,
        geometry: tuple[float, float, float] = DEFAULT_GEOMETRY,
        poll_interval_ms: int = 0,
        on_fit_saved: Callable[[EISEntry, float, float, float], None] | None = None,
        title: str = "EIS Visualizer",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1280, 840)
        self._viewer = EISVisualizerWidget(
            source,
            geometry=geometry,
            poll_interval_ms=poll_interval_ms,
            on_fit_saved=on_fit_saved,
        )
        self.setCentralWidget(self._viewer)

    # Convenience delegation to the inner widget
    def set_source(self, source: EISDataSource) -> None:
        self._viewer.set_source(source)

    def refresh(self) -> None:
        self._viewer.refresh()

    def start_polling(self, interval_ms: int) -> None:
        self._viewer.start_polling(interval_ms)

    def stop_polling(self) -> None:
        self._viewer.stop_polling()

    @classmethod
    def open(cls, source: EISDataSource, **kwargs) -> "EISVisualizerWindow":
        """Create, show, and (if no existing QApplication) run the event loop.

        Intended for use in scripts and notebooks where no Qt event loop is
        already running::

            EISVisualizerWindow.open(ListEISSource(entries))
        """
        import sys

        from PySide6.QtWidgets import QApplication

        app = QApplication.instance()
        owns_app = app is None
        if owns_app:
            app = QApplication(sys.argv)
        win = cls(source, **kwargs)
        win.show()
        if owns_app:
            app.exec()
        return win


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------

__all__ = [
    "EISEntry",
    "EISDataSource",
    "ListEISSource",
    "DataStoreSource",
    "PollableSource",
    "EISVisualizerWidget",
    "EISVisualizerWindow",
]
