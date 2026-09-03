"""Tab 8: EIS Analysis & Data Management.

Provides file loading, interactive Nyquist / Bode plotting, equivalent-
circuit fitting, conductivity calculation, and SQLite-backed results
storage.

The tab imports :mod:`softae.analysis.eis_data` and
:mod:`softae.analysis.circuit_fitting` for the actual computations.
"""

from __future__ import annotations

import csv
import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from softae.analysis.arrhenius import ArrheniusFitter, ArrheniusResult
from softae.analysis.thermal import make_fitter
from softae.analysis.circuit_fitting import CIRCUIT_MODELS, FitResult
from softae.analysis.eis.engine import analyze_spectrum
from softae.analysis.eis_data import EISResult
from softae.gui.eis_sigma import cell_sigma, gui_cell, report_sigma
from softae.gui.widgets.copyable_table import (
    CopyableTableWidget,
    PasteableTableWidget,
)

if TYPE_CHECKING:
    from softae.server.manager import InstrumentManager

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure

    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

logger = structlog.get_logger(__name__)

# Default SQLite database path (next to the running script)
_DEFAULT_DB = Path("softae_results.db")


def _widget_alive(w: "QWidget") -> bool:
    """Return True only if the C++ QWidget backing *w* has not been deleted.

    ``WA_DeleteOnClose`` windows call their C++ destructor when closed, leaving
    a dangling Python wrapper.  Calling any method on such a wrapper raises
    ``RuntimeError: Internal C++ object already deleted``.
    """
    try:
        return w.isVisible()
    except RuntimeError:
        return False


# ---------------------------------------------------------------------------
# SQLite helper
# ---------------------------------------------------------------------------

def _ensure_db(db_path: Path) -> sqlite3.Connection:
    """Open (or create) the results database and ensure tables exist."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS eis_results (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT    NOT NULL,
            channel     INTEGER NOT NULL,
            study_name  TEXT    DEFAULT '',
            model_name  TEXT    DEFAULT '',
            R0          REAL,
            R1          REAL,
            sigma       REAL,
            npts        INTEGER,
            f_min       REAL,
            f_max       REAL,
            file_path   TEXT    DEFAULT '',
            params_json TEXT    DEFAULT '{}',
            fit_success INTEGER DEFAULT 1,
            fit_error   TEXT    DEFAULT ''
        )
    """)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Background fit workers
# ---------------------------------------------------------------------------

class _FitAllWorker(QThread):
    """Runs EIS circuit fitting for all loaded measurements off the main thread."""

    finished: Signal = Signal(object)  # list of (FitResult, sigma, channel, filepath)

    def __init__(
        self, loaded: list, model_name: str, L: float, t: float, w: float, parent=None
    ) -> None:
        super().__init__(parent)
        self._loaded = list(loaded)  # snapshot — don't hold a reference to the live list
        self._model_name = model_name
        self._L, self._t, self._w = L, t, w

    def run(self) -> None:
        results = []
        cell = gui_cell(self._L, self._t, self._w)
        for eis in self._loaded:
            # ``engine`` is deliberately unset: ``[eis] engine`` governs this tab.
            report = analyze_spectrum(eis, cell=cell, model_name=self._model_name)
            results.append((report.fit, report_sigma(report), eis.channel,
                            eis.raw_file_path or ""))
        self.finished.emit(results)


class _ArrhFitWorker(QThread):
    """Runs circuit fitting + Arrhenius analysis off the main thread.

    Groups fits by ``(channel, rh_sp)`` so that measurements taken at
    different RH setpoints produce separate Arrhenius curves rather than
    being mixed into a single (nonsensical) fit per channel.
    """

    # fit_by_index dict,
    # sigma_by_index dict {orig_idx: σ the fit consumed} — see ``run``,
    # arrh_keyed list of (ch, rh_sp, ArrheniusResult),
    # rh_results dict {rh_sp: [ArrheniusResult, ...]} (only non-nan RH keys)
    finished: Signal = Signal(object, object, object, object)
    progress: Signal = Signal(int, int, float)  # current, total, elapsed_s

    def __init__(
        self,
        pairs: list,         # list of (orig_idx, EISResult, T_C, rh_sp_or_none)
        model_name: str,
        L: float,
        t: float,
        w: float,
        thermal_model: str = "arrhenius",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._pairs = list(pairs)
        self._model_name = model_name
        self._L, self._t, self._w = L, t, w
        self._thermal_model = thermal_model

    def run(self) -> None:
        import math as _math
        import time as _time
        t_start = _time.monotonic()
        total = len(self._pairs)
        # Group by (channel, rh_norm).  rh_norm is a rounded float, or None for
        # "no RH" — NOT nan: a fresh float("nan") is a distinct object every
        # iteration (nan != nan), which would scatter every no-RH measurement
        # into its own single-point group and make all fits fail.
        group_data: dict[tuple, tuple[list, list]] = {}
        fit_by_index: dict[int, object] = {}
        # σ is cached per row alongside the fit because the params panel must show
        # the number *this* fit consumed. The cell is built once, here, from the spin
        # values as they stood when "Fit Arrhenius" was pressed; recomputing σ later
        # from the live spins against this stale fit produced a second, different
        # number for the same row with nothing on screen saying which was which.
        sigma_by_index: dict[int, float] = {}
        cell = gui_cell(self._L, self._t, self._w)
        for step, (orig_idx, eis, T_C, rh_sp) in enumerate(self._pairs):
            # ``engine`` unset — the Arrhenius sub-tab follows ``[eis] engine`` too.
            report = analyze_spectrum(eis, cell=cell, model_name=self._model_name)
            fit_by_index[orig_idx] = report.fit
            sigma = report_sigma(report)
            sigma_by_index[orig_idx] = sigma
            rh_norm = (
                round(float(rh_sp), 6)
                if (rh_sp is not None and not _math.isnan(float(rh_sp)))
                else None
            )
            key = (eis.channel, rh_norm)
            if key not in group_data:
                group_data[key] = ([], [])
            group_data[key][0].append(T_C)
            group_data[key][1].append(sigma)
            self.progress.emit(step + 1, total, _time.monotonic() - t_start)

        fitter = make_fitter(self._thermal_model)
        arrh_keyed: list[tuple] = []            # (ch, rh_sp, ThermalResult)
        rh_results: dict[float, list] = {}      # only real-RH keys, for 3D plot
        for (ch, rh_norm), (temps, sigmas) in sorted(
            group_data.items(),
            key=lambda item: (item[0][0], -1.0 if item[0][1] is None else item[0][1]),
        ):
            # Downstream (table/plot/3D) expects nan for "no RH"; convert here so
            # those isnan-based code paths stay unchanged.
            rh_key = float("nan") if rh_norm is None else float(rh_norm)
            rh_label = "norh" if rh_norm is None else f"rh{rh_norm:.0f}"
            res = fitter.fit(temps, sigmas, channel=ch, run_id=f"posthoc_{rh_label}")
            arrh_keyed.append((ch, rh_key, res))
            if rh_norm is not None:
                rh_results.setdefault(rh_key, []).append(res)

        self.finished.emit(fit_by_index, sigma_by_index, arrh_keyed, rh_results)


def _excel_column_name(index: int) -> str:
    """1-based spreadsheet-style label: 1→A, 26→Z, 27→AA, 703→AAA.

    Used to give each results-table row a short, unique sample identifier
    (channel alone isn't unique — the same channel can be measured repeatedly).
    """
    if index < 1:
        return ""
    name = ""
    n = index
    while n > 0:
        n, rem = divmod(n - 1, 26)
        name = chr(ord("A") + rem) + name
    return name


def _sample_id_item(index: int) -> "QTableWidgetItem":
    """A read-only, selectable table cell holding the sample ID for *index*."""
    item = QTableWidgetItem(_excel_column_name(index))
    item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
    return item


# Fit & Export results-table column indices. Thickness (t) is per-row/editable —
# electrode L and w are board-defined (global), but film thickness varies per sample.
(_RCOL_CHECK, _RCOL_SAMPLE, _RCOL_CHANNEL, _RCOL_FILE, _RCOL_MODEL,
 _RCOL_R0, _RCOL_R1, _RCOL_T, _RCOL_SIGMA, _RCOL_STATUS, _RCOL_GATE,
 _RCOL_ERROR) = range(12)
_RCOL_COUNT = 12


def _ro_item(text: str) -> "QTableWidgetItem":
    """A read-only, selectable results cell (data columns aren't hand-edited)."""
    item = QTableWidgetItem(text)
    item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
    return item


def _gate_item(fit_result: Any) -> "QTableWidgetItem":
    """The Gate cell: what the admission gates did to this spectrum.

    ``—`` on the legacy engine, which runs no gates. Otherwise ``pass``,
    ``N dropped``, ``N unchecked``, or ``REJECTED: <gate>``, with the full
    per-gate log in the tooltip — R17's "no point removed without a named gate
    and a reason", carried all the way to where someone actually looks.

    The rendering is
    :meth:`softae.analysis.eis.report.QualityReport.gate_summary`'s, adopted
    token-for-token so the two surfaces cannot disagree. In particular
    ``checked`` is read with **no default**: ``e.get("checked") is False`` only.
    An absent ``checked`` means the entry predates the field and is rendered by
    ``passed`` — deliberately the permissive branch, because every stored
    ``gate_log_json`` predates ``checked``, and it is the same ruling
    ``eis_validate_records.passed_gates()`` makes. Drops and unchecked gates are
    reported *together*: they are independent facts, and neither shadows the
    other.

    **The empty-log guard differs from ``gate_summary``'s by necessity, not by
    choice.** That method returns ``—`` on ``self.engine != "gated"``;
    :class:`~softae.analysis.eis.circuit_fitting.FitResult` carries no ``engine``
    field, so the same guard is not available here. The two are equivalent today
    only because ``engine.py`` hardcodes ``gate_log=()`` on the legacy path —
    an upstream guarantee, not a local one. If the legacy path ever starts
    emitting a log, this cell renders it and ``gate_summary`` still would not.
    """
    log = list(getattr(fit_result, "gate_log", None) or [])
    if not log:
        return _ro_item("—")

    rejected = next(
        (e for e in log
         if not e.get("passed", True)
         and e.get("severity") in ("block_spectrum", "block_session")),
        None,
    )
    dropped = sum(int(e.get("n_dropped", 0) or 0) for e in log)
    unchecked = sum(1 for e in log if e.get("checked") is False)

    if rejected is not None:
        item = _ro_item(f"REJECTED: {rejected.get('gate', '?')}")
        item.setForeground(Qt.GlobalColor.red)
    else:
        parts = []
        if dropped:
            parts.append(f"{dropped} dropped")
        if unchecked:
            parts.append(f"{unchecked} unchecked")
        item = _ro_item(", ".join(parts) if parts else "pass")

    item.setToolTip("\n".join(
        f"{_gate_mark(e)}  {e.get('gate', '?')}: {e.get('detail', '')}"
        for e in log
    ))
    return item


def _gate_mark(entry: dict) -> str:
    """One gate entry's tooltip mark — three states, not two.

    ``unchecked`` exists because ``passed=True`` on an entry that could not
    evaluate its criterion is a *placeholder*, not a verdict, and rendering it
    ``pass`` is the exact conflation ``checked`` was added to remove. Absent
    ``checked`` still renders by ``passed`` — see :func:`_gate_item`.
    """
    if entry.get("checked") is False:
        return "unchecked"
    return "pass" if entry.get("passed", True) else "FAIL"


def _editable_item(text: str) -> "QTableWidgetItem":
    """An editable results cell (used for the per-row thickness column)."""
    item = QTableWidgetItem(text)
    item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
                  | Qt.ItemFlag.ItemIsEditable)
    return item


def _cell_float(item: "QTableWidgetItem | None") -> float | None:
    """Parse a table cell's text as a finite float, else None."""
    if item is None:
        return None
    try:
        v = float(item.text())
    except (TypeError, ValueError):
        return None
    import math
    return v if math.isfinite(v) else None


def _fmt_env(value: Any) -> str:
    """Format an environmental SP/PV for display ('—' when unavailable)."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "—"
    import math

    return f"{f:.1f}" if math.isfinite(f) else "—"


class _DataStoreSelectionDialog(QDialog):
    """Filter and select DataStore measurements before loading/importing."""

    def __init__(self, store, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Browse DataStore Measurements")
        self.resize(980, 560)
        self._store = store
        self._rows: list[dict[str, Any]] = []
        self.action: str | None = None

        root = QVBoxLayout(self)

        filters = QHBoxLayout()
        filters.addWidget(QLabel("Run:"))
        self._combo_run = QComboBox()
        self._combo_run.setMinimumWidth(300)
        filters.addWidget(self._combo_run)

        filters.addWidget(QLabel("Channel:"))
        self._spin_channel = QSpinBox()
        self._spin_channel.setRange(0, 256)
        self._spin_channel.setValue(0)
        self._spin_channel.setToolTip("0 = all channels")
        filters.addWidget(self._spin_channel)

        filters.addWidget(QLabel("Limit:"))
        self._spin_limit = QSpinBox()
        self._spin_limit.setRange(10, 10000)
        self._spin_limit.setValue(300)
        filters.addWidget(self._spin_limit)

        self._btn_refresh = QPushButton("Refresh")
        self._btn_refresh.clicked.connect(self._reload_rows)
        filters.addWidget(self._btn_refresh)

        self._btn_select_all = QPushButton("Select All")
        self._btn_select_all.clicked.connect(self._select_all)
        filters.addWidget(self._btn_select_all)

        self._btn_clear_all = QPushButton("Clear All")
        self._btn_clear_all.clicked.connect(self._clear_all)
        filters.addWidget(self._btn_clear_all)
        filters.addStretch()
        root.addLayout(filters)

        # Copyable rather than plain: Shift-click fills the checkbox range from the
        # last plainly-clicked row, which is what turns "channels 12–34 of 300" from
        # 23 clicks into two, and Ctrl+C copies the visible rows to a notebook.
        self._table = CopyableTableWidget()
        # ☑, Timestamp, Run, Channel, Workflow, then the five environmental
        # SP/PVs captured at measurement time plus the resolved temperature and
        # the thermometer it came from, then the EIS file path. The resolved
        # pair is shown NEXT TO its sources rather than instead of them: the
        # operator's question "why is this row 85 °C" is answered by seeing all
        # three reads and the label side by side.
        self._env_headers = [
            "Stage SP (°C)", "Chamber PV (°C)", "Stage PV (°C)",
            "RH SP (%)", "RH PV (%)", "T resolved (°C)", "T source",
        ]
        headers = (
            ["☑", "Timestamp", "Run", "Channel", "Workflow"]
            + self._env_headers
            + ["EIS File"]
        )
        self._table.setColumnCount(len(headers))
        self._table.setHorizontalHeaderLabels(headers)
        # `selected_rows()` reads checkbox state, so a Shift-filled range is picked
        # up with no further wiring. `checkRangeToggled` is deliberately connected to
        # nothing: this dialog's per-checkbox click path refreshes nothing either —
        # the status label carries a candidate count set by `_reload_rows`, and both
        # action buttons validate on click rather than on selection change.
        self._table.checkable_column = 0
        self._table.setToolTip(
            "Click a checkbox, then Shift-click another to select the whole range.\n"
            "Ctrl+C copies the selected cells as spreadsheet-ready TSV.")
        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self._table.setColumnWidth(0, 32)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        for col in range(3, len(headers) - 1):
            hdr.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(len(headers) - 1, QHeaderView.ResizeMode.Stretch)
        self._eis_file_col = len(headers) - 1
        root.addWidget(self._table)

        actions = QHBoxLayout()
        self._lbl_status = QLabel("")
        actions.addWidget(self._lbl_status)
        actions.addStretch()

        self._btn_browser = QPushButton("Load Selected in Browser")
        self._btn_browser.clicked.connect(self._accept_browser)
        actions.addWidget(self._btn_browser)

        self._btn_import_fit = QPushButton("Import Selected to Fit & Export")
        self._btn_import_fit.clicked.connect(self._accept_fit_export)
        actions.addWidget(self._btn_import_fit)

        self._btn_cancel = QPushButton("Cancel")
        self._btn_cancel.clicked.connect(self.reject)
        actions.addWidget(self._btn_cancel)
        root.addLayout(actions)

        self._populate_run_filter()
        self._reload_rows()

    def selected_rows(self) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        for i, row in enumerate(self._rows):
            item = self._table.item(i, 0)
            if item is not None and item.checkState() == Qt.CheckState.Checked:
                selected.append(row)
        return selected

    def _populate_run_filter(self) -> None:
        self._combo_run.clear()
        self._combo_run.addItem("(all runs)", userData=None)
        try:
            runs = self._store.query_runs()
        except Exception:
            runs = []
        for run in runs[:1000]:
            run_id = str(run.get("run_id") or "")
            if run_id:
                self._combo_run.addItem(run_id, userData=run_id)

    def _reload_rows(self) -> None:
        run_id = self._combo_run.currentData()
        ch = int(self._spin_channel.value())
        channel = ch if ch > 0 else None
        limit = int(self._spin_limit.value())

        try:
            rows = self._store.query_measurements(
                run_id=run_id,
                channel=channel,
                limit=limit,
                descending=True,
            )
        except Exception as exc:
            self._rows = []
            self._table.setRowCount(0)
            self._lbl_status.setText(f"Query failed: {exc}")
            return

        self._rows = [r for r in rows if r.get("eis_file_path")]
        self._table.setRowCount(0)
        for row_data in self._rows:
            row = self._table.rowCount()
            self._table.insertRow(row)
            chk = QTableWidgetItem()
            chk.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
            chk.setCheckState(Qt.CheckState.Unchecked)
            self._table.setItem(row, 0, chk)
            self._table.setItem(row, 1, QTableWidgetItem(str(row_data.get("timestamp") or "")))
            self._table.setItem(row, 2, QTableWidgetItem(str(row_data.get("run_id") or "")))
            self._table.setItem(row, 3, QTableWidgetItem(str(row_data.get("channel") or "")))
            self._table.setItem(row, 4, QTableWidgetItem(str(row_data.get("workflow_name") or "")))
            env = self._env_for(row_data.get("measurement_id"))
            for offset, key in enumerate(
                # Order matches `self._env_headers` exactly: Stage SP, Chamber
                # PV (the air probe), Stage PV, RH SP, RH PV, then the resolved
                # temperature and its source (schema epoch 4 — read off the row,
                # not re-derived here). A key that stops matching the schema does
                # not raise here — `env.get` returns None and the column renders
                # blank under a correct-looking header — so the two tuples move
                # together or not at all.
                ("stage_temp_sp_C", "chamber_air_C", "stage_temp_pv_C",
                 "rh_sp_pct", "rh_pv_pct", "temperature_C", "temperature_source")
            ):
                value = env.get(key)
                # One TEXT column among the numbers: show the label as written
                # rather than letting the numeric formatter turn it into '—'.
                text = (value or "—") if isinstance(value, str) else _fmt_env(value)
                self._table.setItem(row, 5 + offset, QTableWidgetItem(text))
            self._table.setItem(
                row, self._eis_file_col,
                QTableWidgetItem(str(row_data.get("eis_file_path") or "")),
            )

        self._lbl_status.setText(f"{len(self._rows)} candidate measurement(s)")

    def _env_for(self, measurement_id: Any) -> dict[str, Any]:
        """Return the ``measurement``-stage conditions snapshot (latest) for a row."""
        if measurement_id is None:
            return {}
        try:
            snaps = self._store.query_conditions(
                measurement_id=int(measurement_id), stage="measurement"
            )
        except Exception:
            return {}
        return snaps[-1] if snaps else {}

    def _select_all(self) -> None:
        for row in range(self._table.rowCount()):
            item = self._table.item(row, 0)
            if item is not None:
                item.setCheckState(Qt.CheckState.Checked)

    def _clear_all(self) -> None:
        for row in range(self._table.rowCount()):
            item = self._table.item(row, 0)
            if item is not None:
                item.setCheckState(Qt.CheckState.Unchecked)

    def _accept_browser(self) -> None:
        if not self.selected_rows():
            QMessageBox.information(self, "No Selection", "Select at least one measurement.")
            return
        self.action = "browser"
        self.accept()

    def _accept_fit_export(self) -> None:
        if not self.selected_rows():
            QMessageBox.information(self, "No Selection", "Select at least one measurement.")
            return
        self.action = "fit_export"
        self.accept()


# ---------------------------------------------------------------------------
# Analysis Tab
# ---------------------------------------------------------------------------

class AnalysisTab(QWidget):
    """EIS data analysis and results management tab."""

    def __init__(
        self,
        manager: "InstrumentManager",
        parent: QWidget | None = None,
        *,
        data_store=None,
    ):
        super().__init__(parent)
        self._manager = manager
        self._data_store = data_store
        self._loaded: list[EISResult] = []
        self._fits: list[FitResult] = []
        #: channel -> run_id for browser-loaded entries, so a fitted row can
        #: look up the twin's recorded thickness (P7.6).
        self._run_id_by_channel: dict[int, str] = {}
        self._db_path = _DEFAULT_DB
        self._build_ui()

    # ── UI ───────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        self._tab_widget = QTabWidget()
        root.addWidget(self._tab_widget)

        # Sub-tab 1: existing analysis workflow
        self._analysis_tab = QWidget()
        self._tab_widget.addTab(self._analysis_tab, "Fit & Export")
        self._build_analysis_tab(self._analysis_tab)

        # Sub-tab 2: EIS Visualizer browser
        self._vis_container = QWidget()
        self._tab_widget.addTab(self._vis_container, "EIS Browser")
        self._build_visualizer_tab(self._vis_container)

        # Sub-tab 3: post-hoc Arrhenius analysis
        self._arrh_container = QWidget()
        self._tab_widget.addTab(self._arrh_container, "Arrhenius (Post-hoc)")
        self._build_arrhenius_posthoc_tab(self._arrh_container)

    def _build_visualizer_tab(self, parent: QWidget) -> None:
        """Embed EISVisualizerWidget in the EIS Browser sub-tab."""
        from softae.gui.widgets.eis_visualizer_widget import (
            EISVisualizerWidget,
            ListEISSource,
        )

        layout = QVBoxLayout(parent)
        layout.setContentsMargins(4, 4, 4, 4)

        # Toolbar: reload from DataStore / open standalone window
        bar = QHBoxLayout()
        self._btn_vis_reload = QPushButton("↻  Reload from DataStore")
        self._btn_vis_reload.clicked.connect(self._on_vis_reload)
        bar.addWidget(self._btn_vis_reload)

        self._btn_vis_popout = QPushButton("⤢  Pop Out Window")
        self._btn_vis_popout.clicked.connect(self._on_vis_popout)
        bar.addWidget(self._btn_vis_popout)
        bar.addStretch()
        layout.addLayout(bar)

        # Widget itself (starts empty). on_fit_saved persists browser-initiated
        # re-fits back to the DataStore (no-op when no store is connected).
        self._vis_source = ListEISSource([])
        self._vis_widget = EISVisualizerWidget(
            self._vis_source, parent=parent, on_fit_saved=self._on_browser_fit_saved
        )
        layout.addWidget(self._vis_widget)

    def _on_browser_fit_saved(self, entry: Any, L: float, t: float, w: float) -> None:
        """Persist a fit produced in the EIS Browser to the DataStore.

        Appends a new fit row (history-preserving) for the entry's measurement,
        recording the per-sample electrode geometry so σ is reproducible.

        The four ``arc_*`` columns populate from *fit* alone — ``record_fit``
        reads ``fit.arc_closure`` and never the report — so a browser-saved row
        carries the arc verdict with no ``report=`` passed here. A fit the
        browser hands over without an annotation simply lands as four NULLs.
        """
        if self._data_store is None:
            return
        mid = getattr(entry, "measurement_id", None)
        fit = getattr(entry, "fit", None)
        if mid is None or fit is None:
            logger.debug("browser_fit_not_persisted", reason="no measurement_id/fit")
            return
        try:
            self._data_store.record_fit(mid, fit, L_cm=L, t_cm=t, w_cm=w)
            logger.info("browser_fit_persisted", measurement_id=mid, model=fit.model_name)
        except Exception:
            logger.warning("browser_fit_persist_failed", measurement_id=mid, exc_info=True)

    def _on_vis_reload(self) -> None:
        """Browse DataStore rows, then explicitly load/import selected entries."""

        if self._data_store is None:
            QMessageBox.information(
                self, "No DataStore",
                "No project DataStore is connected.\n"
                "Load EIS files via 'Fit & Export' first, then use 'Reload from DataStore'.",
            )
            return

        dlg = _DataStoreSelectionDialog(self._data_store, parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        rows = dlg.selected_rows()
        if not rows:
            return

        entries, failed = self._entries_from_measurement_rows(rows)
        if not entries:
            QMessageBox.warning(
                self,
                "Load Failed",
                "Selected files could not be loaded from disk.",
            )
            return

        if dlg.action == "fit_export":
            self._import_entries_to_fit_export(entries)
            msg = f"Imported {len(entries)} measurement(s) to Fit & Export."
        else:
            from softae.gui.widgets.eis_visualizer_widget import ListEISSource

            self._vis_source = ListEISSource(entries)
            self._vis_widget.set_source(self._vis_source)
            self._vis_widget.refresh()
            msg = f"Loaded {len(entries)} measurement(s) in EIS Browser."

        if failed:
            msg += f"  Skipped {failed} unreadable file(s)."
        QMessageBox.information(self, "DataStore Selection", msg)

    def _entries_from_measurement_rows(self, rows: list[dict[str, Any]]) -> tuple[list[Any], int]:
        """Load selected DataStore measurement rows as EIS visualizer entries."""
        from softae.gui.widgets.eis_visualizer_widget import (
            EISEntry,
            _fitresult_from_fit_row,
        )

        entries: list[EISEntry] = []
        failed = 0
        for row in rows:
            path_str = str(row.get("eis_file_path") or "")
            if not path_str:
                failed += 1
                continue
            path = Path(path_str)
            if not path.is_absolute() and self._data_store is not None:
                path = self._data_store.project_dir / path

            try:
                eis = EISResult.load(str(path))
            except Exception:
                failed += 1
                continue

            run_id = str(row.get("run_id") or "")
            channel = int(row.get("channel") or eis.channel)
            label = f"Ch{channel:02d} — {run_id[:16]}"

            # Carry measurement_id (needed to persist a re-fit) and surface the
            # latest stored fit/geometry so prior work shows on load.
            mid = row.get("measurement_id")
            fit = sigma = geom = None
            if mid is not None and self._data_store is not None:
                try:
                    fit_rows = self._data_store.query_fits(measurement_id=int(mid))
                    if fit_rows:
                        fit, sigma, geom = _fitresult_from_fit_row(fit_rows[-1])
                except Exception:
                    logger.debug("browser_load_fit_failed", measurement_id=mid, exc_info=True)

            entries.append(
                EISEntry(
                    label=label,
                    eis=eis,
                    fit=fit,
                    sigma=sigma,
                    run_id=run_id or None,
                    measurement_id=int(mid) if mid is not None else None,
                    geometry=geom,
                )
            )
        return entries, failed

    def _recorded_thickness_cm(self, channel: int) -> float | None:
        """The twin's recorded dry-film thickness for *channel*, in **cm**.

        The results table works in cm (it feeds σ = L/(R·w·t)); the twin records
        µm. ``None`` when nothing was recorded, which the caller must treat as
        "fall back to the manual default" rather than as a zero thickness.

        A thickness whose ``deposit_area_mm2`` was never recorded also reads as
        nothing (P.11). This is the *second*, independent σ consumer — the campaign
        objective is the other — and both divide by the same unattributable number,
        so guarding only one would leave the table quietly using a thickness the
        campaign path had just refused.
        """
        if self._data_store is None:
            return None
        run_id = self._run_id_by_channel.get(int(channel))
        if not run_id:
            return None
        getter = getattr(self._data_store, "predicted_thickness_record", None)
        try:
            if callable(getter):
                record = getter(run_id, int(channel))
                if record is None:
                    return None
                if record.area_mm2 is None:
                    logger.warning("thickness_withheld_area_never_recorded",
                                   channel=int(channel), run_id=run_id)
                    return None
                um = record.um
            else:
                um = self._data_store.predicted_thickness_um(run_id, int(channel))
        except Exception:
            logger.debug("thickness_lookup_failed", channel=channel, exc_info=True)
            return None
        return (um * 1e-4) if um else None      # µm → cm

    def _import_entries_to_fit_export(self, entries: list[Any]) -> None:
        """Append selected entries into Fit & Export without implicit fitting."""
        # Remember which run each channel came from, so a fitted row can look up
        # the deposition twin's recorded thickness for it (P7.6). The fit worker
        # only hands back (fit, sigma, channel, filepath) — no run_id — so the
        # association has to be captured here, where it is still known.
        for e in entries:
            if getattr(e, "run_id", None):
                self._run_id_by_channel[int(e.eis.channel)] = str(e.run_id)
        self._loaded.extend(e.eis for e in entries)
        self._fits.clear()
        self._results_table.setRowCount(0)
        self._lbl_loaded.setText(f"{len(self._loaded)} file(s) loaded")
        self._update_plots()
        self._tab_widget.setCurrentWidget(self._analysis_tab)

    def _on_vis_popout(self) -> None:
        """Open the current EIS Visualizer source in a detached window."""
        from softae.gui.widgets.eis_visualizer_widget import EISVisualizerWindow

        win = EISVisualizerWindow(
            self._vis_source,
            title="EIS Browser — SoftAE",
            on_fit_saved=self._on_browser_fit_saved,
            parent=None,
        )
        win.show()
        # Keep a reference so the window isn't garbage-collected
        self._popout_windows: list = getattr(self, "_popout_windows", [])
        self._popout_windows.append(win)

    def _build_analysis_tab(self, parent: QWidget) -> None:
        layout = QVBoxLayout(parent)

        # --- Top: file load + circuit config ---
        top = QHBoxLayout()

        load_grp = QGroupBox("Load EIS Data")
        load_lay = QHBoxLayout(load_grp)
        self._btn_load = QPushButton("Load File(s)…")
        self._btn_load.clicked.connect(self._on_load_files)
        load_lay.addWidget(self._btn_load)
        self._lbl_loaded = QLabel("0 files loaded")
        load_lay.addWidget(self._lbl_loaded)
        top.addWidget(load_grp)

        fit_grp = QGroupBox("Circuit Fitting")
        fit_lay = QHBoxLayout(fit_grp)
        fit_lay.addWidget(QLabel("Model:"))
        self._combo_model = QComboBox()
        for name, info in CIRCUIT_MODELS.items():
            self._combo_model.addItem(f"{name} — {info['description']}", userData=name)
        fit_lay.addWidget(self._combo_model)

        self._btn_fit = QPushButton("Fit All")
        self._btn_fit.clicked.connect(self._on_fit_all)
        self._btn_fit.setStyleSheet("background-color: #2196F3; color: white;")
        fit_lay.addWidget(self._btn_fit)
        top.addWidget(fit_grp)

        geom_grp = QGroupBox("Electrode Geometry (cm)")
        geom_lay = QHBoxLayout(geom_grp)
        geom_lay.addWidget(QLabel("L_gap:"))
        self._spin_L = QDoubleSpinBox()
        self._spin_L.setRange(0.001, 10.0)
        self._spin_L.setDecimals(5)  # single-micron (0.0001 cm) resolution
        self._spin_L.setValue(0.2)
        self._spin_L.setToolTip(
            "Electrode gap — the separation conduction crosses.\n"
            "The 'L' of σ = L/(R·t·w); named for what it physically is.")
        geom_lay.addWidget(self._spin_L)

        geom_lay.addWidget(QLabel("t:"))
        self._spin_t = QDoubleSpinBox()
        self._spin_t.setRange(0.001, 10.0)
        self._spin_t.setDecimals(5)  # single-micron (0.0001 cm) resolution
        self._spin_t.setValue(0.175)
        geom_lay.addWidget(self._spin_t)

        geom_lay.addWidget(QLabel("L_stripe:"))
        self._spin_w = QDoubleSpinBox()
        self._spin_w.setRange(0.001, 10.0)
        self._spin_w.setDecimals(5)  # single-micron (0.0001 cm) resolution
        self._spin_w.setValue(0.2)
        self._spin_w.setToolTip(
            "Stripe length — the electrode extent along the gap.\n"
            "The 'w' of σ = L/(R·t·w). Named L_stripe rather than a width to keep\n"
            "it distinct from conductance G in the impedance algebra.")
        geom_lay.addWidget(self._spin_w)

        # The cell constant the three numbers above imply. Read-only, and the
        # cheapest way to make a placeholder thickness visible: t = 0.175 cm reads
        # K = 5.7 /cm against the 50-100 /cm this coplanar cell actually implies.
        self._lbl_K = QLabel()
        self._lbl_K.setToolTip(
            "Cell constant K = L_gap / (t × L_stripe), from the seed thickness.\n"
            "This coplanar geometry implies roughly 50–100 /cm for a real film;\n"
            "a value far below that means t is a placeholder, not a measurement.")
        geom_lay.addWidget(self._lbl_K)
        top.addWidget(geom_grp)

        self._spin_t.setToolTip(
            "Default film thickness, seeded into each new fit's editable t (cm) cell.\n"
            "Film thickness varies per sample, so edit a row's t to use its own value.")
        # L and w are board-defined (shared across rows) — a change rescales every
        # row's σ live. Thickness is per-row, so the t spin only seeds new fits;
        # 'Set t → selected' pushes it into the chosen rows.
        self._spin_L.valueChanged.connect(self._recompute_all_sigmas)
        self._spin_w.valueChanged.connect(self._recompute_all_sigmas)
        for _spin in (self._spin_L, self._spin_t, self._spin_w):
            _spin.valueChanged.connect(self._update_cell_constant_label)
        self._update_cell_constant_label()

        layout.addLayout(top)

        # --- Middle: Nyquist/Bode plots (left) | [results table + σ-vs-sample plot] (right) ---
        splitter = QSplitter(Qt.Orientation.Horizontal)
        # Never let a pane collapse to zero: a collapsed matplotlib canvas keeps a
        # stuck sizeHint and won't grow back ("irreversible shrink"). Non-collapsible
        # panes + an Expanding canvas with a height floor keep the plots auto-fitting.
        splitter.setChildrenCollapsible(False)

        # Left: Nyquist / Bode plot area
        plot_widget = QWidget()
        plot_lay = QVBoxLayout(plot_widget)
        plot_lay.setContentsMargins(0, 0, 0, 0)
        if _HAS_MPL:
            self._fig = Figure(figsize=(8, 4), tight_layout=True)
            self._canvas = FigureCanvasQTAgg(self._fig)
            self._canvas.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            self._canvas.setMinimumHeight(240)  # floor so it can't be squeezed flat
            plot_lay.addWidget(self._canvas)
        else:
            plot_lay.addWidget(QLabel("matplotlib not available — install matplotlib for plots"))
            self._fig = None
            self._canvas = None
        splitter.addWidget(plot_widget)

        # Right: a vertical split of [results table] over [conductivity-vs-sample plot].
        right_split = QSplitter(Qt.Orientation.Vertical)
        right_split.setChildrenCollapsible(False)

        table_widget = QWidget()
        table_lay = QVBoxLayout(table_widget)
        table_lay.setContentsMargins(0, 0, 0, 0)

        tbl_actions = QHBoxLayout()
        self._btn_remove_selected = QPushButton("Remove Selected")
        self._btn_remove_selected.clicked.connect(self._on_remove_selected)
        tbl_actions.addWidget(self._btn_remove_selected)
        self._btn_select_all = QPushButton("Select All")
        self._btn_select_all.clicked.connect(self._on_select_all)
        tbl_actions.addWidget(self._btn_select_all)
        self._btn_deselect_all = QPushButton("Deselect All")
        self._btn_deselect_all.clicked.connect(self._on_deselect_all)
        tbl_actions.addWidget(self._btn_deselect_all)
        self._btn_apply_t = QPushButton("Set t → selected")
        self._btn_apply_t.setToolTip(
            "Write the Electrode Geometry t (cm) value into every selected row.")
        self._btn_apply_t.clicked.connect(self._on_apply_thickness_to_selected)
        tbl_actions.addWidget(self._btn_apply_t)
        tbl_actions.addStretch()
        self._btn_clear_all_loaded = QPushButton("Clear All")
        self._btn_clear_all_loaded.clicked.connect(self._on_clear_all)
        tbl_actions.addWidget(self._btn_clear_all_loaded)
        table_lay.addLayout(tbl_actions)

        # Pasteable, not merely copyable: thicknesses are measured off-rig and
        # arrive as a spreadsheet column. Only `t (cm)` is editable, so a paste
        # physically cannot land in a fitted or derived column.
        self._results_table = PasteableTableWidget()
        self._results_table.setColumnCount(_RCOL_COUNT)
        self._results_table.setHorizontalHeaderLabels([
            "☑", "Sample", "Channel", "File", "Model", "R0 (Ω)", "R1 (Ω)",
            "t (cm)", "σ (S/cm)", "Status", "Gate", "Error",
        ])
        # Rectangular multi-cell selection so a region can be copied (Ctrl+C) to Excel.
        self._results_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._results_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
        self._results_table.setToolTip(
            "Select cells and press Ctrl+C to copy to a spreadsheet.\n"
            "Click a t (cm) cell and press Ctrl+V to paste a column of thicknesses\n"
            "down from that row — σ is recomputed for every row written.\n"
            "Edit a row's t (cm) to recompute its conductivity with that sample's thickness.")
        # Scroll rather than grow: a small height floor lets the user recoup vertical
        # span for the σ plot below and page through rows with the table's scrollbar.
        self._results_table.setMinimumHeight(120)
        hdr = self._results_table.horizontalHeader()
        hdr.setSectionResizeMode(_RCOL_CHECK, QHeaderView.ResizeMode.Fixed)
        self._results_table.setColumnWidth(_RCOL_CHECK, 28)
        hdr.setSectionResizeMode(_RCOL_SAMPLE, QHeaderView.ResizeMode.ResizeToContents)
        for col in range(2, _RCOL_COUNT):
            hdr.setSectionResizeMode(col, QHeaderView.ResizeMode.Stretch)
        # Recompute σ when a thickness cell is edited (and re-render on checkbox toggle).
        self._results_table.itemChanged.connect(self._on_table_item_changed)
        # Shift-click a checkbox to fill the range from the anchor row (one refresh).
        self._results_table.checkable_column = _RCOL_CHECK
        self._results_table.checkRangeToggled.connect(self._update_plots)
        # A paste writes with the table's signals blocked, so `itemChanged` — the
        # only thing that recomputes σ from a thickness — never fires. Without this
        # connection a paste leaves new t beside stale σ, and nothing looks wrong.
        self._results_table.pasteCompleted.connect(self._on_results_pasted)
        table_lay.addWidget(self._results_table)
        right_split.addWidget(table_widget)

        # Aggregated conductivity-vs-sample plot of the checked fit samples.
        sigma_widget = QWidget()
        sigma_lay = QVBoxLayout(sigma_widget)
        sigma_lay.setContentsMargins(0, 0, 0, 0)
        if _HAS_MPL:
            self._sigma_fig = Figure(figsize=(5, 2.4), tight_layout=True)
            self._sigma_canvas = FigureCanvasQTAgg(self._sigma_fig)
            self._sigma_canvas.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            self._sigma_canvas.setMinimumHeight(150)
            sigma_lay.addWidget(self._sigma_canvas)
        else:
            self._sigma_fig = None
            self._sigma_canvas = None
        right_split.addWidget(sigma_widget)
        right_split.setStretchFactor(0, 3)  # table gets the lion's share by default
        right_split.setStretchFactor(1, 2)  # σ plot underneath

        splitter.addWidget(right_split)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter, stretch=1)

        # --- Bottom: DB actions ---
        bottom = QHBoxLayout()

        self._btn_save_db = QPushButton("Save to Database")
        self._btn_save_db.clicked.connect(self._on_save_db)
        bottom.addWidget(self._btn_save_db)

        self._btn_browse_db = QPushButton("Browse Database")
        self._btn_browse_db.clicked.connect(self._on_browse_db)
        bottom.addWidget(self._btn_browse_db)

        self._btn_export_csv = QPushButton("Export CSV")
        self._btn_export_csv.clicked.connect(self._on_export_csv)
        bottom.addWidget(self._btn_export_csv)

        self._lbl_db_status = QLabel("")
        bottom.addWidget(self._lbl_db_status)
        bottom.addStretch()

        layout.addLayout(bottom)

    # ── File loading ─────────────────────────────────────────────────

    def _on_load_files(self) -> None:
        """Open file dialog to load EIS data files."""
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Load EIS Data Files", "",
            "EIS Data (*.txt *.csv *.dat);;All Files (*)",
        )
        if not paths:
            return
        for p in paths:
            try:
                result = EISResult.load(p)
                self._loaded.append(result)
            except Exception as exc:
                logger.warning("eis_load_error", path=p, error=str(exc))
        self._lbl_loaded.setText(f"{len(self._loaded)} file(s) loaded")
        self._update_plots()

    # ── Plotting ─────────────────────────────────────────────────────

    def _update_plots(self) -> None:
        """Redraw Nyquist and Bode plots, respecting checked row visibility."""
        # Keep the aggregated σ-vs-sample plot in sync with every refresh.
        self._update_sigma_plot()
        if self._fig is None or not self._loaded:
            return

        # Which loaded spectra are currently visible?
        n_rows = self._results_table.rowCount()
        if n_rows == len(self._loaded):
            # Table matches loaded list — honour checkbox state.
            visible = [
                i for i in range(n_rows)
                if (self._results_table.item(i, 0) is not None
                    and self._results_table.item(i, 0).checkState()
                    == Qt.CheckState.Checked)
            ]
            if not visible:
                visible = list(range(len(self._loaded)))  # all-off → show all
        else:
            visible = list(range(len(self._loaded)))  # table not in sync → show all

        # Left: Nyquist (full height). Right column: |Z| over Z'/-Z'' (shared x).
        self._fig.clear()
        gs = self._fig.add_gridspec(2, 2)
        ax_ny = self._fig.add_subplot(gs[:, 0])
        ax_mag = self._fig.add_subplot(gs[0, 1])
        ax_bode = self._fig.add_subplot(gs[1, 1], sharex=ax_mag)

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                  "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]

        for ci, i in enumerate(visible):
            r = self._loaded[i]
            label = f"Ch {r.channel}"
            color = colors[ci % len(colors)]
            ax_ny.plot(r.z_real, r.z_imag_neg, "o", label=label,
                       color=color, markersize=3)
            ax_mag.loglog(r.frequency, r.z_magnitude, "-",
                          label=label, color=color, alpha=0.8)
            ax_bode.loglog(r.frequency, r.z_real, "-",
                           label=f"{label} Z'", color=color, alpha=0.7)
            ax_bode.loglog(r.frequency, r.z_imag_neg, "--",
                           label=f"{label} -Z''", color=color, alpha=0.7)

        # ── Fit overlay when exactly one trace is visible ─────────────────────
        if len(visible) == 1:
            i = visible[0]
            if (i < len(self._fits)
                    and self._fits[i].success
                    and self._fits[i].z_fit is not None):
                zf = self._fits[i].z_fit
                ax_ny.plot(
                    zf.real, -zf.imag, "-",
                    color="#E69F00", linewidth=2, zorder=5,
                    label=f"Fit ({self._fits[i].model_name})",
                )

        ax_ny.set_xlabel("Z' (Ω)")
        ax_ny.set_ylabel("-Z'' (Ω)")
        ax_ny.set_title("Nyquist")
        ax_ny.set_aspect("equal", adjustable="datalim")
        ax_ny.grid(True, linestyle="--", alpha=0.5)
        ax_ny.legend(fontsize=7)

        # Top-right: |Z| only. Shares the frequency axis with the plot below, so
        # its own x tick labels are hidden to avoid duplication.
        ax_mag.set_ylabel("|Z| (Ω)")
        ax_mag.set_title("Bode — |Z|")
        ax_mag.grid(True, which="both", linestyle="--", alpha=0.5)
        ax_mag.tick_params(labelbottom=False)
        ax_mag.legend(fontsize=7)

        # Bottom-right: real / imaginary components (as before).
        ax_bode.set_xlabel("Frequency (Hz)")
        ax_bode.set_ylabel("Impedance (Ω)")
        ax_bode.set_title("Bode — Z′ / −Z″")
        ax_bode.grid(True, which="both", linestyle="--", alpha=0.5)
        ax_bode.legend(fontsize=7)

        self._canvas.draw_idle()

    def _update_sigma_plot(self) -> None:
        """Redraw the aggregated conductivity-vs-sample plot for the checked fits.

        One point per checked row that has a successful fit and a positive σ
        (σ = L/(R1·t·w), using the current electrode geometry).  When no row is
        checked, every row is considered — mirroring the Nyquist/Bode "all-off →
        show all" rule.
        """
        if self._sigma_fig is None:
            return
        n_rows = self._results_table.rowCount()
        checked = [
            i for i in range(n_rows)
            if (self._results_table.item(i, _RCOL_CHECK) is not None
                and self._results_table.item(i, _RCOL_CHECK).checkState()
                == Qt.CheckState.Checked)
        ]
        rows = checked if checked else list(range(n_rows))

        labels: list[str] = []
        sigmas: list[float] = []
        n_fit = 0
        for i in rows:
            if self._row_r1(i) is None:  # no successful fit / no R1
                continue
            n_fit += 1
            sigma = self._row_sigma(i)  # uses this row's own thickness
            if sigma is None:
                continue
            # Label by the unique sample ID (col 1) — channel alone can repeat.
            id_item = self._results_table.item(i, _RCOL_SAMPLE)
            label = (id_item.text() if id_item and id_item.text()
                     else _excel_column_name(i + 1))
            labels.append(label)
            sigmas.append(sigma)

        self._sigma_fig.clear()
        ax = self._sigma_fig.add_subplot(111)
        if sigmas:
            x = list(range(len(sigmas)))
            ax.plot(x, sigmas, "o-", color="#0072B2", markersize=6)
            ax.set_yscale("log")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=0)
        else:
            msg = ("No conductivities to plot — run 'Fit All' and check samples."
                   if n_fit == 0 else
                   "Selected fits have no positive σ — check each row's t (cm) and the L/w geometry.")
            ax.text(0.5, 0.5, msg, ha="center", va="center",
                    transform=ax.transAxes, color="#888888", fontsize=8)
        ax.set_xlabel("Sample")
        ax.set_ylabel("σ (S/cm)")
        ax.set_title("Conductivity vs sample")
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        self._sigma_fig.tight_layout()
        self._sigma_canvas.draw_idle()

    # ── Fitting ──────────────────────────────────────────────────────

    def _on_fit_all(self) -> None:
        """Fit the selected circuit model to all loaded EIS results (background thread)."""
        if not self._loaded:
            QMessageBox.information(self, "No Data", "Load EIS data files first.")
            return

        model_name = self._combo_model.currentData()
        L = self._spin_L.value()
        t = self._spin_t.value()
        w = self._spin_w.value()

        self._fits.clear()
        self._results_table.setRowCount(0)
        self._btn_fit.setEnabled(False)
        self._btn_fit.setText("Fitting…")

        worker = _FitAllWorker(self._loaded, model_name, L, t, w, parent=self)
        worker.finished.connect(self._on_fit_all_done)
        # Keep reference so GC doesn't collect the thread before it finishes.
        worker.finished.connect(worker.deleteLater)
        self._fit_worker = worker
        worker.start()

    def _on_fit_all_done(self, results: list) -> None:
        """Populate results table with data returned from the fit worker."""
        self._btn_fit.setEnabled(True)
        self._btn_fit.setText("Fit All")
        model_name = self._combo_model.currentData()
        t_seed = self._spin_t.value()  # per-row thickness default (editable per row)
        self._results_table.blockSignals(True)
        for fr, _sigma, channel, filepath in results:
            self._fits.append(fr)
            row = self._results_table.rowCount()
            self._results_table.insertRow(row)
            # col 0 — visibility checkbox (checked by default)
            chk = QTableWidgetItem()
            chk.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
            chk.setCheckState(Qt.CheckState.Checked)
            self._results_table.setItem(row, _RCOL_CHECK, chk)
            self._results_table.setItem(row, _RCOL_SAMPLE, _sample_id_item(row + 1))
            self._results_table.setItem(row, _RCOL_CHANNEL, _ro_item(str(channel)))
            self._results_table.setItem(row, _RCOL_FILE, _ro_item(Path(filepath).name))
            self._results_table.setItem(row, _RCOL_MODEL, _ro_item(model_name))
            self._results_table.setItem(row, _RCOL_R0, _ro_item(f"{fr.R0:.2f}"))
            self._results_table.setItem(row, _RCOL_R1, _ro_item(f"{fr.R1:.2f}"))
            # Editable per-row thickness; σ is (re)derived from it. Seeded from
            # the deposition twin's computed thickness for this channel when one
            # was recorded (P7.6), falling back to the manual default — so a
            # campaign-cast sample no longer needs its t typed in from memory.
            row_t = self._recorded_thickness_cm(channel)
            self._results_table.setItem(
                row, _RCOL_T, _editable_item(f"{(row_t if row_t else t_seed):g}"))
            self._results_table.setItem(row, _RCOL_SIGMA, _ro_item("—"))
            status_item = _ro_item("✓" if fr.success else "✗")
            if not fr.success:
                status_item.setForeground(Qt.GlobalColor.red)
            self._results_table.setItem(row, _RCOL_STATUS, status_item)
            self._results_table.setItem(row, _RCOL_GATE, _gate_item(fr))
            self._results_table.setItem(row, _RCOL_ERROR, _ro_item(fr.error_msg[:80]))
            self._recompute_row_sigma(row)
        self._results_table.blockSignals(False)
        self._update_plots()

    # ── Spectrum visibility toggles ──────────────────────────────────

    def _on_table_item_changed(self, item: "QTableWidgetItem") -> None:
        """React to a visibility toggle (col 0) or a per-row thickness edit."""
        col = item.column()
        if col == _RCOL_CHECK:
            self._update_plots()
        elif col == _RCOL_T:
            self._results_table.blockSignals(True)
            self._recompute_row_sigma(item.row())
            self._results_table.blockSignals(False)
            self._update_sigma_plot()

    # ── Conductivity (per-row geometry) ──────────────────────────────────

    def _row_r1(self, row: int) -> float | None:
        """Bulk resistance R1 for *row* — from the live fit, else the R1 cell."""
        if row < len(self._fits) and getattr(self._fits[row], "success", False):
            r1 = self._fits[row].R1
            return float(r1) if r1 and np.isfinite(r1) else None
        return _cell_float(self._results_table.item(row, _RCOL_R1))

    def _row_sigma(self, row: int) -> float | None:
        """σ for *row* from the shared cell constant — global L/w, the row's own t.

        Runs on every keystroke in the thickness column, so it builds the cell and
        divides and does nothing else: the fit is *not* re-run here.
        """
        r1 = self._row_r1(row)
        t = _cell_float(self._results_table.item(row, _RCOL_T))
        cell = gui_cell(self._spin_L.value(), t, self._spin_w.value())
        return cell_sigma(cell, r1)

    def _recompute_row_sigma(self, row: int) -> None:
        """Refresh a row's σ cell from its R1 and per-row thickness."""
        sigma = self._row_sigma(row)
        item = self._results_table.item(row, _RCOL_SIGMA)
        text = f"{sigma:.4e}" if sigma is not None else "—"
        if item is None:
            self._results_table.setItem(row, _RCOL_SIGMA, _ro_item(text))
        else:
            item.setText(text)

    def _update_cell_constant_label(self) -> None:
        """Show the cell constant the three geometry spins imply.

        Purely informational, and the cheapest available way to surface a defect
        that has been in this tab since it was written: the seed thickness defaults
        to 0.175 cm (1.75 mm), which is roughly ten times any drop-cast film and
        yields K ≈ 5.7 /cm where this coplanar geometry implies 50–100 /cm.

        The number is *shown*, not corrected. Changing the default silently would
        move every σ an operator has ever read off this table.
        """
        lbl = getattr(self, "_lbl_K", None)
        if lbl is None:
            return
        try:
            from softae.analysis.eis.geometry import CellConstant

            cell = CellConstant(
                L_gap_cm=float(self._spin_L.value()),
                L_stripe_cm=float(self._spin_w.value()),
                thickness_cm=float(self._spin_t.value()),
            )
            K = cell.K_per_cm
        except Exception:
            lbl.setText("")
            return

        if not (K == K):
            lbl.setText("K —")
            return
        lbl.setText(f"K = {K:.1f} /cm" + ("" if cell.plausible else "  ⚠"))
        lbl.setStyleSheet("" if cell.plausible else "color: #B36B00;")

    def _recompute_all_sigmas(self) -> None:
        """Recompute every row's σ (e.g. after an L or w change) and refresh the plot."""
        self._results_table.blockSignals(True)
        for row in range(self._results_table.rowCount()):
            self._recompute_row_sigma(row)
        self._results_table.blockSignals(False)
        self._update_sigma_plot()

    def _on_apply_thickness_to_selected(self) -> None:
        """Write the Electrode-Geometry t (cm) value into every selected row."""
        rows = {idx.row() for idx in self._results_table.selectedIndexes()}
        if not rows:
            QMessageBox.information(
                self, "No Selection", "Select one or more rows first.")
            return
        t_text = f"{self._spin_t.value():g}"
        self._results_table.blockSignals(True)
        for row in sorted(rows):
            item = self._results_table.item(row, _RCOL_T)
            if item is None:
                self._results_table.setItem(row, _RCOL_T, _editable_item(t_text))
            else:
                item.setText(t_text)
            self._recompute_row_sigma(row)
        self._results_table.blockSignals(False)
        self._update_sigma_plot()

    def _on_results_pasted(self, rows: list) -> None:
        """Recompute σ for every row a clipboard paste wrote, then refresh once.

        The paste itself runs with the table's signals blocked, so this is the only
        thing that keeps σ in step with a pasted thickness.

        Two outcomes are reported rather than left silent:

        * **Nothing written.** A single-column clipboard pastes into whatever column
          the cursor is in, and only ``t (cm)`` is editable — so a cursor anywhere
          else writes nothing at all. The paste is *not* redirected to the thickness
          column: silently relocating an operator's paste is worse than refusing it.
        * **Unparseable values.** A spreadsheet column often carries a header row or
          blanks. Those land as text, σ renders as '—' for them, and the count is
          named here. The text stays in the cell so the operator can see what landed.
        """
        if not rows:
            self._lbl_db_status.setText(
                "Paste wrote nothing — click a cell in the t (cm) column first.")
            return
        self._results_table.blockSignals(True)
        unparseable = 0
        for row in rows:
            if _cell_float(self._results_table.item(row, _RCOL_T)) is None:
                unparseable += 1
            self._recompute_row_sigma(row)
        self._results_table.blockSignals(False)
        self._update_sigma_plot()
        message = f"Pasted {len(rows)} thickness value(s)"
        if unparseable:
            message += f" — {unparseable} not numeric (σ shown as '—')"
        self._lbl_db_status.setText(message)

    def _on_select_all(self) -> None:
        self._results_table.blockSignals(True)
        for row in range(self._results_table.rowCount()):
            it = self._results_table.item(row, 0)
            if it is not None:
                it.setCheckState(Qt.CheckState.Checked)
        self._results_table.blockSignals(False)
        self._update_plots()

    def _on_deselect_all(self) -> None:
        self._results_table.blockSignals(True)
        for row in range(self._results_table.rowCount()):
            it = self._results_table.item(row, 0)
            if it is not None:
                it.setCheckState(Qt.CheckState.Unchecked)
        self._results_table.blockSignals(False)
        self._update_plots()

    # ── Table management ─────────────────────────────────────────────

    def _on_remove_selected(self) -> None:
        """Remove the rows currently selected in the results table and their
        corresponding loaded EIS entries."""
        selected_rows = sorted(
            {idx.row() for idx in self._results_table.selectedIndexes()},
            reverse=True,  # remove from bottom so indices stay valid
        )
        for row in selected_rows:
            self._results_table.removeRow(row)
            if row < len(self._loaded):
                del self._loaded[row]
            if row < len(self._fits):
                del self._fits[row]
        self._renumber_sample_ids()  # keep IDs contiguous (A, B, C, …) after removal
        self._lbl_loaded.setText(f"{len(self._loaded)} file(s) loaded")
        self._update_plots()

    def _renumber_sample_ids(self) -> None:
        """Reassign Excel-style sample IDs so they stay contiguous with row order."""
        self._results_table.blockSignals(True)
        for row in range(self._results_table.rowCount()):
            item = self._results_table.item(row, 1)
            if item is None:
                self._results_table.setItem(row, 1, _sample_id_item(row + 1))
            else:
                item.setText(_excel_column_name(row + 1))
        self._results_table.blockSignals(False)

    def _on_clear_all(self) -> None:
        """Remove all loaded EIS data, fit results, and table rows."""
        self._loaded.clear()
        self._fits.clear()
        self._results_table.setRowCount(0)
        self._lbl_loaded.setText("0 file(s) loaded")
        if self._fig is not None:
            self._fig.clear()
            self._canvas.draw_idle()
        self._update_sigma_plot()

    # ── Database ─────────────────────────────────────────────────────

    def _on_save_db(self) -> None:
        """Save current fit results to the SQLite database."""
        if not self._fits:
            QMessageBox.information(self, "No Fits", "Run circuit fitting first.")
            return

        try:
            conn = _ensure_db(self._db_path)
            for i, (eis, fr) in enumerate(zip(self._loaded, self._fits)):
                # Save σ computed with this row's own thickness (per-row t).
                sigma = self._row_sigma(i) if fr.success else None
                conn.execute(
                    """INSERT INTO eis_results
                       (timestamp, channel, model_name, R0, R1, sigma,
                        npts, f_min, f_max, file_path, params_json,
                        fit_success, fit_error)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        eis.timestamp.isoformat(),
                        eis.channel,
                        fr.model_name,
                        float(fr.R0),
                        float(fr.R1),
                        float(sigma) if sigma is not None and not np.isnan(sigma) else None,
                        eis.npts,
                        float(eis.frequency.min()) if eis.npts else None,
                        float(eis.frequency.max()) if eis.npts else None,
                        eis.raw_file_path or "",
                        json.dumps(eis.eis_params),
                        1 if fr.success else 0,
                        fr.error_msg,
                    ),
                )
            conn.commit()
            conn.close()
            n = len(self._fits)
            self._lbl_db_status.setText(f"Saved {n} result(s) to {self._db_path.name}")
        except Exception as exc:
            QMessageBox.warning(self, "Database Error", str(exc))

    def _on_browse_db(self) -> None:
        """Load and display all records from the database."""
        try:
            conn = _ensure_db(self._db_path)
            rows = conn.execute(
                "SELECT channel, model_name, R0, R1, sigma, file_path, "
                "fit_success, fit_error, timestamp FROM eis_results "
                "ORDER BY id DESC LIMIT 200"
            ).fetchall()
            conn.close()
        except Exception as exc:
            QMessageBox.warning(self, "Database Error", str(exc))
            return

        self._results_table.setRowCount(0)
        # DB view: σ derives from the table's stored R1 + per-row t (no live fits).
        self._fits.clear()
        t_seed = self._spin_t.value()
        self._results_table.blockSignals(True)
        for r in rows:
            row = self._results_table.rowCount()
            self._results_table.insertRow(row)
            chk = QTableWidgetItem()
            chk.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
            chk.setCheckState(Qt.CheckState.Checked)
            self._results_table.setItem(row, _RCOL_CHECK, chk)
            self._results_table.setItem(row, _RCOL_SAMPLE, _sample_id_item(row + 1))
            self._results_table.setItem(row, _RCOL_CHANNEL, _ro_item(str(r[0])))
            self._results_table.setItem(row, _RCOL_FILE, _ro_item(Path(r[5] or "").name))
            self._results_table.setItem(row, _RCOL_MODEL, _ro_item(str(r[1])))
            self._results_table.setItem(row, _RCOL_R0, _ro_item(f"{r[2]:.2f}" if r[2] else "—"))
            self._results_table.setItem(row, _RCOL_R1, _ro_item(f"{r[3]:.2f}" if r[3] else "—"))
            self._results_table.setItem(row, _RCOL_T, _editable_item(f"{t_seed:g}"))
            self._results_table.setItem(
                row, _RCOL_SIGMA, _ro_item(f"{r[4]:.4e}" if r[4] else "—"))
            self._results_table.setItem(row, _RCOL_STATUS, _ro_item("✓" if r[6] else "✗"))
            # The local eis_results table predates the gate columns, so a browsed
            # record simply has no gate history to show.
            self._results_table.setItem(row, _RCOL_GATE, _ro_item("—"))
            self._results_table.setItem(row, _RCOL_ERROR, _ro_item(str(r[7] or "")))
        self._results_table.blockSignals(False)

        self._lbl_db_status.setText(f"Showing {len(rows)} record(s) from database")

    # ── Export ────────────────────────────────────────────────────────

    def _on_export_csv(self) -> None:
        """Export the results table to CSV."""
        if self._results_table.rowCount() == 0:
            QMessageBox.information(self, "No Data", "No results to export.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Results CSV", "eis_analysis.csv", "CSV Files (*.csv)"
        )
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                headers = [
                    self._results_table.horizontalHeaderItem(c).text()
                    for c in range(self._results_table.columnCount())
                ]
                writer.writerow(headers)
                for r in range(self._results_table.rowCount()):
                    row = [
                        self._results_table.item(r, c).text()
                        if self._results_table.item(r, c) else ""
                        for c in range(self._results_table.columnCount())
                    ]
                    writer.writerow(row)
            QMessageBox.information(self, "Saved", f"Exported to {path}")
        except Exception as exc:
            QMessageBox.warning(self, "Export Error", str(exc))

    # =========================================================================
    # Arrhenius Post-hoc sub-tab
    # =========================================================================

    # -- Temperature / RH parsers ---------------------------------------------
    _T_PATTERNS = [
        re.compile(r"_T(\d+(?:\.\d+)?)(?:_RH|[^\d]|$)"),  # _T35_RH60, _T35C, _T35.0
        re.compile(r"[Tt]_?(\d+(?:\.\d+)?)"),               # T45, T_45, T45.0
        re.compile(r"(\d+(?:\.\d+)?)deg"),                  # 45deg
        re.compile(r"_(\d+(?:\.\d+)?)C"),                   # _45C
    ]
    _RH_PATTERN = re.compile(r"[Rr][Hh]_?(\d+(?:\.\d+)?)")

    @staticmethod
    def _guess_temperature(filename: str) -> float | None:
        """Try to parse a temperature (°C) from a filename.  Returns ``None`` on failure."""
        for pat in AnalysisTab._T_PATTERNS:
            m = pat.search(filename)
            if m:
                try:
                    return float(m.group(1))
                except ValueError:
                    pass
        return None

    @staticmethod
    def _guess_rh(filename: str) -> float | None:
        """Try to parse a relative-humidity (%) value from a filename.

        Recognises ``_RH35``, ``RH35``, ``rh35``, ``RH_35``, ``RH35.0``.
        Returns ``None`` on failure.
        """
        m = AnalysisTab._RH_PATTERN.search(filename)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
        return None

    @staticmethod
    def _natural_sort_key(path: str) -> list:
        """Key for natural (human) sort: split into alternating text/int chunks.

        E.g. ``"eis_ch10_T35.txt"`` sorts *after* ``"eis_ch9_T35.txt"``.
        """
        import re as _re
        fname = path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1].lower()
        return [
            int(chunk) if chunk.isdigit() else chunk
            for chunk in _re.split(r"(\d+)", fname)
        ]

    # -- UI builder -----------------------------------------------------------

    def _build_arrhenius_posthoc_tab(self, parent: QWidget) -> None:
        """Build the 'Arrhenius (Post-hoc)' sub-tab UI."""
        self._arrh_loaded: list[EISResult] = []
        self._arrh_results: list[ArrheniusResult] = []      # flat list for export
        self._arrh_results_keyed: list[tuple] = []          # (ch, rh_sp, ThermalResult)
        self._arrh_rh_results: dict[float, list] = {}       # rh_sp -> [ThermalResult] for 3D
        self._arrh_model: str = "arrhenius"                 # model of the last fit run
        # Per-file fit results: keyed by index in _arrh_loaded.
        # Populated by _arrh_on_fit() so the EIS preview can overlay the fit.
        self._arrh_fit_by_index: dict[int, FitResult] = {}
        # The σ each of those fits actually consumed, keyed the same way. Held
        # separately from the live geometry spins on purpose: the spins can be moved
        # after a fit, and the params panel must report what was fitted, not what
        # would be fitted now.
        self._arrh_sigma_by_index: dict[int, float] = {}

        root = QVBoxLayout(parent)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(6)

        # ── Top toolbar ──────────────────────────────────────────────────────
        toolbar = QHBoxLayout()

        load_grp = QGroupBox("EIS Files")
        load_lay = QHBoxLayout(load_grp)
        btn_load = QPushButton("Load File(s)…")
        btn_load.clicked.connect(self._arrh_on_load)
        load_lay.addWidget(btn_load)
        btn_load_dir = QPushButton("Load Folder…")
        btn_load_dir.clicked.connect(self._arrh_on_load_dir)
        load_lay.addWidget(btn_load_dir)
        self._arrh_lbl_count = QLabel("0 files")
        load_lay.addWidget(self._arrh_lbl_count)
        toolbar.addWidget(load_grp)

        model_grp = QGroupBox("Circuit Model")
        model_lay = QHBoxLayout(model_grp)
        self._arrh_combo_model = QComboBox()
        for name, info in CIRCUIT_MODELS.items():
            self._arrh_combo_model.addItem(f"{name} — {info['description']}", userData=name)
        model_lay.addWidget(self._arrh_combo_model)
        toolbar.addWidget(model_grp)

        thermal_grp = QGroupBox("σ(T) Model")
        thermal_lay = QHBoxLayout(thermal_grp)
        self._arrh_combo_thermal = QComboBox()
        self._arrh_combo_thermal.addItems(["arrhenius", "vft"])
        self._arrh_combo_thermal.setToolTip(
            "Temperature-dependence model fitted to σ(T):\n"
            "• arrhenius — σ = A·exp(−Eₐ/k_BT) (linear in 1/T; ≥ 2 temperatures)\n"
            "• vft — σ = A·exp(−B/(T−T₀)) (curved; ≥ 3 temperatures)"
        )
        thermal_lay.addWidget(self._arrh_combo_thermal)
        toolbar.addWidget(thermal_grp)

        geom_grp = QGroupBox("Electrode Geometry (cm)")
        geom_lay = QHBoxLayout(geom_grp)
        geom_lay.addWidget(QLabel("L:"))
        self._arrh_spin_L = QDoubleSpinBox()
        self._arrh_spin_L.setRange(0.001, 10.0)
        self._arrh_spin_L.setDecimals(5)  # single-micron (0.0001 cm) resolution
        self._arrh_spin_L.setValue(0.2)
        geom_lay.addWidget(self._arrh_spin_L)
        geom_lay.addWidget(QLabel("t:"))
        self._arrh_spin_t = QDoubleSpinBox()
        self._arrh_spin_t.setRange(0.001, 10.0)
        self._arrh_spin_t.setDecimals(5)  # single-micron (0.0001 cm) resolution
        self._arrh_spin_t.setValue(0.175)
        geom_lay.addWidget(self._arrh_spin_t)
        geom_lay.addWidget(QLabel("w:"))
        self._arrh_spin_w = QDoubleSpinBox()
        self._arrh_spin_w.setRange(0.001, 10.0)
        self._arrh_spin_w.setDecimals(5)  # single-micron (0.0001 cm) resolution
        self._arrh_spin_w.setValue(0.2)
        geom_lay.addWidget(self._arrh_spin_w)
        toolbar.addWidget(geom_grp)

        btn_fit = QPushButton("Fit Arrhenius")
        btn_fit.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        btn_fit.clicked.connect(self._arrh_on_fit)
        toolbar.addWidget(btn_fit)
        self._arrh_btn_fit = btn_fit

        btn_clear = QPushButton("Clear")
        btn_clear.clicked.connect(self._arrh_on_clear)
        toolbar.addWidget(btn_clear)

        btn_export = QPushButton("Export CSV")
        btn_export.clicked.connect(self._arrh_on_export)
        toolbar.addWidget(btn_export)

        self._arrh_btn_3d = QPushButton("3D Plot…")
        self._arrh_btn_3d.setToolTip(
            "Show per-channel 3D Arrhenius pop-out.\n"
            "Requires an RH column in the file table."
        )
        self._arrh_btn_3d.setEnabled(False)
        self._arrh_btn_3d.clicked.connect(self._arrh_on_3d_popout)
        toolbar.addWidget(self._arrh_btn_3d)

        toolbar.addStretch()
        root.addLayout(toolbar)

        # ── Progress row (hidden until fitting starts) ───────────────────────
        progress_row = QHBoxLayout()
        self._arrh_progress_bar = QProgressBar()
        self._arrh_progress_bar.setRange(0, 100)
        self._arrh_progress_bar.setValue(0)
        self._arrh_progress_bar.setTextVisible(True)
        self._arrh_progress_bar.setVisible(False)
        progress_row.addWidget(self._arrh_progress_bar, stretch=1)
        self._arrh_lbl_eta = QLabel("")
        self._arrh_lbl_eta.setVisible(False)
        progress_row.addWidget(self._arrh_lbl_eta)
        root.addLayout(progress_row)

        # ── Middle: file table | EIS plots ──────────────────────────────────
        mid_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: file–temperature assignment table
        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.addWidget(QLabel("Assign temperature to each file (edit T column as needed):"))
        self._arrh_file_table = QTableWidget()
        self._arrh_file_table.setColumnCount(5)
        self._arrh_file_table.setHorizontalHeaderLabels(
            ["File", "Channel", "T (°C)", "RH (%)", "Loaded"]
        )
        self._arrh_file_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        for col in (1, 2, 3, 4):
            self._arrh_file_table.horizontalHeader().setSectionResizeMode(
                col, QHeaderView.ResizeMode.ResizeToContents
            )
        left_lay.addWidget(self._arrh_file_table)
        mid_splitter.addWidget(left)

        # Center: fit parameter preview
        center = QWidget()
        center_lay = QVBoxLayout(center)
        center_lay.setContentsMargins(0, 0, 0, 0)
        self._arrh_params_lbl = QLabel("Fit parameters (run Fit Arrhenius, then click a row):")
        self._arrh_params_lbl.setWordWrap(True)
        center_lay.addWidget(self._arrh_params_lbl)
        self._arrh_params_table = QTableWidget()
        self._arrh_params_table.setColumnCount(2)
        self._arrh_params_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        self._arrh_params_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self._arrh_params_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self._arrh_params_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )
        self._arrh_params_table.setSelectionMode(
            QTableWidget.SelectionMode.NoSelection
        )
        center_lay.addWidget(self._arrh_params_table)
        mid_splitter.addWidget(center)

        # Right: Nyquist + Bode plots for selected file
        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.addWidget(QLabel("EIS preview (click row to inspect):"))
        if _HAS_MPL:
            self._arrh_eis_fig = Figure(figsize=(6, 4), tight_layout=True)
            self._arrh_eis_canvas = FigureCanvasQTAgg(self._arrh_eis_fig)
            right_lay.addWidget(self._arrh_eis_canvas)
        else:
            self._arrh_eis_fig = None
            self._arrh_eis_canvas = None
            right_lay.addWidget(QLabel("matplotlib not available"))
        mid_splitter.addWidget(right)

        mid_splitter.setStretchFactor(0, 5)   # file table
        mid_splitter.setStretchFactor(1, 3)   # fit parameters
        mid_splitter.setStretchFactor(2, 6)   # Nyquist + Bode

        # ── Bottom: Arrhenius plot | Ea results table ───────────────────────
        bot_splitter = QSplitter(Qt.Orientation.Horizontal)

        arrh_plot_widget = QWidget()
        arrh_plot_lay = QVBoxLayout(arrh_plot_widget)
        arrh_plot_lay.setContentsMargins(0, 0, 0, 0)
        arrh_plot_lay.addWidget(QLabel("Arrhenius plot — log₁₀(σ) vs 1000/T (K⁻¹):"))
        if _HAS_MPL:
            self._arrh_fig = Figure(tight_layout=True)
            self._arrh_canvas = FigureCanvasQTAgg(self._arrh_fig)
            self._arrh_canvas.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
            )
            self._arrh_canvas.setMinimumSize(0, 0)
            arrh_plot_lay.addWidget(self._arrh_canvas)
        else:
            self._arrh_fig = None
            self._arrh_canvas = None
            arrh_plot_lay.addWidget(QLabel("matplotlib not available"))
        bot_splitter.addWidget(arrh_plot_widget)

        ea_widget = QWidget()
        ea_lay = QVBoxLayout(ea_widget)
        ea_lay.setContentsMargins(0, 0, 0, 0)
        ea_lay.addWidget(QLabel("Activation energies per channel:"))
        self._arrh_ea_table = QTableWidget()
        self._arrh_ea_table.setColumnCount(7)
        self._arrh_ea_table.setHorizontalHeaderLabels(
            ["Channel", "RH (%)", "Eₐ (eV)", "Eₐ (kJ/mol)", "σ₀ (S/cm)", "R²", "n pts"]
        )
        self._arrh_ea_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        ea_lay.addWidget(self._arrh_ea_table)
        bot_splitter.addWidget(ea_widget)

        bot_splitter.setStretchFactor(0, 3)
        bot_splitter.setStretchFactor(1, 2)

        # ── Vertical sash: EIS table+preview (top) | Arrhenius+Ea (bottom) ──
        v_split = QSplitter(Qt.Orientation.Vertical)
        v_split.setHandleWidth(6)
        v_split.setChildrenCollapsible(False)
        v_split.addWidget(mid_splitter)
        v_split.addWidget(bot_splitter)
        v_split.setStretchFactor(0, 3)
        v_split.setStretchFactor(1, 2)
        root.addWidget(v_split)
        self._arrh_file_table.currentCellChanged.connect(self._arrh_on_row_selected)

    # -- Arrhenius post-hoc slots ---------------------------------------------

    def _arrh_on_load(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Load EIS Data Files", "",
            "EIS Data (*.txt *.csv *.dat);;All Files (*)",
        )
        self._arrh_add_files(paths)

    def _arrh_on_load_dir(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Folder with EIS Files")
        if not folder:
            return
        p = Path(folder)
        paths = [str(f) for ext in ("*.txt", "*.csv", "*.dat") for f in p.glob(ext)]
        self._arrh_add_files(paths)

    def _arrh_add_files(self, paths: list[str]) -> None:
        # Natural sort so ch10 follows ch9, not ch1
        paths = sorted(paths, key=self._natural_sort_key)
        for path in paths:
            try:
                result = EISResult.load(path)
                self._arrh_loaded.append(result)
                row = self._arrh_file_table.rowCount()
                self._arrh_file_table.insertRow(row)
                fname = Path(path).name
                self._arrh_file_table.setItem(row, 0, QTableWidgetItem(fname))
                ch_item = QTableWidgetItem(str(result.channel))
                ch_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self._arrh_file_table.setItem(row, 1, ch_item)
                # Prefer T_pv from header; fall back to filename guess
                import math as _math
                t_guess: float | None = None
                if not _math.isnan(result.T_pv):
                    t_guess = result.T_pv
                else:
                    t_guess = self._guess_temperature(fname)
                t_item = QTableWidgetItem("" if t_guess is None else f"{t_guess:.1f}")
                t_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self._arrh_file_table.setItem(row, 2, t_item)
                # Auto-fill RH from filename if present
                rh_guess = self._guess_rh(fname)
                rh_item = QTableWidgetItem("" if rh_guess is None else f"{rh_guess:.1f}")
                rh_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self._arrh_file_table.setItem(row, 3, rh_item)
                ok_item = QTableWidgetItem("✓")
                ok_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self._arrh_file_table.setItem(row, 4, ok_item)
            except Exception as exc:
                row = self._arrh_file_table.rowCount()
                self._arrh_file_table.insertRow(row)
                self._arrh_file_table.setItem(row, 0, QTableWidgetItem(Path(path).name))
                err_item = QTableWidgetItem(f"Error: {exc}")
                err_item.setForeground(Qt.GlobalColor.red)
                self._arrh_file_table.setItem(row, 4, err_item)
        self._arrh_lbl_count.setText(f"{len(self._arrh_loaded)} file(s)")

    def _arrh_on_row_selected(self, row: int, *_) -> None:
        """Update EIS preview and fit parameter panels when user clicks a file row."""
        if row < 0 or row >= len(self._arrh_loaded):
            return
        eis = self._arrh_loaded[row]

        # ── Fit parameter panel ──────────────────────────────────────────────
        self._arrh_params_table.setRowCount(0)
        fr = self._arrh_fit_by_index.get(row)
        if fr is None:
            self._arrh_params_lbl.setText(
                "Fit parameters — run \"Fit Arrhenius\" first, then click a row."
            )
        elif not fr.success:
            self._arrh_params_lbl.setText(f"Fit parameters \u2717 failed: {fr.error_msg}")
        else:
            self._arrh_params_lbl.setText(f"Fit parameters \u2713 {fr.model_name}")
            # Resolve parameter names from impedance library (best effort)
            try:
                from impedance.models.circuits import CustomCircuit  # type: ignore
                cfg = CIRCUIT_MODELS[fr.model_name]
                _m = CustomCircuit(cfg["circuit"], initial_guess=fr.parameters.tolist())
                # impedance >=0.9 exposes get_param_names(); older has param_names attr
                if hasattr(_m, "get_param_names"):
                    _pnames, _ = _m.get_param_names()
                else:
                    _pnames = list(getattr(_m, "param_names", []))
            except Exception:
                _pnames = []
            if not _pnames or len(_pnames) != len(fr.parameters):
                _pnames = [f"p[{i}]" for i in range(len(fr.parameters))]

            # The σ this fit consumed, not a recomputation from the live spins.
            # The worker built one cell at launch and the Arrhenius curve was fitted
            # against the σ that produced; dividing the stale R1 by whatever the
            # spins say *now* showed a number no fit had ever used.
            sigma = self._arrh_sigma_by_index.get(row, float("nan"))

            import math as _math
            highlight = Qt.GlobalColor.red if _math.isnan(sigma) or sigma <= 0 else None

            def _add(name: str, value: str, fg=None) -> None:
                r = self._arrh_params_table.rowCount()
                self._arrh_params_table.insertRow(r)
                n_item = QTableWidgetItem(name)
                v_item = QTableWidgetItem(value)
                v_item.setTextAlignment(
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                )
                if fg is not None:
                    v_item.setForeground(fg)
                self._arrh_params_table.setItem(r, 0, n_item)
                self._arrh_params_table.setItem(r, 1, v_item)

            _add("\u03c3  (S/cm)", f"{sigma:.4e}" if not _math.isnan(sigma) else "NaN",
                 fg=highlight)
            _add("R0  (\u03a9)", f"{fr.R0:.4e}")
            _add("R1  (\u03a9)", f"{fr.R1:.4e}")
            for name, val in zip(_pnames, fr.parameters):
                _add(name, f"{val:.4e}")

        # ── Nyquist + Bode preview ───────────────────────────────────────────
        if self._arrh_eis_fig is None:
            return
        self._arrh_eis_fig.clear()
        ax_ny = self._arrh_eis_fig.add_subplot(1, 2, 1)
        ax_bd = self._arrh_eis_fig.add_subplot(1, 2, 2)

        # Raw data
        ax_ny.plot(eis.z_real, eis.z_imag_neg, "b.", markersize=4, label="Measured")
        ax_bd.loglog(eis.frequency, eis.z_real, "b-", label="Z'")
        ax_bd.loglog(eis.frequency, eis.z_imag_neg, "b--", label="-Z''")

        # Overlay circuit fit if one exists for this row
        fr = self._arrh_fit_by_index.get(row)
        if fr is not None and fr.success:
            try:
                from softae.analysis.circuit_fitting import predict_fit_curve
                z_fit = predict_fit_curve(fr, eis.frequency)
                if z_fit is not None:
                    z_fit_real = np.real(z_fit)
                    z_fit_imag_neg = -np.imag(z_fit)
                    ax_ny.plot(z_fit_real, z_fit_imag_neg, "r-",
                               linewidth=1.5, label=f"Fit ({fr.model_name})")
                    ax_bd.loglog(eis.frequency, z_fit_real, "r-",
                                 linewidth=1.5, alpha=0.8, label="Z' fit")
                    ax_bd.loglog(eis.frequency, z_fit_imag_neg, "r--",
                                 linewidth=1.5, alpha=0.8, label="-Z'' fit")
            except Exception:
                pass

        ax_ny.set_xlabel("Z' (Ω)")
        ax_ny.set_ylabel("-Z'' (Ω)")
        ax_ny.set_title(f"Nyquist  Ch {eis.channel}")
        ax_ny.set_aspect("equal", adjustable="datalim")
        ax_ny.grid(True, linestyle="--", alpha=0.5)
        ax_ny.legend(fontsize=7)
        ax_bd.set_xlabel("Frequency (Hz)")
        ax_bd.set_ylabel("Impedance (Ω)")
        ax_bd.set_title("Bode")
        ax_bd.legend(fontsize=7)
        ax_bd.grid(True, which="both", linestyle="--", alpha=0.5)
        self._arrh_eis_canvas.draw_idle()

    def _arrh_on_fit(self) -> None:
        """Read temperature assignments, fit circuits, run Arrhenius (background thread)."""
        if not self._arrh_loaded:
            QMessageBox.information(self, "No Data", "Load EIS files first.")
            return

        model_name = self._arrh_combo_model.currentData()
        thermal_model = self._arrh_combo_thermal.currentText()
        L = self._arrh_spin_L.value()
        t = self._arrh_spin_t.value()
        w = self._arrh_spin_w.value()

        # Collect (loaded_index, eis, temperature, rh_sp_or_none) tuples for rows with valid T
        pairs: list[tuple] = []
        for i, eis in enumerate(self._arrh_loaded):
            t_item = self._arrh_file_table.item(i, 2)
            if t_item is None or not t_item.text().strip():
                continue
            try:
                T_C = float(t_item.text())
            except ValueError:
                continue
            rh_item = self._arrh_file_table.item(i, 3)
            rh_sp = None
            if rh_item is not None and rh_item.text().strip():
                try:
                    rh_sp = float(rh_item.text())
                except ValueError:
                    pass
            pairs.append((i, eis, T_C, rh_sp))

        if not pairs:
            QMessageBox.warning(
                self, "No Temperatures",
                "Enter at least 2 temperature values (°C) in the T column before fitting.",
            )
            return

        self._arrh_btn_fit.setEnabled(False)
        self._arrh_btn_fit.setText("Fitting…")
        self._arrh_progress_bar.setRange(0, len(pairs))
        self._arrh_progress_bar.setValue(0)
        self._arrh_progress_bar.setVisible(True)
        self._arrh_lbl_eta.setText("Starting…")
        self._arrh_lbl_eta.setVisible(True)

        self._arrh_model = thermal_model
        worker = _ArrhFitWorker(
            pairs, model_name, L, t, w, thermal_model=thermal_model, parent=self
        )
        worker.progress.connect(self._on_arrh_fit_progress)
        worker.finished.connect(self._on_arrh_fit_done)
        worker.finished.connect(worker.deleteLater)
        self._arrh_fit_worker = worker
        worker.start()

    def _on_arrh_fit_progress(self, current: int, total: int, elapsed_s: float) -> None:
        self._arrh_progress_bar.setValue(current)
        if current > 0 and elapsed_s > 0 and current < total:
            rate = elapsed_s / current
            remaining_s = rate * (total - current)
            self._arrh_lbl_eta.setText(f"~{remaining_s:.0f}s remaining  ({current}/{total})")
        elif current >= total:
            self._arrh_lbl_eta.setText(f"Fitting complete ({total}/{total})")
        else:
            self._arrh_lbl_eta.setText(f"{current}/{total}")

    def _on_arrh_fit_done(self, fit_by_index: dict, sigma_by_index: dict,
                          arrh_keyed: list, rh_results: dict) -> None:
        """Apply Arrhenius fit results from the background worker to the UI."""
        self._arrh_progress_bar.setVisible(False)
        self._arrh_lbl_eta.setVisible(False)
        self._arrh_btn_fit.setEnabled(True)
        self._arrh_btn_fit.setText("Fit Arrhenius")

        self._arrh_fit_by_index.clear()
        self._arrh_fit_by_index.update(fit_by_index)
        self._arrh_sigma_by_index.clear()
        self._arrh_sigma_by_index.update(sigma_by_index)
        self._arrh_results_keyed = list(arrh_keyed)  # (ch, rh_sp, ArrheniusResult)
        self._arrh_results = [r for _, _, r in arrh_keyed]  # flat list for export
        self._arrh_rh_results = dict(rh_results)

        self._arrh_update_ea_table()
        self._arrh_update_plot()

        # Enable 3D button only when ≥2 distinct non-nan RH values are fitted
        self._arrh_btn_3d.setEnabled(len(rh_results) >= 2)

    def _arrh_update_ea_table(self) -> None:
        import math as _math
        is_vft = self._arrh_model == "vft"
        # Columns 2 & 3 are model-specific; the rest are shared.
        if is_vft:
            self._arrh_ea_table.setHorizontalHeaderLabels(
                ["Channel", "RH (%)", "Eₐ (eV)", "T₀ (°C)", "σ∞ (S/cm)", "R²", "n pts"]
            )
        else:
            self._arrh_ea_table.setHorizontalHeaderLabels(
                ["Channel", "RH (%)", "Eₐ (eV)", "Eₐ (kJ/mol)", "σ₀ (S/cm)", "R²", "n pts"]
            )
        self._arrh_ea_table.setRowCount(0)
        for ch, rh_sp, res in self._arrh_results_keyed:
            row = self._arrh_ea_table.rowCount()
            self._arrh_ea_table.insertRow(row)
            self._arrh_ea_table.setItem(row, 0, QTableWidgetItem(str(ch)))
            rh_txt = f"{rh_sp:.1f}" if (rh_sp is not None and not _math.isnan(rh_sp)) else "—"
            self._arrh_ea_table.setItem(row, 1, QTableWidgetItem(rh_txt))
            ok = res.fit_success
            if is_vft:
                col2 = f"{res.Ea_eV:.4f}" if ok else "—"
                col3 = f"{res.T0_C:.1f}" if ok else "—"
            else:
                col2 = f"{res.Ea_eV:.4f}" if ok else "—"
                col3 = f"{res.Ea_kJ_per_mol:.2f}" if ok else "—"
            sigma0 = (
                f"{_math.exp(res.ln_A):.4e}"
                if ok and not _math.isnan(res.ln_A)
                else "—"
            )
            r2 = f"{res.R_squared:.4f}" if ok else "—"
            self._arrh_ea_table.setItem(row, 2, QTableWidgetItem(col2))
            self._arrh_ea_table.setItem(row, 3, QTableWidgetItem(col3))
            self._arrh_ea_table.setItem(row, 4, QTableWidgetItem(sigma0))
            self._arrh_ea_table.setItem(row, 5, QTableWidgetItem(r2))
            self._arrh_ea_table.setItem(row, 6, QTableWidgetItem(str(res.n_points)))

    def _arrh_update_plot(self) -> None:
        import math as _math
        if self._arrh_fig is None:
            return
        self._arrh_fig.clear()
        ax = self._arrh_fig.add_subplot(1, 1, 1)
        cmap = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
        linestyles = ["-", "--", ":", "-."]
        markers = ["o", "s", "^", "D", "v", "p", "h", "*"]
        all_channels = sorted({ch for ch, _, _ in self._arrh_results_keyed})
        all_rhs = sorted({
            rh for _, rh, _ in self._arrh_results_keyed
            if rh is not None and not _math.isnan(rh)
        })
        for ch, rh_sp, res in self._arrh_results_keyed:
            ch_idx = all_channels.index(ch)
            color = cmap[ch_idx % len(cmap)]
            marker = markers[ch_idx % len(markers)]
            if rh_sp is None or _math.isnan(rh_sp):
                ls = "-"
                series_label = f"Ch {ch}"
            else:
                rh_idx = all_rhs.index(rh_sp) if rh_sp in all_rhs else 0
                ls = linestyles[rh_idx % len(linestyles)]
                series_label = f"Ch {ch}  RH={rh_sp:.0f}%"
            temps_K = np.array(res.temperatures_C) + 273.15
            sigmas = np.array(res.conductivities)
            valid = np.isfinite(sigmas) & (sigmas > 0)
            if valid.any():
                x_data = 1000.0 / temps_K[valid]
                ax.scatter(x_data, np.log10(sigmas[valid]),
                           marker=marker, facecolors="none", edgecolors=color,
                           linewidths=1.5, s=40, zorder=3, label=series_label)
            if res.fit_success:
                T_range = np.linspace(res.T_min_C, res.T_max_C, 100) + 273.15
                x_fit = 1000.0 / T_range
                if getattr(res, "model", "arrhenius") == "vft":
                    # σ = σ∞·exp(−Eₐ/(k_B(T−T₀))); curved in 1000/T.
                    log10_sigma_fit = (
                        res.ln_A - res.B / (T_range - res.T0_K)
                    ) / np.log(10)
                    fit_label = (f"{series_label}  Eₐ={res.Ea_eV:.3f} eV, "
                                 f"T₀={res.T0_C:.0f} °C  R²={res.R_squared:.3f}")
                else:
                    log10_sigma_fit = (
                        res.ln_A - (res.Ea_eV / ArrheniusFitter.KB_EV) / T_range
                    ) / np.log(10)
                    fit_label = (f"{series_label}  Eₐ={res.Ea_eV:.3f} eV  "
                                 f"R²={res.R_squared:.3f}")
                ax.plot(x_fit, log10_sigma_fit, color=color, linestyle=ls,
                        label=fit_label, linewidth=1.5)
        ax.set_xlabel("1000 / T  (K⁻¹)")
        ax.set_ylabel("log₁₀(σ)  [σ in S/cm]")
        ax.set_title("VFT Plot" if self._arrh_model == "vft" else "Arrhenius Plot")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=7, loc="upper right")
        self._arrh_canvas.draw_idle()

    def _arrh_on_clear(self) -> None:
        self._arrh_loaded.clear()
        self._arrh_results.clear()
        self._arrh_results_keyed.clear()
        self._arrh_rh_results.clear()
        self._arrh_fit_by_index.clear()
        self._arrh_sigma_by_index.clear()
        self._arrh_file_table.setRowCount(0)
        self._arrh_ea_table.setRowCount(0)
        self._arrh_lbl_count.setText("0 files")
        self._arrh_btn_3d.setEnabled(False)
        if self._arrh_fig is not None:
            self._arrh_fig.clear()
            self._arrh_canvas.draw_idle()
        if self._arrh_eis_fig is not None:
            self._arrh_eis_fig.clear()
            self._arrh_eis_canvas.draw_idle()

    def _arrh_on_export(self) -> None:
        """Export Ea results to CSV."""
        if not self._arrh_results:
            QMessageBox.information(self, "No Results", "Run Fit Arrhenius first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Arrhenius Results", "arrhenius_posthoc.csv", "CSV Files (*.csv)"
        )
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                # Unified header covers both models; inapplicable cells are blank.
                writer.writerow([
                    "channel", "model", "Ea_eV", "Ea_kJ_per_mol",
                    "B_K", "T0_C", "ln_A", "R_squared",
                    "n_points", "T_min_C", "T_max_C", "fit_success",
                ])
                for res in self._arrh_results:
                    writer.writerow([
                        res.channel,
                        getattr(res, "model", "arrhenius"),
                        getattr(res, "Ea_eV", ""),
                        getattr(res, "Ea_kJ_per_mol", ""),
                        getattr(res, "B", ""),
                        getattr(res, "T0_C", ""),
                        res.ln_A,
                        res.R_squared,
                        res.n_points,
                        res.T_min_C,
                        res.T_max_C,
                        res.fit_success,
                    ])
            QMessageBox.information(self, "Saved", f"Exported to {path}")
        except Exception as exc:
            QMessageBox.warning(self, "Export Error", str(exc))

    def _arrh_on_3d_popout(self) -> None:
        """Open a per-channel 3D Arrhenius pop-out using pre-computed RH-grouped results.

        Uses ``_arrh_rh_results`` built by ``_ArrhFitWorker`` — no additional
        fitting is performed here (avoids blocking the main thread).
        """
        if not self._arrh_rh_results:
            QMessageBox.information(self, "No Results", "Run Fit Arrhenius first.")
            return

        rh_results = self._arrh_rh_results  # dict {rh_sp: [ArrheniusResult, ...]}

        if len(rh_results) < 2:
            QMessageBox.information(
                self, "Not Enough RH Points",
                "Need at least 2 distinct RH values to generate a 3D plot.",
            )
            return

        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            from PySide6.QtWidgets import QHBoxLayout, QScrollBar, QSizePolicy
            import numpy as np

            rh_vals = sorted(rh_results.keys())
            ch_ids  = sorted({r.channel for rlist in rh_results.values() for r in rlist})

            win = QWidget(None, Qt.WindowType.Window)
            win.setWindowTitle("RH-Arrhenius 3D (Post-hoc)")
            vlayout = QVBoxLayout(win)
            vlayout.setContentsMargins(6, 6, 6, 6)
            vlayout.setSpacing(4)

            fig3d   = Figure(tight_layout=True)
            canvas3d = FigureCanvasQTAgg(fig3d)
            canvas3d.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
            )
            vlayout.addWidget(canvas3d, stretch=1)

            nav_row = QHBoxLayout()
            lbl_ch  = QLabel()
            lbl_ch.setFixedWidth(140)
            nav_row.addWidget(lbl_ch)
            slider = QScrollBar(Qt.Orientation.Horizontal)
            slider.setRange(0, len(ch_ids) - 1)
            slider.setValue(0)
            slider.setSingleStep(1)
            slider.setPageStep(1)
            nav_row.addWidget(slider, stretch=1)
            vlayout.addLayout(nav_row)

            # Pre-compute per-channel arrays once so _render never iterates the
            # full results dict on every slider event.
            _ch_plot_data: dict[int, list[tuple]] = {}
            for ch_id in ch_ids:
                series: list[tuple] = []
                for rh_sp in rh_vals:
                    for res in rh_results[rh_sp]:
                        if res.channel != ch_id:
                            continue
                        temps_K = np.array(res.temperatures_C) + 273.15
                        sigmas  = np.array(res.conductivities)
                        valid   = np.isfinite(sigmas) & (sigmas > 0)
                        if valid.any():
                            series.append((
                                1000.0 / temps_K[valid],
                                np.full(valid.sum(), rh_sp),
                                np.log10(sigmas[valid]),
                                f"RH={rh_sp:.0f}%",
                            ))
                _ch_plot_data[ch_id] = series

            def _render(idx: int) -> None:
                ch = ch_ids[idx]
                lbl_ch.setText(f"  Channel {ch}  ({idx + 1}/{len(ch_ids)})")
                fig3d.clear()
                ax3 = fig3d.add_subplot(1, 1, 1, projection="3d")
                ax3.set_title(f"Channel {ch} — RH-Arrhenius", pad=10)
                ax3.set_xlabel("1000/T  (K\u207b\u00b9)", labelpad=8)
                ax3.set_ylabel("RH (%)", labelpad=8)
                ax3.set_zlabel("log\u2081\u2080(\u03c3)", labelpad=8)
                for xs, ys, zs, lbl in _ch_plot_data[ch]:
                    ax3.plot(xs, ys, zs, "o-", linewidth=1.5, markersize=4, label=lbl)
                if _ch_plot_data[ch]:
                    ax3.legend(fontsize=7, loc="best")
                canvas3d.draw_idle()

            # Debounce slider: collapse rapid drag events to one render after 150 ms idle.
            _debounce_timer = QTimer()
            _debounce_timer.setSingleShot(True)
            _debounce_timer.setInterval(150)
            _pending_idx: list[int] = [0]

            def _on_slider_changed(idx: int) -> None:
                _pending_idx[0] = idx
                _debounce_timer.start()

            _debounce_timer.timeout.connect(lambda: _render(_pending_idx[0]))
            slider.valueChanged.connect(_on_slider_changed)
            _render(0)
            win.resize(720, 580)
            win.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
            win.show()
            if not hasattr(self, "_arrh_3d_windows"):
                self._arrh_3d_windows: list = []
            self._arrh_3d_windows = [w for w in self._arrh_3d_windows if _widget_alive(w)]
            self._arrh_3d_windows.append(win)
        except Exception as exc:
            QMessageBox.warning(self, "3D Plot Error", str(exc))
